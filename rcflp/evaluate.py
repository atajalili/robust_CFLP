"""
rcflp.evaluate
--------------
Post-solution evaluation functions for the Robust CFLP.

These functions take fixed first-stage decisions x_jr and a fixed disruption
scenario epsilon_jh, then solve for the optimal recourse decisions (allocation y,
congestion c, auxiliary q) and compute the resulting cost breakdown.

Notation follows the paper throughout:
  - Λ_i   = demand[i]       potential arrival rate at node i
  - v_i   = value_max[i]    maximum willingness-to-pay at node i
  - μ_jr  = capacity[j,r]   service rate at facility j, level r
  - μ_j   = Σ_r μ_jr*x_jr  nominal capacity of open facility j
  - ε_jh  = epsilon_jh      binary disruption variable (1 if level h active at j)
  - ε_j   = Σ_h (1-h/(H-1))*ε_jh  scalar fraction of capacity remaining
  - p_i   = v_i*(1 - Σ_j y_ij)     optimal price (eq. 4)

Key functions
-------------
evaluate_second_stage   : solve inner SOCP (Corollary 1, eq. 15) for fixed x, ε
evaluate_fixed_price    : solve allocation problem with fixed prices (Remark 2, eq. 8)
compute_prices          : recover prices from allocation via eq. (4)
compute_cost_breakdown  : decompose objective into fixed / transport / revenue / congestion
compute_epsilon_scalar  : convert epsilon_jh dict to scalar ε_j per facility
sample_disruptions      : generate random ε ∈ Ξ for Monte Carlo evaluation
worst_case_disruption   : find ε* maximising recourse cost (wrapper for separation oracle)
"""

import time
import random

import gurobipy as gp
from gurobipy import GRB

from rcflp.subproblem import solve_subproblem_dual


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def compute_epsilon_scalar(epsilon_jh: dict, J: list, Hn: int) -> dict:
    """
    Convert binary disruption map ε_jh to scalar capacity fractions ε_j.

    ε_j = Σ_{h∈H} (1 - h/(H-1)) * ε_jh      (paper eq. 11 definition)

    Parameters
    ----------
    epsilon_jh : {(j,h): float}  disruption map (ε_jh values)
    J          : list of facility indices
    Hn         : number of disruption levels (H = {0, …, Hn-1})

    Returns
    -------
    {j: float}  ε_j ∈ [0,1] for each facility j
    """
    H = list(range(Hn))
    denom = max(Hn - 1, 1)  # guard against Hn=1
    return {
        j: sum((1 - h / denom) * epsilon_jh.get((j, h), 0.0) for h in H)
        for j in J
    }


def compute_prices(inst: dict, y_ij: dict) -> dict:
    """
    Recover optimal prices from allocation via eq. (4): p_i = v_i*(1 - Σ_j y_ij).

    At optimality the pricing constraint (2f) is binding, so prices are fully
    determined by the allocation decisions.

    Parameters
    ----------
    inst  : instance dict
    y_ij  : {(i,j): float}  optimal allocation fractions

    Returns
    -------
    {i: float}  price charged at each customer node
    """
    I = inst["I"]
    J = inst["J"]
    return {
        i: inst["value_max"][i] * (1.0 - sum(y_ij.get((i, j), 0.0) for j in J))
        for i in I
    }


def compute_cost_breakdown(
    inst: dict,
    x_jr: dict,
    y_ij: dict,
    c_j: dict,
    effective_demand: dict = None,
    fixed_prices: dict = None,
) -> dict:
    """
    Decompose total cost into interpretable components.

    In the adaptive-pricing (full recourse) case prices are derived from y_ij
    via eq. (4).  In the fixed-price case, supply fixed_prices explicitly.

    Parameters
    ----------
    inst             : instance dict
    x_jr             : {(j,r): float}  first-stage solution (binary)
    y_ij             : {(i,j): float}  allocation fractions
    c_j              : {j: float}      average work-in-process per facility
    effective_demand : {i: float}      optional effective demand rates λ_i
                       (= demand[i] if None; set to Λ_i*D_i(p̂_i) for fixed-price)
    fixed_prices     : {i: float}      optional fixed prices p̂_i
                       (derived from allocation via eq. (4) if None)

    Returns
    -------
    dict with keys:
        fixed_cost   – Σ_jr f_jr * x_jr
        transport    – Σ_ij λ_i * d_ij * y_ij
        revenue      – Σ_ij λ_i * p_i * y_ij  (p_i from eq. 4 or fixed)
        congestion   – Σ_j w_j * c_j
        total_cost   – fixed + transport - revenue + congestion  (= -profit)
        profit       – revenue - transport - fixed - congestion
        fill_rate    – {i: float}  Σ_j y_ij for each customer
        prices       – {i: float}  p_i per customer
    """
    I = inst["I"]
    J = inst["J"]
    R = inst["R"]

    lam = effective_demand if effective_demand is not None else inst["demand"]

    fill  = {i: sum(y_ij.get((i, j), 0.0) for j in J) for i in I}
    prices = fixed_prices if fixed_prices is not None else compute_prices(inst, y_ij)

    fixed    = sum(inst["fixed_cost"][j, r] * x_jr.get((j, r), 0.0) for j in J for r in R)
    transport = sum(lam[i] * inst["distance"][i, j] * y_ij.get((i, j), 0.0)
                    for i in I for j in J)
    revenue   = sum(lam[i] * prices[i] * fill[i] for i in I)
    congestion = sum(inst["congestion_cost"][j] * c_j.get(j, 0.0) for j in J)

    total_cost = fixed + transport - revenue + congestion

    return {
        "fixed_cost":  fixed,
        "transport":   transport,
        "revenue":     revenue,
        "congestion":  congestion,
        "total_cost":  total_cost,
        "profit":      -total_cost,
        "fill_rate":   fill,
        "prices":      prices,
    }


# ---------------------------------------------------------------------------
# Core evaluation: second-stage SOCP (Corollary 1, eq. 15)
# ---------------------------------------------------------------------------

def evaluate_second_stage(
    inst: dict,
    x_jr: dict,
    epsilon_jh: dict,
    Hn: int,
    time_limit: float = 300,
    verbose: bool = False,
) -> dict:
    """
    Solve the second-stage SOCP for fixed first-stage x and disruption ε.

    Implements Corollary 1 (eq. 15) of the paper.  Given fixed location-capacity
    decisions x_jr and a realized disruption ε_jh, optimises pricing and
    allocation decisions (y_ij, q_i, c_j) to maximise profit.

    The effective capacity of facility j after disruption is:
        μ̲_j = ε_j * Σ_r μ_jr * x_jr          (paper Section 1.2)

    where ε_j = Σ_h (1 - h/(H-1)) * ε_jh.

    Prices are not explicit decision variables; at optimality they are recovered
    from the allocation via eq. (4):  p_i = v_i * (1 - Σ_j y_ij).

    Parameters
    ----------
    inst        : instance dict from instancemaker()
    x_jr        : {(j,r): float}  fixed first-stage solution
    epsilon_jh  : {(j,h): float}  disruption scenario (binary values)
    Hn          : number of disruption levels (H = {0, …, Hn-1})
    time_limit  : Gurobi time limit (seconds)
    verbose     : print Gurobi output

    Returns
    -------
    dict with keys:
        y_ij      – {(i,j): float}  optimal allocation fractions
        c_j       – {j: float}      average work-in-process per facility
        q_i       – {i: float}      auxiliary squared fill-rate variable
        obj_val   – float           SOCP objective value (operational cost, no fixed)
        profit    – float           total profit = revenue - transport - fixed - congestion
        prices    – {i: float}      p_i = v_i*(1 - Σ_j y_ij) per customer (eq. 4)
        fill_rate – {i: float}      Σ_j y_ij per customer
        breakdown – dict            detailed cost components (see compute_cost_breakdown)
        eps_j     – {j: float}      scalar capacity fraction ε_j per facility
        mu_eff    – {j: float}      effective capacity μ̲_j = ε_j * μ_j per facility
        status    – int             Gurobi status code
        runtime   – float           wall-clock time (seconds)
    """
    I  = inst["I"]
    J  = inst["J"]
    R  = inst["R"]
    demand          = inst["demand"]
    capacity        = inst["capacity"]
    coeff1          = inst["coeff1"]   # Λ_i*(d_ij - v_i)
    coeff2          = inst["coeff2"]   # Λ_i*v_i
    congestion_cost = inst["congestion_cost"]

    # ε_j: scalar capacity fraction (Section 1.2)
    eps_j  = compute_epsilon_scalar(epsilon_jh, J, Hn)
    # μ_j: nominal capacity of open facility j
    mu_j   = {j: sum(capacity[j, r] * x_jr.get((j, r), 0.0) for r in R) for j in J}
    # μ̲_j: effective capacity after disruption
    mu_eff = {j: eps_j[j] * mu_j[j] for j in J}

    # Facilities that are both open and not fully disrupted
    active_j = [j for j in J if mu_eff[j] > 1e-9]

    m = gp.Model("eval_second_stage")
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit  = time_limit

    # --- decision variables ---
    # y_ij : fraction of potential demand from i served by facility j  (eq. 15)
    yy = m.addVars(I, J, lb=0.0, ub=1.0, name="y")
    # c_j  : average work-in-process at facility j  (M/M/1 queue length)
    CC = m.addVars(J, lb=0.0, name="C")
    # q_i  : auxiliary variable; q_i = (Σ_j y_ij)^2 at optimality
    QQ = m.addVars(I, lb=0.0, ub=1.0, name="Q")

    # Auxiliary variables for revenue SOCP (eq. 15d / 6g):
    #   ||[2*Σ_j y_ij, 1-q_i]||_2 ≤ 1+q_i
    VV1 = m.addVars(I, lb=0.0,          name="VV1")
    VV2 = m.addVars(I,                   name="VV2")
    VV3 = m.addVars(I, lb=0.0,          name="VV3")

    # Auxiliary variables for congestion SOCP (eq. 15c / 6f):
    #   ||[2λ_j, μ̲_j*c_j - μ̲_j]||_2 ≤ μ̲_j + μ̲_j*c_j - 2λ_j
    V1  = m.addVars(J,                   name="V1")
    V2  = m.addVars(J,                   name="V2")
    V3  = m.addVars(J, lb=0.0,          name="V3")

    # --- objective (eq. 15a): operational cost only (fixed cost is sunk) ---
    m.setObjective(
        gp.quicksum(coeff1[i, j] * yy[i, j] for i in I for j in J)
        + gp.quicksum(coeff2[i] * QQ[i] for i in I)
        + gp.quicksum(congestion_cost[j] * CC[j] for j in J),
        GRB.MINIMIZE,
    )

    # --- customer constraints ---
    for i in I:
        # (15b): Σ_j y_ij ≤ 1
        m.addConstr(
            gp.quicksum(yy[i, j] for j in J) <= 1.0,
            name=f"alloc_{i}"
        )
        # Revenue SOCP (15d): ||[2*Σ_j y_ij, 1-q_i]||_2 ≤ 1+q_i
        fill = gp.quicksum(yy[i, j] for j in J)
        m.addConstr(VV1[i] == 2.0 * fill,   name=f"vv1_{i}")
        m.addConstr(VV2[i] == 1.0 - QQ[i],  name=f"vv2_{i}")
        m.addConstr(VV3[i] == 1.0 + QQ[i],  name=f"vv3_{i}")
        m.addQConstr(VV1[i] ** 2 + VV2[i] ** 2 <= VV3[i] ** 2, name=f"rev_socp_{i}")

        # Allocation only to active (non-disrupted) facilities
        for j in J:
            if j not in active_j:
                m.addConstr(yy[i, j] == 0.0, name=f"no_alloc_{i}_{j}")

    # --- facility constraints ---
    for j in J:
        if j in active_j:
            M   = mu_eff[j]   # μ̲_j
            lam = gp.quicksum(demand[i] * yy[i, j] for i in I)

            # Capacity constraint: Σ_i Λ_i*y_ij ≤ ε_j*μ_j  (guarantees queue stability)
            m.addConstr(lam <= M, name=f"cap_{j}")

            # Congestion SOCP (15c):
            #   ||[2λ_j, μ̲_j*c_j - μ̲_j]||_2 ≤ μ̲_j + μ̲_j*c_j - 2λ_j
            m.addConstr(V1[j] == 2.0 * lam,                   name=f"v1_{j}")
            m.addConstr(V2[j] == M * CC[j] - M,               name=f"v2_{j}")
            m.addConstr(V3[j] == M + M * CC[j] - 2.0 * lam,  name=f"v3_{j}")
            m.addQConstr(V1[j] ** 2 + V2[j] ** 2 <= V3[j] ** 2, name=f"cong_socp_{j}")
        else:
            # Facility closed or fully disrupted: no allocation, zero congestion
            for i in I:
                m.addConstr(yy[i, j] == 0.0)
            m.addConstr(CC[j] == 0.0)
            m.addConstr(V1[j] == 0.0)
            m.addConstr(V2[j] == 0.0)
            m.addConstr(V3[j] == 0.0)

    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    y_sol = {(i, j): yy[i, j].x for i in I for j in J}
    c_sol = {j: CC[j].x for j in J}
    q_sol = {i: QQ[i].x for i in I}

    prices    = compute_prices(inst, y_sol)
    fill_rate = {i: sum(y_sol[i, j] for j in J) for i in I}
    breakdown = compute_cost_breakdown(inst, x_jr, y_sol, c_sol)

    return {
        "y_ij":      y_sol,
        "c_j":       c_sol,
        "q_i":       q_sol,
        "obj_val":   m.ObjVal,
        "profit":    breakdown["profit"],
        "prices":    prices,
        "fill_rate": fill_rate,
        "breakdown": breakdown,
        "eps_j":     eps_j,
        "mu_eff":    mu_eff,
        "status":    m.Status,
        "runtime":   runtime,
    }


# ---------------------------------------------------------------------------
# Fixed-price evaluation (Remark 2, eq. 8): Task 4b
# ---------------------------------------------------------------------------

def evaluate_fixed_price(
    inst: dict,
    x_jr: dict,
    prices_fixed: dict,
    epsilon_jh: dict,
    Hn: int,
    time_limit: float = 300,
    verbose: bool = False,
) -> dict:
    """
    Solve the allocation SOCP with fixed prices under disruption (Remark 2, eq. 8).

    This models the scenario where the provider cannot adjust prices after a
    disruption (price flexibility is restricted), but can still re-optimise
    the allocation of customers to facilities.

    The effective (materialised) demand rate at node i is:
        λ_i = Λ_i * D_i(p̂_i) = Λ_i * (v_i - p̂_i) / v_i

    which, when p̂_i = v_i*(1 - fill_i^nom) (nominal optimal price, eq. 4),
    simplifies to  λ_i = Λ_i * fill_i^nom.

    Parameters
    ----------
    inst          : instance dict from instancemaker()
    x_jr          : {(j,r): float}  fixed first-stage solution
    prices_fixed  : {i: float}      fixed prices p̂_i (e.g. from nominal solution)
    epsilon_jh    : {(j,h): float}  disruption scenario (binary values)
    Hn            : number of disruption levels (H = {0, …, Hn-1})
    time_limit    : Gurobi time limit (seconds)
    verbose       : print Gurobi output

    Returns
    -------
    dict with keys:
        y_ij           – {(i,j): float}  allocation (fraction of λ_i served by j)
        c_j            – {j: float}      average work-in-process per facility
        obj_val        – float           SOCP objective value (no fixed cost)
        profit         – float           total profit (revenue - transport - fixed - congestion)
        effective_demand – {i: float}   λ_i = Λ_i * D_i(p̂_i)
        demand_served  – {i: float}     actual demand served = λ_i * Σ_j y_ij
        fill_rate      – {i: float}     Σ_j y_ij (fraction of effective demand served)
        breakdown      – dict           cost breakdown
        eps_j          – {j: float}     ε_j per facility
        mu_eff         – {j: float}     μ̲_j per facility
        status         – int            Gurobi status code
        runtime        – float          wall-clock time (seconds)
    """
    I  = inst["I"]
    J  = inst["J"]
    R  = inst["R"]
    demand          = inst["demand"]
    capacity        = inst["capacity"]
    distance        = inst["distance"]
    value_max       = inst["value_max"]
    congestion_cost = inst["congestion_cost"]

    # Effective (materialised) demand rate: λ_i = Λ_i * D_i(p̂_i) = Λ_i*(v_i-p̂_i)/v_i
    lambda_i = {
        i: demand[i] * max(0.0, (value_max[i] - prices_fixed[i]) / value_max[i])
        for i in I
    }

    # ε_j and μ̲_j (same as evaluate_second_stage)
    eps_j  = compute_epsilon_scalar(epsilon_jh, J, Hn)
    mu_j   = {j: sum(capacity[j, r] * x_jr.get((j, r), 0.0) for r in R) for j in J}
    mu_eff = {j: eps_j[j] * mu_j[j] for j in J}
    active_j = [j for j in J if mu_eff[j] > 1e-9]

    m = gp.Model("eval_fixed_price")
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit  = time_limit

    # y_ij: fraction of EFFECTIVE demand λ_i from node i served by facility j
    yy = m.addVars(I, J, lb=0.0, ub=1.0, name="y")
    # c_j: average work-in-process at facility j
    CC = m.addVars(J, lb=0.0, name="C")

    # Auxiliary variables for congestion SOCP (eq. 8b, same structure as 15c
    # but with effective demand λ_i instead of Λ_i):
    #   ||[2λ_j, μ̲_j*c_j - μ̲_j]||_2 ≤ μ̲_j + μ̲_j*c_j - 2λ_j
    V1 = m.addVars(J, name="V1")
    V2 = m.addVars(J, name="V2")
    V3 = m.addVars(J, lb=0.0, name="V3")

    # --- objective (eq. 8a, operational part): minimize transport - revenue + congestion ---
    # Revenue per unit of allocated demand: p̂_i per unit of λ_i served
    # Net cost per unit: d_ij - p̂_i
    m.setObjective(
        gp.quicksum(
            lambda_i[i] * (distance[i, j] - prices_fixed[i]) * yy[i, j]
            for i in I for j in J
        )
        + gp.quicksum(congestion_cost[j] * CC[j] for j in J),
        GRB.MINIMIZE,
    )

    # --- customer constraints ---
    for i in I:
        # Σ_j y_ij ≤ 1  (fraction of effective demand served ≤ 100%)
        m.addConstr(gp.quicksum(yy[i, j] for j in J) <= 1.0, name=f"alloc_{i}")

        for j in J:
            if j not in active_j:
                m.addConstr(yy[i, j] == 0.0)

    # --- facility constraints ---
    for j in J:
        if j in active_j:
            M    = mu_eff[j]
            # Total demand load at facility j: Σ_i λ_i * y_ij
            load = gp.quicksum(lambda_i[i] * yy[i, j] for i in I)

            # Capacity (queue stability): Σ_i λ_i*y_ij ≤ μ̲_j
            m.addConstr(load <= M, name=f"cap_{j}")

            # Congestion SOCP (eq. 8b):
            #   ||[2*load, μ̲_j*c_j - μ̲_j]||_2 ≤ μ̲_j + μ̲_j*c_j - 2*load
            m.addConstr(V1[j] == 2.0 * load,                  name=f"v1_{j}")
            m.addConstr(V2[j] == M * CC[j] - M,               name=f"v2_{j}")
            m.addConstr(V3[j] == M + M * CC[j] - 2.0 * load,  name=f"v3_{j}")
            m.addQConstr(V1[j] ** 2 + V2[j] ** 2 <= V3[j] ** 2, name=f"cong_socp_{j}")
        else:
            for i in I:
                m.addConstr(yy[i, j] == 0.0)
            m.addConstr(CC[j] == 0.0)
            m.addConstr(V1[j] == 0.0)
            m.addConstr(V2[j] == 0.0)
            m.addConstr(V3[j] == 0.0)

    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    y_sol = {(i, j): yy[i, j].x for i in I for j in J}
    c_sol = {j: CC[j].x for j in J}

    fill_rate      = {i: sum(y_sol[i, j] for j in J) for i in I}
    demand_served  = {i: lambda_i[i] * fill_rate[i] for i in I}

    # Cost breakdown using effective demand and fixed prices
    breakdown = compute_cost_breakdown(
        inst, x_jr, y_sol, c_sol,
        effective_demand=lambda_i,
        fixed_prices=prices_fixed,
    )

    return {
        "y_ij":             y_sol,
        "c_j":              c_sol,
        "obj_val":          m.ObjVal,
        "profit":           breakdown["profit"],
        "effective_demand": lambda_i,
        "demand_served":    demand_served,
        "fill_rate":        fill_rate,
        "breakdown":        breakdown,
        "eps_j":            eps_j,
        "mu_eff":           mu_eff,
        "status":           m.Status,
        "runtime":          runtime,
    }


# ---------------------------------------------------------------------------
# Worst-case disruption (wrapper for separation oracle)
# ---------------------------------------------------------------------------

def worst_case_disruption(
    inst: dict,
    x_jr: dict,
    Gamma: float,
    Hn: int,
    big_M: float = 10_000,
    time_limit: float = 3600,
) -> tuple:
    """
    Find the worst-case disruption ε* for given first-stage solution x.

    Solves the recourse problem (Corollary 2, eq. 17) via the separation oracle.
    This is the adversary's maximisation over the uncertainty set Ξ.

    Parameters
    ----------
    inst       : instance dict
    x_jr       : {(j,r): float}  fixed first-stage solution
    Gamma      : uncertainty budget Γ
    Hn         : number of disruption levels
    big_M      : big-M for bilinear linearisation in subproblem
    time_limit : Gurobi time limit (seconds)

    Returns
    -------
    (epsilon_jh, recourse_cost)
        epsilon_jh   : {(j,h): float}  worst-case disruption scenario ε*
        recourse_cost: float           recourse cost under ε* (excl. fixed cost)
    """
    eps_list, obj_val = solve_subproblem_dual(
        x_jr, inst, Gamma, Hn,
        big_M=big_M,
        time_limit=time_limit,
        return_duals=False,
    )
    return eps_list[0], obj_val


# ---------------------------------------------------------------------------
# Monte Carlo: sample random disruptions ε ∈ Ξ
# ---------------------------------------------------------------------------

def sample_disruptions(
    inst: dict,
    Gamma: float,
    Hn: int,
    n_samples: int,
    seed: int = 42,
) -> list:
    """
    Generate random disruption scenarios ε ∈ Ξ for average-case evaluation.

    Each sample satisfies the uncertainty set definition (paper eq. 11):
        Σ_j Σ_h (h/(H-1)) * ε_jh ≤ Γ,   Σ_h ε_jh = 1  ∀j

    Sampling procedure: for each sample, randomly assign disruption levels h_j
    to facilities by sequentially drawing from a random permutation until the
    budget Γ is exhausted.  Facilities not assigned a positive level default
    to h=0 (no disruption).

    Parameters
    ----------
    inst      : instance dict
    Gamma     : uncertainty budget Γ
    Hn        : number of disruption levels (H = {0, …, Hn-1})
    n_samples : number of scenarios to generate
    seed      : random seed for reproducibility

    Returns
    -------
    list of {(j,h): float}  — one epsilon_jh dict per sample
    """
    J     = inst["J"]
    H     = list(range(Hn))
    denom = max(Hn - 1, 1)
    rng   = random.Random(seed)

    samples = []
    for _ in range(n_samples):
        h_j = {j: 0 for j in J}               # default: no disruption
        budget_left = Gamma

        facilities = list(J)
        rng.shuffle(facilities)

        for j in facilities:
            if budget_left < 1.0 / denom:     # no meaningful budget remains
                break
            # Maximum disruption level this facility can absorb given remaining budget
            max_h = min(Hn - 1, int(budget_left * denom))
            if max_h >= 1:
                h_j[j] = rng.randint(1, max_h)
                budget_left -= h_j[j] / denom

        # Convert to epsilon_jh format (binary)
        eps = {(j, h): (1.0 if h == h_j[j] else 0.0) for j in J for h in H}
        samples.append(eps)

    return samples


# ---------------------------------------------------------------------------
# Convenience: no-disruption scenario
# ---------------------------------------------------------------------------

def no_disruption_scenario(inst: dict, Hn: int) -> dict:
    """
    Return the epsilon_jh dict for the no-disruption scenario (ε_j = 1 for all j).

    This corresponds to h=0 for every facility, i.e. full capacity operational.

    Parameters
    ----------
    inst : instance dict
    Hn   : number of disruption levels

    Returns
    -------
    {(j,h): float}  with ε_j0 = 1 and ε_jh = 0 for h > 0
    """
    J = inst["J"]
    H = list(range(Hn))
    return {(j, h): (1.0 if h == 0 else 0.0) for j in J for h in H}
