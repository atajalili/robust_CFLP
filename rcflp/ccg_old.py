"""
rcflp.ccg
---------
Column-and-Constraint Generation (C&CG) algorithm.

At each iteration:
  1. Solve the subproblem (separation) to find the worst-case disruption ε*
     and update the upper bound.
  2. Add a new block of primal recourse variables (y, C, Q) and the
     corresponding SOCP constraints for scenario ε* to the master problem.
  3. Solve the master problem and update the lower bound.
  4. Terminate when the relative gap ≤ tol or time limit is reached.

The master problem grows by one full scenario block per iteration.
This is an iterative (non-recursive) implementation.
"""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from rcflp.subproblem import solve_subproblem_dual


def solve_CCG(
    inst: dict,
    uncertainty_budget: float,
    Hn: int,
    x_init: dict,
    tol: float = 0.01,
    big_M: float = 10_000,
    time_limit: float = 6 * 3600,
    master_mip_gap: float = 0.015,
    master_time_limit: float = 2000,
    verbose: bool = False,
) -> dict:
    """
    Solve the robust CFLP with the C&CG algorithm.

    Parameters
    ----------
    inst               : instance dict from instancemaker()
    uncertainty_budget : Γ
    Hn                 : number of disruption levels
    x_init             : {(j,r): float}  warm-start (e.g. nominal solution)
    tol                : relative optimality gap tolerance for termination
    big_M              : big-M for the subproblem bilinear linearisation
    time_limit         : total wall-clock time limit (seconds)
    master_mip_gap     : MIPGap used when outer gap is small (≤5%); looser
                         gaps are used automatically in early iterations
                         (10% when gap>20%, 5% when gap>5%)
    master_time_limit  : time limit for each individual master solve
    verbose            : print iteration log

    Returns
    -------
    dict with keys:
        x_jr        – {(j,r): float}  optimal first-stage solution
        LB          – float           final lower bound  (minimisation obj)
        UB          – float           final upper bound  (minimisation obj)
        profit_LB   – float           = -UB  (best upper bound on profit)
        n_iter      – int             number of C&CG iterations
        runtime     – float           total wall-clock time (seconds)
        converged   – bool            True if gap tolerance was met
        iter_log    – list of dicts   per-iteration details
    """
    I  = inst["I"]
    J  = inst["J"]
    R  = inst["R"]
    H  = list(range(Hn))
    demand          = inst["demand"]
    capacity        = inst["capacity"]
    fixed_cost      = inst["fixed_cost"]
    coeff1          = inst["coeff1"]
    coeff2          = inst["coeff2"]
    congestion_cost = inst["congestion_cost"]

    start  = time.time()
    LB     = -np.inf
    UB     =  np.inf
    x0     = x_init
    n_iter = 0
    converged = False
    iter_log  = []

    # ------------------------------------------------------------------ #
    #  Initialise master problem                                           #
    # ------------------------------------------------------------------ #
    master = gp.Model("CCG_master")
    master.Params.OutputFlag = 0
    master.Params.TimeLimit  = master_time_limit
    # MIPGap is set adaptively inside the loop — do not set it here.

    x   = master.addVars(J, R, vtype=GRB.BINARY,     name="x")
    nue = master.addVar(lb=-GRB.INFINITY,             name="nue")

    master.setObjective(
        gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue,
        GRB.MINIMIZE,
    )
    for j in J:
        master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)

    # Per-scenario variable containers (keyed by scenario index s)
    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # ---- Step 1: subproblem (separation) -------------------------
        t_sub_start = time.time()
        eps_bar, sub_obj = solve_subproblem_dual(
            x0, inst, uncertainty_budget, Hn,
            big_M=big_M,
            time_limit=min(master_time_limit, time_limit - elapsed),
            return_duals=False,
        )
        t_sub = time.time() - t_sub_start

        fixed_now = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)
        UB = min(UB, sub_obj + fixed_now)

        # ε_bar collapsed to a scalar per facility: ε̄_j = Σ_h (1 - h/(H-1)) ε[j,h]
        eps_scalar = {
            j: sum((1 - h / (Hn - 1)) * eps_bar[j, h] for h in H)
            for j in J
        }

        # ---- Step 2: add scenario block to master --------------------
        s = n_iter

        for i in I:
            Q[s, i]   = master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"Q_{s}_{i}")
            VV1[s, i] = master.addVar(vtype=GRB.CONTINUOUS, lb=0,      name=f"VV1_{s}_{i}")

        for j in J:
            for r in R:
                V1[s, j, r] = master.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"V1_{s}_{j}_{r}")
                V2[s, j, r] = master.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"V2_{s}_{j}_{r}")
                V3[s, j, r] = master.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"V3_{s}_{j}_{r}")
                C[s, j, r]  = master.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{s}_{j}_{r}")
                for i in I:
                    y[s, i, j, r] = master.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{s}_{i}_{j}_{r}"
                    )

        # Optimality cut  η ≥ recourse_obj(y^s, C^s, Q^s)
        master.addConstr(
            nue >= gp.quicksum(coeff1[i, j] * y[s, i, j, r]
                               for i in I for j in J for r in R)
                 + gp.quicksum(coeff2[i] * Q[s, i] for i in I)
                 + gp.quicksum(congestion_cost[j] * C[s, j, r]
                               for j in J for r in R),
            name=f"opt_cut_{s}",
        )

        for i in I:
            master.addConstr(
                VV1[s, i] == gp.quicksum(y[s, i, j, r] for j in J for r in R)
            )
            # Revenue SOC:  (sum y)^2  ≤  Q
            # NOTE: this is the simplified form used in the original code.
            # The full rotated form is  ||[2*sum_y, 1-Q]||^2 ≤ (1+Q)^2,
            # which is equivalent to (sum_y)^2 ≤ Q  only when Q ∈ [0,1].
            # Since Q is bounded [0,1] here this simplification is valid.
            master.addQConstr(VV1[s, i] ** 2 <= Q[s, i], name=f"rev_soc_{s}_{i}")
            master.addConstr(
                gp.quicksum(y[s, i, j, r] for j in J for r in R) <= 1
            )
            for j in J:
                master.addConstr(gp.quicksum(y[s, i, j, r] for r in R) <= 1)

        for j in J:
            # Capacity constraint under disruption
            master.addConstr(
                gp.quicksum(demand[i] * y[s, i, j, r] for i in I for r in R)
                <= eps_scalar[j] * gp.quicksum(capacity[j, r] * x[j, r] for r in R)
            )
            master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)

            for r in R:
                lam = gp.quicksum(demand[i] * y[s, i, j, r] for i in I)
                master.addConstr(V1[s, j, r] == lam)
                master.addConstr(
                    V2[s, j, r] == eps_scalar[j] * capacity[j, r] * C[s, j, r] - lam
                )
                master.addConstr(
                    V3[s, j, r] == eps_scalar[j] * capacity[j, r] * x[j, r] - lam
                )
                if congestion_cost[j] != 0:
                    # Congestion SOC (rotated form): V1^2 ≤ V2 * V3
                    master.addQConstr(
                        V1[s, j, r] ** 2 <= V2[s, j, r] * V3[s, j, r],
                        name=f"cong_soc_{s}_{j}_{r}",
                    )
                for i in I:
                    master.addConstr(y[s, i, j, r] <= x[j, r])

        # ---- Step 3: solve master ------------------------------------
        # Adaptive MIP gap: solve loosely when outer gap is large,
        # tighten as we approach convergence. This avoids wasting time
        # solving the master to high precision in early iterations.
        current_gap = abs(UB - LB) / (abs(UB) + 1e-10) if UB != np.inf else 1.0
        if current_gap > 0.20:
            master.Params.MIPGap = 0.10   # 10% — early iterations
        elif current_gap > 0.05:
            master.Params.MIPGap = 0.05   # 5%  — mid iterations
        else:
            master.Params.MIPGap = master_mip_gap  # tight — near convergence

        # Warm-start: inject current x0 and nue as MIP start hint.
        # The previous solution is still feasible after adding one block,
        # so this gives Gurobi a strong starting point every iteration.
        for j in J:
            for r in R:
                x[j, r].Start = x0[j, r]
        nue.Start = UB - sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)

        master.update()
        t_master_start = time.time()
        master.optimize()
        t_master = time.time() - t_master_start

        x0 = {(j, r): x[j, r].x for j in J for r in R}
        LB = max(master.ObjVal, LB)
        n_iter += 1

        gap_rel = abs(UB - LB) / (abs(UB) + 1e-10)

        iter_log.append({
            "iter":       n_iter,
            "LB":         LB,
            "UB":         UB,
            "gap_pct":    100 * gap_rel,
            "t_sub":      round(t_sub,    3),
            "t_master":   round(t_master, 3),
            "elapsed":    time.time() - start,
            "eps_bar":    eps_bar,   # worst-case scenario for duplicate detection
        })

        if verbose:
            print(
                f"  iter {n_iter:3d} | LB={LB:12.2f} | UB={UB:12.2f} "
                f"| gap={100*gap_rel:.2f}% | sub={t_sub:.1f}s | master={t_master:.1f}s"
            )

        if gap_rel <= tol:
            converged = True
            break

    return {
        "x_jr":      x0,
        "LB":        LB,
        "UB":        UB,
        "profit_LB": -UB,
        "n_iter":    n_iter,
        "runtime":   time.time() - start,
        "converged": converged,
        "iter_log":  iter_log,
    }
