"""
rcflp.warmstart
---------------
Warm-start strategies for the C&CG algorithm.

solve_robust_warmstart()
    Single-stage robust warm-start. Finds a better x_init for CCG by:

    Step 1 — Find worst-case scenario for x_nom:
        Call the subproblem once on x_nom (the nominal solution). This
        returns eps_worst, the disruption scenario that is most damaging
        to the nominal solution. This is free — CCG would do this anyway
        in iteration 1.

    Step 2 — Solve a single-scenario MISOCP:
        Build a master with just this one scenario block and solve it
        to (near) optimality. The result x_robust is optimised against
        the true worst-case disruption, not a random or nominal scenario.

    Why this is better than SAA:
        SAA minimises average recourse cost across random scenarios.
        The CCG subproblem is adversarial — it finds the WORST case for
        whatever x is given. x_SAA may be good on average but still
        terrible in the worst case, so the initial CCG gap stays large.

        x_robust is optimised against an actual worst-case scenario, so
        it is structurally more robust and gives a much smaller initial
        gap when CCG evaluates it in iteration 1.

    Overhead:
        One subproblem solve (cheap — ~1-2s) + one master solve (same
        cost as one CCG master iteration). Total overhead is roughly
        equal to one CCG iteration. If this saves 5+ iterations, it's
        a net win.
"""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from rcflp.subproblem import solve_subproblem_dual


def solve_robust_warmstart(
    inst              : dict,
    uncertainty_budget: float,
    Hn                : int,
    x_nom             : dict,
    big_M             : float = 10_000,
    mip_gap           : float = 0.05,
    time_limit        : float = 120,
    verbose           : bool  = False,
) -> dict:
    """
    Single-stage robust warm-start for CCG.

    Parameters
    ----------
    inst               : instance dict from instancemaker()
    uncertainty_budget : Gamma
    Hn                 : number of disruption levels
    x_nom              : nominal solution from solve_nominal()['x_jr']
    big_M              : big-M for subproblem (same as CCG)
    mip_gap            : MIP gap for the single-scenario solve (default 5%)
    time_limit         : wall-clock limit in seconds (default 120s)
    verbose            : print solve progress

    Returns
    -------
    dict with keys:
        x_jr       - {(j,r): float}  warm-start first-stage solution
        n_open     - int             number of open facilities
        obj        - float           single-scenario objective value
        t_sub      - float           subproblem solve time (s)
        t_master   - float           single-scenario master solve time (s)
        runtime    - float           total wall-clock time (s)
        converged  - bool            True if MIP gap was met
    """
    I            = inst["I"]
    J            = inst["J"]
    R            = inst["R"]
    H            = list(range(Hn))
    fixed_cost   = inst["fixed_cost"]
    demand       = inst["demand"]
    capacity     = inst["capacity"]
    coeff1       = inst["coeff1"]
    coeff2       = inst["coeff2"]
    cong_cost    = inst["congestion_cost"]

    start = time.time()

    # ------------------------------------------------------------------ #
    #  Step 1: find worst-case scenario for x_nom                         #
    # ------------------------------------------------------------------ #
    if verbose:
        print("  Warmstart step 1: subproblem on x_nom...")

    t_sub_start = time.time()
    eps_bar, sub_obj = solve_subproblem_dual(
        x_nom, inst, uncertainty_budget, Hn,
        big_M=big_M,
        time_limit=min(60, time_limit),
        return_duals=False,
    )
    t_sub = time.time() - t_sub_start

    # Compute eps_scalar from eps_bar (same formula as CCG)
    eps_scalar = {
        j: sum((1 - h / (Hn - 1)) * eps_bar[j, h] for h in H)
        for j in J
    }

    n_disrupted = sum(1 for j in J if eps_scalar[j] < 0.999)
    if verbose:
        print(f"    sub_obj={sub_obj:.1f}  disrupted={n_disrupted} facilities  "
              f"t_sub={t_sub:.1f}s")

    # ------------------------------------------------------------------ #
    #  Step 2: solve single-scenario MISOCP                               #
    # ------------------------------------------------------------------ #
    if verbose:
        print("  Warmstart step 2: single-scenario MISOCP...")

    t_master_start = time.time()

    master = gp.Model("robust_warmstart")
    master.Params.OutputFlag   = 1 if verbose else 0
    master.Params.MIPGap       = mip_gap
    master.Params.TimeLimit    = max(1, time_limit - t_sub)

    x   = master.addVars(J, R, vtype=GRB.BINARY, name="x")
    nue = master.addVar(lb=-GRB.INFINITY, name="nue")

    master.setObjective(
        gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue,
        GRB.MINIMIZE,
    )

    for j in J:
        master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)

    # Add the single worst-case scenario block
    s = 0
    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    for i in I:
        Q[i]   = master.addVar(lb=0, ub=1, name=f"Q_{i}")
        VV1[i] = master.addVar(lb=0,       name=f"VV1_{i}")

    for j in J:
        for r in R:
            V1[j, r] = master.addVar(lb=0, name=f"V1_{j}_{r}")
            V2[j, r] = master.addVar(lb=0, name=f"V2_{j}_{r}")
            V3[j, r] = master.addVar(lb=0, name=f"V3_{j}_{r}")
            C[j, r]  = master.addVar(lb=0, name=f"C_{j}_{r}")
            for i in I:
                y[i, j, r] = master.addVar(lb=0, ub=1, name=f"y_{i}_{j}_{r}")

    master.addConstr(
        nue >= gp.quicksum(coeff1[i, j] * y[i, j, r]
                           for i in I for j in J for r in R)
             + gp.quicksum(coeff2[i] * Q[i] for i in I)
             + gp.quicksum(cong_cost[j] * C[j, r] for j in J for r in R)
    )

    for i in I:
        master.addConstr(
            VV1[i] == gp.quicksum(y[i, j, r] for j in J for r in R)
        )
        master.addConstr(
            gp.quicksum(y[i, j, r] for j in J for r in R) <= 1
        )
        for j in J:
            master.addConstr(gp.quicksum(y[i, j, r] for r in R) <= 1)

    for j in J:
        master.addConstr(
            gp.quicksum(demand[i] * y[i, j, r] for i in I for r in R)
            <= eps_scalar[j] * gp.quicksum(capacity[j, r] * x[j, r] for r in R)
        )
        for r in R:
            lam = gp.quicksum(demand[i] * y[i, j, r] for i in I)
            master.addConstr(V1[j, r] == lam)
            master.addConstr(
                V2[j, r] == eps_scalar[j] * capacity[j, r] * C[j, r] - lam
            )
            master.addConstr(
                V3[j, r] == eps_scalar[j] * capacity[j, r] * x[j, r] - lam
            )
            for i in I:
                master.addConstr(y[i, j, r] <= x[j, r])

    for i in I:
        master.addQConstr(VV1[i] ** 2 <= Q[i])

    for j in J:
        if cong_cost[j] != 0:
            for r in R:
                master.addQConstr(V1[j, r] ** 2 <= V2[j, r] * V3[j, r])

    master.optimize()
    t_master = time.time() - t_master_start
    runtime  = time.time() - start

    if master.SolCount == 0:
        if verbose:
            print("  Warmstart: no solution found, falling back to x_nom")
        return None

    x_robust = {(j, r): x[j, r].x for j in J for r in R}
    n_open   = sum(1 for v in x_robust.values() if v > 0.5)
    converged = master.MIPGap <= mip_gap

    if verbose:
        print(f"  Warmstart: {n_open} facilities open  "
              f"obj={master.ObjVal:.1f}  gap={100*master.MIPGap:.1f}%  "
              f"t_sub={t_sub:.1f}s  t_master={t_master:.1f}s  "
              f"total={runtime:.1f}s  converged={converged}")

    return {
        "x_jr":      x_robust,
        "n_open":    n_open,
        "obj":       master.ObjVal,
        "t_sub":     t_sub,
        "t_master":  t_master,
        "runtime":   runtime,
        "converged": converged,
    }
