"""
rcflp.ccg
---------
Exact Column-and-Constraint Generation (C&CG) algorithm.
Implements Algorithm 1 from Zeng & Zhao (2013) as applied to the
two-stage robust CFLP.

Step order:
  1. Solve subproblem for current x → find worst-case ε*, update UB.
  2. Add one scenario block (columns and constraints) to the master.
  3. Solve master exactly → update LB and x.
  4. Terminate when (UB - LB) / |UB| ≤ tol or time limit reached.
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
    master_time_limit: float = 600,
    verbose: bool = False,
) -> dict:
    """
    Solve the robust CFLP with the exact Column-and-Constraint Generation algorithm.

    Parameters
    ----------
    inst               : instance dict from instancemaker()
    uncertainty_budget : Γ
    Hn                 : number of disruption levels
    x_init             : {(j,r): float}  warm-start first-stage solution
    tol                : relative optimality gap tolerance (e.g. 0.01 = 1%)
    big_M              : big-M for the subproblem bilinear linearisation
    time_limit         : total wall-clock time limit (seconds)
    master_time_limit  : per-master-solve time limit (seconds)
    verbose            : print iteration log

    Returns
    -------
    dict with keys:
        x_jr        – {(j,r): float}  optimal first-stage solution
        LB          – float           final lower bound  (minimisation obj)
        UB          – float           final upper bound  (minimisation obj)
        profit_LB   – float           = -UB
        n_iter      – int             number of C&CG iterations
        n_blocks    – int             number of scenario blocks added
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

    start     = time.time()
    LB        = -np.inf
    UB        =  np.inf
    x0        = x_init
    n_iter    = 0
    s_counter = 0
    converged = False
    iter_log  = []

    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    # ------------------------------------------------------------------ #
    #  Build master problem                                                #
    # ------------------------------------------------------------------ #
    master = gp.Model("CCG_master")
    master.Params.OutputFlag = 0
    master.Params.TimeLimit  = master_time_limit

    x   = master.addVars(J, R, vtype=GRB.BINARY, name="x")
    nue = master.addVar(lb=-GRB.INFINITY,          name="nue")

    master.setObjective(
        gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue,
        GRB.MINIMIZE,
    )
    for j in J:
        master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)

    # ------------------------------------------------------------------ #
    #  Helper: add one scenario block to master                           #
    # ------------------------------------------------------------------ #
    def _add_block(eps_bar):
        nonlocal s_counter
        eps_scalar = {
            j: sum((1 - h / (Hn - 1)) * eps_bar[j, h] for h in H)
            for j in J
        }
        s = s_counter
        J_active = [j for j in J if eps_scalar[j] > 0]

        for i in I:
            Q[s, i]   = master.addVar(lb=0, ub=1, name=f"Q_{s}_{i}")
            VV1[s, i] = master.addVar(lb=0,       name=f"VV1_{s}_{i}")

        for j in J_active:
            for r in R:
                V1[s, j, r] = master.addVar(lb=0, name=f"V1_{s}_{j}_{r}")
                V2[s, j, r] = master.addVar(lb=0, name=f"V2_{s}_{j}_{r}")
                V3[s, j, r] = master.addVar(lb=0, name=f"V3_{s}_{j}_{r}")
                C[s, j, r]  = master.addVar(lb=0, name=f"C_{s}_{j}_{r}")
                for i in I:
                    y[s, i, j, r] = master.addVar(lb=0, ub=1,
                                                   name=f"y_{s}_{i}_{j}_{r}")

        master.addConstr(
            nue >= gp.quicksum(coeff1[i, j] * y[s, i, j, r]
                               for i in I for j in J_active for r in R)
                 + gp.quicksum(coeff2[i] * Q[s, i] for i in I)
                 + gp.quicksum(congestion_cost[j] * C[s, j, r]
                               for j in J_active for r in R),
            name=f"opt_cut_{s}",
        )

        for i in I:
            master.addConstr(
                VV1[s, i] == gp.quicksum(y[s, i, j, r]
                                         for j in J_active for r in R)
            )
            master.addConstr(
                gp.quicksum(y[s, i, j, r]
                            for j in J_active for r in R) <= 1
            )
            for j in J_active:
                master.addConstr(
                    gp.quicksum(y[s, i, j, r] for r in R) <= 1
                )

        for j in J_active:
            master.addConstr(
                gp.quicksum(demand[i] * y[s, i, j, r] for i in I for r in R)
                <= eps_scalar[j] * gp.quicksum(capacity[j, r] * x[j, r] for r in R)
            )
            master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)
            for r in R:
                lam = gp.quicksum(demand[i] * y[s, i, j, r] for i in I)
                master.addConstr(V1[s, j, r] == lam)
                master.addConstr(
                    V2[s, j, r] ==
                    eps_scalar[j] * capacity[j, r] * C[s, j, r] - lam
                )
                master.addConstr(
                    V3[s, j, r] ==
                    eps_scalar[j] * capacity[j, r] * x[j, r] - lam
                )
                for i in I:
                    master.addConstr(y[s, i, j, r] <= x[j, r])

        for i in I:
            master.addQConstr(VV1[s, i] ** 2 <= Q[s, i])

        for j in J_active:
            if congestion_cost[j] != 0:
                for r in R:
                    master.addQConstr(
                        V1[s, j, r] ** 2 <= V2[s, j, r] * V3[s, j, r]
                    )

        s_counter += 1

    # ------------------------------------------------------------------ #
    #  Main loop: subproblem → add block → master → gap check             #
    # ------------------------------------------------------------------ #
    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # ---- Step 1: subproblem — find worst-case ε* -----------------
        t_sub_start = time.time()
        eps_bar_list, sub_obj = solve_subproblem_dual(
            x0, inst, uncertainty_budget, Hn,
            big_M=big_M,
            time_limit=min(master_time_limit, time_limit - elapsed),
            return_duals=False,
        )
        t_sub = time.time() - t_sub_start

        fixed_now = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)
        UB = min(UB, sub_obj + fixed_now)

        # ---- Step 2: add scenario block ------------------------------
        _add_block(eps_bar_list[0])

        # ---- Step 3: solve master ------------------------------------
        master.update()
        t_master_start = time.time()
        master.optimize()
        t_master = time.time() - t_master_start

        x0 = {(j, r): x[j, r].x for j in J for r in R}
        LB = max(master.ObjVal, LB)
        n_iter += 1

        gap = (UB - LB) / (abs(UB) + 1e-10)

        iter_log.append({
            "iter":     n_iter,
            "LB":       LB,
            "UB":       UB,
            "gap_pct":  round(gap * 100, 4),
            "n_blocks": s_counter,
            "t_sub":    round(t_sub,    3),
            "t_master": round(t_master, 3),
            "elapsed":  time.time() - start,
        })

        if verbose:
            print(
                f"  iter {n_iter:3d} | LB={-LB:12.2f} | UB={-UB:12.2f} "
                f"| gap={gap*100:.2f}% | blocks={s_counter} "
                f"| sub={t_sub:.1f}s | master={t_master:.1f}s"
            )

        if gap <= tol:
            converged = True
            break

    return {
        "x_jr":      x0,
        "LB":        LB,
        "UB":        UB,
        "profit_LB": -UB,
        "n_iter":    n_iter,
        "n_blocks":  s_counter,
        "runtime":   time.time() - start,
        "converged": converged,
        "iter_log":  iter_log,
    }
