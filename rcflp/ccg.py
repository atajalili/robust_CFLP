"""
rcflp.ccg
---------
Column-and-Constraint Generation (C&CG) algorithm with optional
inexact (i-C&CG) exploit phase (Tsang, Shehadeh & Curtis, 2022).

Loop order matches the original implementation and Zeng & Zhao (2013):
  1. Solve subproblem for current x → get ε*, update UB
  2. Add scenario block(s) to master
  3. Solve master → get new x, update LB
  4. Termination / backtracking check

When eps_e = 0 (default when exploit is disabled) the lb_anchor
constraint is omitted entirely, which lets Gurobi warm-start each
master re-solve from the previous integer solution without disruption.

Key parameters
--------------
master_mip_gap    : MIPGap tolerance for master                  [0.015]
master_time_limit : per-master time limit (seconds)              [2000]
n_scenarios       : worst-case scenarios added per iteration      [1]
eps_e             : inexact gap threshold for exploitation        [0.0]
                    0.0 = pure basic C&CG (no exploit phase)
                    must satisfy eps_e < tol/(1+tol) when > 0
alpha             : MIPGap reduction factor per exploit step      [0.8]
beta              : time limit increment per exploit step (s)     [300]
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
    n_scenarios: int = 1,
    L_init: float = -1e10,
    eps_e: float = 0.0,
    alpha: float = 0.8,
    beta: float = 300,
    verbose: bool = False,
) -> dict:

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

    use_exploit = eps_e > 0.0

    start     = time.time()
    UB        = 0.0
    x0        = x_init
    n_iter    = 0
    s_counter = 0
    converged = False
    iter_log  = []

    # i-C&CG state (only used when use_exploit=True)
    L      = L_init
    L_ell  = L_init
    ell    = 0
    eps_mp = master_mip_gap
    tau    = master_time_limit

    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    # ------------------------------------------------------------------ #
    # Build master                                                         #
    # ------------------------------------------------------------------ #
    master = gp.Model("CCG_master")
    master.Params.OutputFlag = 0
    master.Params.MIPGap     = eps_mp
    master.Params.TimeLimit  = tau

    x   = master.addVars(J, R, vtype=GRB.BINARY, name="x")
    nue = master.addVar(lb=-GRB.INFINITY, name="nue")

    master.setObjective(
        gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue,
        GRB.MINIMIZE,
    )
    for j in J:
        master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)

    # lb_anchor is only needed for the exploit phase
    lb_constr = None
    if use_exploit:
        obj_expr  = gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue
        lb_constr = master.addConstr(obj_expr >= L, name="lb_anchor")

    # ------------------------------------------------------------------ #
    # Helper: add one scenario block to master                            #
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
    # Helper: solve subproblem for current x0                             #
    # ------------------------------------------------------------------ #
    def _solve_subproblem(remaining):
        if remaining < 1.0:
            return None, None, None
        fixed_now = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)

        eps_bar_list, sub_obj = solve_subproblem_dual(
            x0, inst, uncertainty_budget, Hn,
            big_M=big_M,
            time_limit=min(tau, remaining),
            return_duals=False,
            n_scenarios=1,
        )

        if n_scenarios >= 2:
            key1 = frozenset(
                (j, h) for (j, h), v in eps_bar_list[0].items() if v > 0.5
            )
            eps_bar_list2, _ = solve_subproblem_dual(
                x0, inst, uncertainty_budget, Hn,
                big_M=big_M,
                time_limit=min(tau, remaining),
                return_duals=False,
                n_scenarios=1,
                exclude_scenario=eps_bar_list[0],
            )
            key2 = frozenset(
                (j, h) for (j, h), v in eps_bar_list2[0].items() if v > 0.5
            )
            if key2 != key1:
                eps_bar_list = eps_bar_list + eps_bar_list2

        return eps_bar_list, sub_obj, fixed_now

    # ------------------------------------------------------------------ #
    # Helper: solve master                                                 #
    # ------------------------------------------------------------------ #
    def _solve_master():
        nonlocal n_iter, L, L_ell, ell, x0
        master.Params.MIPGap    = eps_mp
        master.Params.TimeLimit = tau
        if use_exploit and lb_constr is not None:
            lb_constr.RHS = L
        master.update()
        t0 = time.time()
        master.optimize()
        t_master = time.time() - t0

        x0  = {(j, r): x[j, r].x for j in J for r in R}
        U_j = master.ObjVal
        L_j = master.ObjBound
        n_iter += 1

        if use_exploit:
            if L_j > L_ell and L_j > L + 1e-6:
                L_ell = L_j
                ell   = n_iter
            L = U_j

        return U_j, L_j, t_master

    # ------------------------------------------------------------------ #
    # Helper: log                                                          #
    # ------------------------------------------------------------------ #
    def _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, mode):
        iter_log.append({
            "iter":     n_iter,
            "LB":       L_ell if use_exploit else -U_j,
            "UB":       UB,
            "U_j":      U_j,
            "L_j":      L_j,
            "gap_pct":  round(true_gap * 100, 4),
            "inex_gap": round(inex_gap * 100, 4),
            "mode":     mode,
            "ell":      ell,
            "eps_mp":   eps_mp,
            "tau":      tau,
            "n_blocks": s_counter,
            "t_sub":    round(t_sub,    3),
            "t_master": round(t_master, 3),
            "elapsed":  time.time() - start,
        })
        if verbose:
            lb_show = -L_ell if use_exploit else -U_j
            print(
                f"  iter {n_iter:3d} [{mode:7s}] "
                f"| profit≥{-UB:10.2f} | LB={lb_show:10.2f} "
                f"| gap={true_gap*100:.2f}% "
                f"| sub={t_sub:.1f}s master={t_master:.1f}s"
            )

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # Loop order: subproblem → add block(s) → solve master                #
    # This matches the original implementation and gives Gurobi the best  #
    # chance to warm-start each master re-solve from the previous one.    #
    # ------------------------------------------------------------------ #
    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # ---- Step 1: subproblem for current x0 ----------------------- #
        t_sub_start = time.time()
        remaining   = max(0.0, time_limit - elapsed)
        eps_bar_list, sub_obj, fixed_now = _solve_subproblem(remaining)
        t_sub = time.time() - t_sub_start

        if eps_bar_list is None:
            break

        UB = min(UB, sub_obj + fixed_now)

        # ---- Step 2: add scenario block(s) --------------------------- #
        for eps_bar in eps_bar_list:
            _add_block(eps_bar)

        # ---- Step 3: solve master ------------------------------------ #
        U_j, L_j, t_master = _solve_master()

        # LB for gap: use ObjBound (proven lower bound from Gurobi B&B)
        LB_now = L_ell if use_exploit else L_j
        abs_UB   = abs(UB) + 1e-10
        true_gap = (UB - LB_now) / abs_UB
        inex_gap = (UB - U_j)   / abs_UB

        _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "explore")

        # ---- Step 4: convergence check ------------------------------- #
        if true_gap <= tol:
            converged = True
            break

        # ---- Step 5: exploit phase (only when eps_e > 0) ------------ #
        if use_exploit and inex_gap < eps_e:
            max_exploit_iters = 5
            n_exploit_iters   = 0
            while inex_gap < eps_e and n_exploit_iters < max_exploit_iters:
                elapsed = time.time() - start
                if elapsed > time_limit:
                    break

                eps_mp = alpha * eps_mp
                tau    = tau + beta
                L      = L_ell
                n_exploit_iters += 1

                if verbose:
                    print(f"    → exploit: eps_mp={eps_mp:.5f} tau={tau:.0f}s")

                U_j, L_j, t_master = _solve_master()

                t_sub_start = time.time()
                remaining   = max(0.0, time_limit - (time.time() - start))
                eps_bar_list, sub_obj, fixed_now = _solve_subproblem(remaining)
                t_sub = time.time() - t_sub_start

                if eps_bar_list is None:
                    break

                UB = min(UB, sub_obj + fixed_now)

                abs_UB   = abs(UB) + 1e-10
                true_gap = (UB - L_ell) / abs_UB
                inex_gap = (UB - U_j)   / abs_UB

                _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "exploit")

                if true_gap <= tol:
                    converged = True
                    break

            if converged:
                break

            # Back to exploration: add blocks from last subproblem call
            if eps_bar_list is not None:
                for eps_bar in eps_bar_list:
                    _add_block(eps_bar)

    # Final LB: best proven lower bound across all master solves
    best_LB = L_ell if use_exploit else (iter_log[-1]["L_j"] if iter_log else L_init)

    return {
        "x_jr":      x0,
        "LB":        best_LB,
        "UB":        UB,
        "profit_LB": -UB,
        "n_iter":    n_iter,
        "n_blocks":  s_counter,
        "runtime":   time.time() - start,
        "converged": converged,
        "iter_log":  iter_log,
    }
