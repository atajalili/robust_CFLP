"""
rcflp.ccg
---------
Column-and-Constraint Generation (C&CG) algorithm.
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
    n_scenarios: int = 2,
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

    start     = time.time()
    LB        = -np.inf
    UB        = 0.0
    x0        = x_init
    n_iter    = 0
    s_counter = 0
    converged = False
    iter_log  = []
    diff_prev = 100.0   # drives loose MIPGap on first iteration

    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    master = gp.Model("CCG_master")
    master.Params.OutputFlag = 0
    master.Params.MIPGap     = master_mip_gap
    master.Params.TimeLimit  = master_time_limit

    x   = master.addVars(J, R, vtype=GRB.BINARY, name="x")
    nue = master.addVar(lb=-GRB.INFINITY, name="nue")

    master.setObjective(
        gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue,
        GRB.MINIMIZE,
    )
    for j in J:
        master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)

    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # Step 1: subproblem — worst-case scenario for current x0
        t_sub_start = time.time()
        remaining = time_limit - elapsed
        eps_bar_list, sub_obj = solve_subproblem_dual(
            x0, inst, uncertainty_budget, Hn,
            big_M=big_M,
            time_limit=min(master_time_limit, remaining),
            return_duals=False,
            n_scenarios=1,
        )

        # Optionally find a second genuinely adversarial scenario by re-solving
        # with an integer cut that excludes the first scenario.
        if n_scenarios >= 2:
            key1 = frozenset(
                (j, h) for (j, h), v in eps_bar_list[0].items() if v > 0.5
            )
            eps_bar_list2, _ = solve_subproblem_dual(
                x0, inst, uncertainty_budget, Hn,
                big_M=big_M,
                time_limit=min(master_time_limit, remaining),
                return_duals=False,
                n_scenarios=1,
                exclude_scenario=eps_bar_list[0],
            )
            key2 = frozenset(
                (j, h) for (j, h), v in eps_bar_list2[0].items() if v > 0.5
            )
            if key2 != key1:
                eps_bar_list = eps_bar_list + eps_bar_list2

        t_sub = time.time() - t_sub_start

        fixed_now = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)
        UB = min(UB, sub_obj + fixed_now)

        # Step 2: add one scenario block per returned scenario
        for eps_bar in eps_bar_list:
            eps_scalar = {
                j: sum((1 - h / (Hn - 1)) * eps_bar[j, h] for h in H)
                for j in J
            }
            s = s_counter

            for i in I:
                Q[s, i]   = master.addVar(lb=0, ub=1, name=f"Q_{s}_{i}")
                VV1[s, i] = master.addVar(lb=0,       name=f"VV1_{s}_{i}")

            for j in J:
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
                master.addConstr(
                    gp.quicksum(y[s, i, j, r] for j in J for r in R) <= 1
                )
                for j in J:
                    master.addConstr(
                        gp.quicksum(y[s, i, j, r] for r in R) <= 1
                    )

            for j in J:
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

            for j in J:
                if congestion_cost[j] != 0:
                    for r in R:
                        master.addQConstr(
                            V1[s, j, r] ** 2 <= V2[s, j, r] * V3[s, j, r]
                        )

            s_counter += 1
        # end for eps_bar in eps_bar_list

        # Step 3: solve master
        # Adaptive MIPGap: loose when far from convergence, tight at the end
        #if diff_prev > 15:
        #    master.Params.MIPGap = 0.10
        #elif diff_prev > 3:
        #    master.Params.MIPGap = 0.05
        #else:
        #    master.Params.MIPGap = master_mip_gap
        master.Params.MIPGap = master_mip_gap

        master.update()
        t_master_start = time.time()
        master.optimize()
        t_master = time.time() - t_master_start

        x0 = {(j, r): x[j, r].x for j in J for r in R}
        LB = max(master.ObjVal, LB)
        n_iter += 1

        # Termination: 1% relative gap, matching original code exactly
        diff = 100 * (round(UB, 2) - round(LB, 2)) / (abs(UB) + 0.000001)
        diff_prev = diff

        iter_log.append({
            "iter":     n_iter,
            "LB":       LB,
            "UB":       UB,
            "gap_pct":  diff,
            "t_sub":    round(t_sub,    3),
            "t_master": round(t_master, 3),
            "elapsed":  time.time() - start,
        })

        if verbose:
            print(
                f"  iter {n_iter:3d} | LB={LB:12.2f} | UB={UB:12.2f} "
                f"| gap={diff:.2f}% "
                f"| sub={t_sub:.1f}s | master={t_master:.1f}s"
            )

        if diff <= 1:
            eps_bar_final_list, sub_obj_final = solve_subproblem_dual(
                x0, inst, uncertainty_budget, Hn,
                big_M=big_M, time_limit=master_time_limit,
                return_duals=False, n_scenarios=1,
            )
            fixed_final = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)
            UB = min(UB, sub_obj_final + fixed_final)
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
