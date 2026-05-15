"""
rcflp.ccg
---------
Inexact Column-and-Constraint Generation (i-C&CG) algorithm,
time-limit variant (Algorithm 4, Tsang, Shehadeh & Curtis, 2022).

Step order follows the paper exactly:
  1. Solve master (inexactly) → get x^j, U^j, L^j
  2. Solve subproblem for x^j → get ξ*, update U
  3. Backtracking / termination check
  4. Add scenario block, go to 1

Key parameters
--------------
master_mip_gap    : initial MIPGap tolerance                  [0.015]
master_time_limit : initial per-master time limit (seconds)   [100]
L_init            : initial valid lower bound on master obj   (required,
                    pass negative nominal profit e.g. -173716)
eps_e             : inexact gap threshold for exploitation     [0.009]
                    must satisfy eps_e < tol/(1+tol)
alpha             : MIPGap reduction factor per exploit step   [0.8]
beta              : time limit increment per exploit step (s)  [300]
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
    master_time_limit: float = 100,
    n_scenarios: int = 2,
    L_init: float = -1e10,
    eps_e: float = 0.009,
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

    start     = time.time()
    UB        = 0.0          # U in paper: best upper bound found (min cost)
    x0        = x_init
    n_iter    = 0
    s_counter = 0
    converged = False
    iter_log  = []

    # i-C&CG state (Algorithm 4)
    L      = L_init    # RHS of lb_anchor; initialised to valid lower bound
    L_ell  = L_init    # L^ℓ: best VALID lower bound seen so far
    ell    = 0         # iteration index of L_ell
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

    # Lower bound anchor: c'x + δ >= L  (eq. 4c)
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
    # Helper: solve master, update L, L_ell, x0                          #
    # ------------------------------------------------------------------ #
    def _solve_master():
        nonlocal n_iter, L, L_ell, ell, x0
        master.Params.MIPGap    = eps_mp
        master.Params.TimeLimit = tau
        # Update lb_anchor RHS before solving
        lb_constr.RHS = L
        master.update()
        t0 = time.time()
        master.optimize()
        t_master = time.time() - t0

        x0  = {(j, r): x[j, r].x for j in J for r in R}
        U_j = master.ObjVal     # best feasible (step 1.2: U^j)
        L_j = master.ObjBound   # proven lower bound (step 1.2: L^j)
        n_iter += 1

        # Step 1.2: if L^j > L, update ℓ
        if L_j > L_ell and L_j > L + 1e-6:
            L_ell = L_j
            ell   = n_iter

        # Step 1.3: L ← U^j  (paper uses U^j, not L^j)
        L             = U_j
        lb_constr.RHS = L
        master.update()   # apply RHS change immediately

        return U_j, L_j, t_master

    # ------------------------------------------------------------------ #
    # Helper: solve subproblem for current x0                             #
    # ------------------------------------------------------------------ #
    def _solve_subproblem(remaining):
        if remaining < 1.0:
            return None, None, None   # signal to caller that time is up
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
    # Helper: log iteration                                               #
    # ------------------------------------------------------------------ #
    def _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, mode):
        iter_log.append({
            "iter":     n_iter,
            "LB":       L_ell,
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
            print(
                f"  iter {n_iter:3d} [{mode:7s}] "
                f"| L_ell={-L_ell:10.2f} | UB={-UB:10.2f} "
                f"| gap={true_gap*100:.2f}% | inex={inex_gap*100:.2f}% "
                f"| tau={tau:.0f}s eps_mp={eps_mp:.4f} "
                f"| sub={t_sub:.1f}s master={t_master:.1f}s"
            )
            print(
                f"         [RAW] L_ell={L_ell:.4f} UB={UB:.4f} "
                f"U_j={U_j:.4f} L_j={L_j:.4f} L={L:.4f}"
            )

    # ------------------------------------------------------------------ #
    # Main loop — follows Algorithm 4 step order exactly:                 #
    # Step 1 (master) → Step 2 (subproblem) → Step 3 (backtrack/term)   #
    # → Step 4 (add block) → repeat                                      #
    # ------------------------------------------------------------------ #
    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # ---- Step 1: solve master ------------------------------------ #
        U_j, L_j, t_master = _solve_master()

        # ---- Step 2: solve subproblem for x^j ----------------------- #
        t_sub_start = time.time()
        remaining   = max(0.0, time_limit - (time.time() - start))
        if remaining == 0.0:
            break
        eps_bar_list, sub_obj, fixed_now = _solve_subproblem(remaining)
        t_sub = time.time() - t_sub_start

        UB = min(UB, sub_obj + fixed_now)

        # Compute gaps
        abs_UB   = abs(UB) + 1e-10
        true_gap = (UB - L_ell) / abs_UB
        inex_gap = (UB - U_j)   / abs_UB

        _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "explore")

        # ---- Step 3: optimality test --------------------------------- #
        if true_gap <= tol:
            remaining_final = max(0.0, time_limit - (time.time() - start))
            if remaining_final > 0.0:
                _, sub_obj_f, fixed_f = _solve_subproblem(remaining_final)
                UB = min(UB, sub_obj_f + fixed_f)
            converged = True
            break

        # ---- Step 3: backtracking routine ---------------------------- #
        if inex_gap < eps_e:
            # Exploitation: tighten master, restore L^ℓ as anchor.
            # max_exploit_iters prevents infinite loop when inex_gap
            # is stuck (master keeps returning same solution).
            max_exploit_iters = 5
            n_exploit_iters   = 0
            while inex_gap < eps_e and n_exploit_iters < max_exploit_iters:
                elapsed = time.time() - start
                if elapsed > time_limit:
                    break

                eps_mp        = alpha * eps_mp
                tau           = tau + beta
                L             = L_ell   # restore valid lower bound anchor
                n_exploit_iters += 1

                if verbose:
                    print(f"    → exploit: eps_mp={eps_mp:.5f} "
                          f"tau={tau:.0f}s L_anchor={L:.4f}")

                # Step 1: re-solve master without adding new block
                U_j, L_j, t_master = _solve_master()

                # Step 2: solve subproblem for new x^j
                t_sub_start = time.time()
                remaining   = max(0.0, time_limit - (time.time() - start))
                if remaining == 0.0:
                    eps_bar_list = None
                    break
                eps_bar_list, sub_obj, fixed_now = _solve_subproblem(remaining)
                t_sub = time.time() - t_sub_start

                if eps_bar_list is None:
                    break

                UB = min(UB, sub_obj + fixed_now)

                abs_UB   = abs(UB) + 1e-10
                true_gap = (UB - L_ell) / abs_UB
                inex_gap = (UB - U_j)   / abs_UB

                _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "exploit")

                # Termination check
                if true_gap <= tol:
                    remaining_final = max(0.0, time_limit - (time.time() - start))
                    if remaining_final > 0.0:
                        result = _solve_subproblem(remaining_final)
                        if result[0] is not None:
                            UB = min(UB, result[1] + result[2])
                    converged = True
                    break

            if converged:
                break
            # inex_gap >= eps_e or max exploit iters reached
            # → fall through to exploration (add block)

        # ---- Step 4: add scenario block(s) -------------------------- #
        if eps_bar_list is None:
            break   # time limit hit, stop cleanly
        for eps_bar in eps_bar_list:
            _add_block(eps_bar)

    return {
        "x_jr":      x0,
        "LB":        L_ell,
        "UB":        UB,
        "profit_LB": -UB,
        "n_iter":    n_iter,
        "n_blocks":  s_counter,
        "runtime":   time.time() - start,
        "converged": converged,
        "iter_log":  iter_log,
    }
