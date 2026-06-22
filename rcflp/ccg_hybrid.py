"""
rcflp.ccg_hybrid
----------------
Hybrid C&CG algorithm: adds a linear Benders-style optimality cut alongside
each SOCP scenario block.

Standard CCG adds one SOCP block per iteration (constraints 18b-18g in the
paper). Each block introduces O(|I||J||R|) new variables and SOC constraints,
making the master MISOCP progressively more expensive. The SOCP constraints
are relaxed at every B&B node, so the LP relaxation is loose.

This variant additionally adds a linear Benders cut at every iteration using
the dual solution from the subproblem (the same dual variables used by BDCP).
The cut is linear in x[j,r] and therefore tightens the LP relaxation at every
node of the master's B&B tree at no structural cost.

Cut form (same as BDCP):
  η ≥ -Σ_i α_i
      - Σ_{j,r,h} capacity[j,r] * x[j,r] * (1 - h/(H-1)) * θ̃[j,h]
      - Σ_{j,r,h} capacity[j,r] * x[j,r] * (1 - h/(H-1)) * ũ2[j,h]
      - Σ_{j,r,h} capacity[j,r] * x[j,r] * (1 - h/(H-1)) * γ̃[j,h]
      + Σ_i t2_i - Σ_i β_i

All other algorithmic logic (i-C&CG, exploit/explore, termination) is
identical to ccg.py.
"""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from rcflp.subproblem import solve_subproblem_dual


def solve_CCG_hybrid(
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
    # Option A / RPBD parameters accepted but not used (API compatibility with solve_CCG)
    n_warmstart: int = 0,
    max_active_blocks: int = None,
    drop_patience: int = 2,
    partial_block_fraction: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Hybrid C&CG: SOCP blocks + linear Benders cuts added simultaneously.

    Parameters are identical to solve_CCG in ccg.py.
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
    UB        = 0.0
    x0        = x_init
    n_iter    = 0
    s_counter = 0
    converged = False
    iter_log  = []

    # i-C&CG state
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
    master = gp.Model("CCG_hybrid_master")
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

    obj_expr  = gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue
    lb_constr = master.addConstr(obj_expr >= L, name="lb_anchor")

    # ------------------------------------------------------------------ #
    # Helper: add SOCP scenario block (unchanged from ccg.py)             #
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
    # Helper: add linear Benders cut from dual variables                  #
    # ------------------------------------------------------------------ #
    def _add_benders_cut(alpha_bar, beta_bar, t2_bar,
                         theta_bar, u2_bar, gamma_bar, cut_id):
        """
        Add the linear optimality cut derived from the subproblem dual.

        Uses the same formula as BDCP: all j ∈ J are included, with
        big_M returned by the subproblem for non-open facilities.  The
        large negative coefficient (-capacity * big_M * x[j,r]) for a
        closed facility makes the cut trivially satisfied whenever the
        master opens a new facility, preserving global validity.

        Skipped when x0 has no open facilities: the subproblem is then
        degenerate (all duals zero) and the constant term is 0, so the
        cut `nue >= 0` would permanently lock the master's lower bound.
        """
        if not any(x0[j, r] > 0.5 for j in J for r in R):
            return  # degenerate first iteration — skip

        master.addConstr(
            nue >= - gp.quicksum(alpha_bar[i] for i in I)
                   - gp.quicksum(
                         capacity[j, r] * x[j, r] * (1 - h / (Hn - 1))
                         * theta_bar[j, h]
                         for j in J for r in R for h in H)
                   - gp.quicksum(
                         capacity[j, r] * x[j, r] * (1 - h / (Hn - 1))
                         * u2_bar[j, h]
                         for j in J for r in R for h in H)
                   - gp.quicksum(
                         capacity[j, r] * x[j, r] * (1 - h / (Hn - 1))
                         * gamma_bar[j, h]
                         for j in J for r in R for h in H)
                   + gp.quicksum(t2_bar[i]   for i in I)
                   - gp.quicksum(beta_bar[i] for i in I),
            name=f"benders_cut_{cut_id}",
        )

    # ------------------------------------------------------------------ #
    # Helper: solve master                                                 #
    # ------------------------------------------------------------------ #
    def _solve_master():
        nonlocal n_iter, L, L_ell, ell, x0
        master.Params.MIPGap    = eps_mp
        master.Params.TimeLimit = tau
        lb_constr.RHS = L
        master.update()
        t0 = time.time()
        master.optimize()
        t_master = time.time() - t0

        x0  = {(j, r): x[j, r].x for j in J for r in R}
        U_j = master.ObjVal
        L_j = master.ObjBound
        n_iter += 1

        if L_j > L_ell and L_j > L + 1e-6:
            L_ell = L_j
            ell   = n_iter

        L             = U_j
        lb_constr.RHS = L
        master.update()

        return U_j, L_j, t_master

    # ------------------------------------------------------------------ #
    # Helper: solve subproblem — returns eps_bar_list, dual vars, obj     #
    # ------------------------------------------------------------------ #
    def _solve_subproblem(remaining):
        if remaining < 1.0:
            return None, None, None, None

        fixed_now = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)

        # Primary call: return_duals=True to get both ε* and dual variables
        (eps_bar_primary, alpha_bar, beta_bar, t2_bar,
         theta_bar, u2_bar, gamma_bar, sub_obj) = solve_subproblem_dual(
            x0, inst, uncertainty_budget, Hn,
            big_M=big_M,
            time_limit=min(tau, remaining),
            return_duals=True,
        )

        # Build epsilon map in the same format as return_duals=False
        eps_bar_list = [eps_bar_primary]

        # Optional second scenario (diversity, same logic as ccg.py)
        if n_scenarios >= 2:
            key1 = frozenset(
                (j, h) for (j, h), v in eps_bar_primary.items() if v > 0.5
            )
            eps_bar_list2, _ = solve_subproblem_dual(
                x0, inst, uncertainty_budget, Hn,
                big_M=big_M,
                time_limit=min(tau, remaining),
                return_duals=False,
                n_scenarios=1,
                exclude_scenario=eps_bar_primary,
            )
            key2 = frozenset(
                (j, h) for (j, h), v in eps_bar_list2[0].items() if v > 0.5
            )
            if key2 != key1:
                eps_bar_list = eps_bar_list + eps_bar_list2

        duals = (alpha_bar, beta_bar, t2_bar, theta_bar, u2_bar, gamma_bar)
        return eps_bar_list, sub_obj, fixed_now, duals

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
                f"| sub={t_sub:.1f}s master={t_master:.1f}s "
                f"| blocks={s_counter}"
            )

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    benders_cut_id = 0

    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # ---- Step 1: solve master ------------------------------------ #
        U_j, L_j, t_master = _solve_master()

        # ---- Step 2: solve subproblem -------------------------------- #
        t_sub_start = time.time()
        remaining   = max(0.0, time_limit - (time.time() - start))
        if remaining == 0.0:
            break
        eps_bar_list, sub_obj, fixed_now, duals = _solve_subproblem(remaining)
        t_sub = time.time() - t_sub_start

        if eps_bar_list is None:
            break

        UB = min(UB, sub_obj + fixed_now)

        abs_UB   = abs(UB) + 1e-10
        true_gap = (UB - L_ell) / abs_UB
        inex_gap = (UB - U_j)   / abs_UB

        _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "explore")

        # ---- Step 3: optimality test --------------------------------- #
        if true_gap <= tol:
            remaining_final = max(0.0, time_limit - (time.time() - start))
            if remaining_final > 0.0:
                _, sub_obj_f, fixed_f, _ = _solve_subproblem(remaining_final)
                if sub_obj_f is not None:
                    UB = min(UB, sub_obj_f + fixed_f)
            converged = True
            break

        # ---- Step 3: backtracking ------------------------------------ #
        if inex_gap < eps_e:
            max_exploit_iters = 5
            n_exploit_iters   = 0
            while inex_gap < eps_e and n_exploit_iters < max_exploit_iters:
                elapsed = time.time() - start
                if elapsed > time_limit:
                    break

                eps_mp        = alpha * eps_mp
                tau           = tau + beta
                L             = L_ell
                n_exploit_iters += 1

                if verbose:
                    print(f"    → exploit: eps_mp={eps_mp:.5f} "
                          f"tau={tau:.0f}s L_anchor={L:.4f}")

                U_j, L_j, t_master = _solve_master()

                t_sub_start = time.time()
                remaining   = max(0.0, time_limit - (time.time() - start))
                if remaining == 0.0:
                    eps_bar_list = None
                    duals = None
                    break
                eps_bar_list, sub_obj, fixed_now, duals = _solve_subproblem(remaining)
                t_sub = time.time() - t_sub_start

                if eps_bar_list is None:
                    break

                UB = min(UB, sub_obj + fixed_now)

                abs_UB   = abs(UB) + 1e-10
                true_gap = (UB - L_ell) / abs_UB
                inex_gap = (UB - U_j)   / abs_UB

                _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "exploit")

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

        # ---- Step 4: add SOCP block(s) + Benders cut ---------------- #
        if eps_bar_list is None:
            break

        for eps_bar in eps_bar_list:
            _add_block(eps_bar)

        # Add one Benders cut per iteration using the primary dual solution
        if duals is not None:
            alpha_bar, beta_bar, t2_bar, theta_bar, u2_bar, gamma_bar = duals
            _add_benders_cut(alpha_bar, beta_bar, t2_bar,
                             theta_bar, u2_bar, gamma_bar,
                             benders_cut_id)
            benders_cut_id += 1

    return {
        "x_jr":         x0,
        "LB":           L_ell,
        "UB":           UB,
        "profit_LB":    -UB,
        "n_iter":       n_iter,
        "n_blocks":     s_counter,
        "n_benders":    benders_cut_id,
        "runtime":      time.time() - start,
        "converged":    converged,
        "iter_log":     iter_log,
    }
