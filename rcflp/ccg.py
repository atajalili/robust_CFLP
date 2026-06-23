"""
rcflp.ccg
---------
Column-and-Constraint Generation (C&CG) for the two-stage robust CFLP.

════════════════════════════════════════════════════════════════════════════
OPERATING MODES
════════════════════════════════════════════════════════════════════════════

MODE 1 · Pure CCG                                           eps_e = 0.0
──────────────────────────────────────────────────────────────────────────
  Each iteration: solve subproblem → add full SOC block → solve master.
  Master grows by O(|I|×|J|×|R|) per iteration.

MODE 2 · i-C&CG                                            eps_e > 0.0
──────────────────────────────────────────────────────────────────────────
  Adds an exploit phase when the primal gap (UB − U_j)/|UB| drops below
  eps_e: tightens master MIPGap (×alpha each step) and extends its time
  limit (+beta seconds each step) to push L_j up before adding new blocks.

  WARNING: activates an lb_anchor constraint whose RHS changes each
  iteration, which invalidates Gurobi's LP warm-start.  Use only when
  the master is very fast to re-solve from scratch.

════════════════════════════════════════════════════════════════════════════
PARAMETERS
════════════════════════════════════════════════════════════════════════════
master_mip_gap    MIPGap for the master MISOCP                    [0.015]
master_time_limit Per-master Gurobi time limit (s)                 [2000]
n_scenarios       Diverse worst-case scenarios added per iter    [1 or 2]
eps_e             i-C&CG primal gap threshold — 0.0 = MODE 1       [0.0]
alpha             i-C&CG MIPGap reduction factor per exploit step   [0.8]
beta              i-C&CG time-limit increment per exploit step (s)  [300]
════════════════════════════════════════════════════════════════════════════
"""

import time
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
    """
    Solve the two-stage robust CFLP via C&CG.

    Returns a dict with keys:
      x_jr, LB, UB, profit_LB, n_iter, n_blocks, runtime, converged, iter_log
    """
    I               = inst["I"]
    J               = inst["J"]
    R               = inst["R"]
    H               = list(range(Hn))
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

    # i-C&CG state (only meaningful when use_exploit=True)
    L      = L_init
    L_ell  = L_init
    ell    = 0
    eps_mp = master_mip_gap
    tau    = master_time_limit

    # Per-scenario variable dicts (keyed by scenario index s)
    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    # ── Build master ─────────────────────────────────────────────────────────
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

    lb_constr = None
    if use_exploit:
        obj_expr  = gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue
        lb_constr = master.addConstr(obj_expr >= L, name="lb_anchor")

    # ── Add one full SOC scenario block ──────────────────────────────────────
    def _add_block(eps_bar):
        nonlocal s_counter

        eps_scalar = {
            j: sum((1 - h / (Hn - 1)) * eps_bar[j, h] for h in H)
            for j in J
        }
        s        = s_counter
        J_active = [j for j in J if eps_scalar[j] > 0]

        bvars    = []
        bconstrs = []

        for i in I:
            Q[s, i]   = master.addVar(lb=0, ub=1, name=f"Q_{s}_{i}")
            VV1[s, i] = master.addVar(lb=0,       name=f"VV1_{s}_{i}")
            bvars += [Q[s, i], VV1[s, i]]

        for j in J_active:
            for r in R:
                V1[s, j, r] = master.addVar(lb=0, name=f"V1_{s}_{j}_{r}")
                V2[s, j, r] = master.addVar(lb=0, name=f"V2_{s}_{j}_{r}")
                V3[s, j, r] = master.addVar(lb=0, name=f"V3_{s}_{j}_{r}")
                C[s, j, r]  = master.addVar(lb=0, name=f"C_{s}_{j}_{r}")
                bvars += [V1[s, j, r], V2[s, j, r], V3[s, j, r], C[s, j, r]]
                for i in I:
                    y[s, i, j, r] = master.addVar(lb=0, ub=1, name=f"y_{s}_{i}_{j}_{r}")
                    bvars.append(y[s, i, j, r])

        master.addConstr(
            nue >= gp.quicksum(coeff1[i, j] * y[s, i, j, r]
                               for i in I for j in J_active for r in R)
                 + gp.quicksum(coeff2[i] * Q[s, i] for i in I)
                 + gp.quicksum(congestion_cost[j] * C[s, j, r]
                               for j in J_active for r in R),
            name=f"opt_cut_{s}",
        )

        for i in I:
            bconstrs.append(master.addConstr(
                VV1[s, i] == gp.quicksum(y[s, i, j, r] for j in J_active for r in R)
            ))
            bconstrs.append(master.addConstr(
                gp.quicksum(y[s, i, j, r] for j in J_active for r in R) <= 1
            ))
            for j in J_active:
                bconstrs.append(master.addConstr(
                    gp.quicksum(y[s, i, j, r] for r in R) <= 1
                ))

        for j in J_active:
            bconstrs.append(master.addConstr(
                gp.quicksum(demand[i] * y[s, i, j, r] for i in I for r in R)
                <= eps_scalar[j] * gp.quicksum(capacity[j, r] * x[j, r] for r in R)
            ))
            bconstrs.append(master.addConstr(
                gp.quicksum(x[j, r] for r in R) <= 1
            ))
            for r in R:
                lam = gp.quicksum(demand[i] * y[s, i, j, r] for i in I)
                bconstrs.append(master.addConstr(V1[s, j, r] == lam))
                bconstrs.append(master.addConstr(
                    V2[s, j, r] == eps_scalar[j] * capacity[j, r] * C[s, j, r] - lam
                ))
                bconstrs.append(master.addConstr(
                    V3[s, j, r] == eps_scalar[j] * capacity[j, r] * x[j, r] - lam
                ))
                for i in I:
                    bconstrs.append(master.addConstr(y[s, i, j, r] <= x[j, r]))

        for i in I:
            master.addQConstr(VV1[s, i] ** 2 <= Q[s, i])

        for j in J_active:
            if congestion_cost[j] != 0:
                for r in R:
                    master.addQConstr(V1[s, j, r] ** 2 <= V2[s, j, r] * V3[s, j, r])

        s_counter += 1

    # ── Solve subproblem; optionally add a second diverse scenario ────────────
    def _solve_subproblem(remaining):
        if remaining < 1.0:
            return None, None, None

        fixed_now    = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)
        eps_bar_list, sub_obj = solve_subproblem_dual(
            x0, inst, uncertainty_budget, Hn,
            big_M=big_M, time_limit=min(tau, remaining),
            return_duals=False, n_scenarios=1,
        )

        if n_scenarios >= 2:
            eps_bar_list2, _ = solve_subproblem_dual(
                x0, inst, uncertainty_budget, Hn,
                big_M=big_M, time_limit=min(tau, remaining),
                return_duals=False, n_scenarios=1,
                exclude_scenario=eps_bar_list[0],
            )
            key1 = frozenset((j, h) for (j, h), v in eps_bar_list[0].items()  if v > 0.5)
            key2 = frozenset((j, h) for (j, h), v in eps_bar_list2[0].items() if v > 0.5)
            if key2 != key1:
                eps_bar_list = eps_bar_list + eps_bar_list2

        return eps_bar_list, sub_obj, fixed_now

    # ── Solve master; update i-C&CG state ────────────────────────────────────
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

    # ── Log one iteration ─────────────────────────────────────────────────────
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
                f"  iter {n_iter:3d} [{mode:7s}]"
                f" | profit≥{-UB:10.2f} | LB={lb_show:10.2f}"
                f" | gap={true_gap*100:.2f}%"
                f" | blocks={s_counter}"
                f" | sub={t_sub:.1f}s master={t_master:.1f}s"
            )

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # Step 1: solve subproblem → update UB
        t_sub_start              = time.time()
        eps_bar_list, sub_obj, fixed_now = _solve_subproblem(
            max(0.0, time_limit - elapsed)
        )
        t_sub = time.time() - t_sub_start

        if eps_bar_list is None:
            break

        UB = min(UB, sub_obj + fixed_now)

        # Step 2: add block(s) to master
        for eps_bar in eps_bar_list:
            _add_block(eps_bar)

        # Step 3: solve master → update LB
        U_j, L_j, t_master = _solve_master()

        LB_now   = L_ell if use_exploit else L_j
        abs_UB   = abs(UB) + 1e-10
        true_gap = (UB - LB_now) / abs_UB
        inex_gap = (UB - U_j)    / abs_UB

        _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "explore")

        # Step 4: check convergence
        if true_gap <= tol:
            converged = True
            break

        # Step 5: i-C&CG exploit phase (MODE 2 only)
        if use_exploit and inex_gap < eps_e:
            n_exploit_iters = 0
            while inex_gap < eps_e and n_exploit_iters < 5:
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
                eps_bar_list, sub_obj, fixed_now = _solve_subproblem(
                    max(0.0, time_limit - (time.time() - start))
                )
                t_sub = time.time() - t_sub_start

                if eps_bar_list is None:
                    break

                UB       = min(UB, sub_obj + fixed_now)
                abs_UB   = abs(UB) + 1e-10
                true_gap = (UB - L_ell) / abs_UB
                inex_gap = (UB - U_j)   / abs_UB

                _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "exploit")

                if true_gap <= tol:
                    converged = True
                    break

            if converged:
                break
            if eps_bar_list is not None:
                for eps_bar in eps_bar_list:
                    _add_block(eps_bar)

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
