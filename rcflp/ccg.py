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

Scenario management (Option A scalability)
------------------------------------------
n_warmstart       : heuristic scenario blocks added before main loop [0]
                    Populates master with structurally likely worst-
                    case scenarios (top-Γ facilities by congestion
                    cost/capacity) so first master solve is tighter.
max_active_blocks : hard cap on live scenario blocks in master  [None]
                    When the cap is reached, the oldest non-binding
                    block is dropped. Bounds master size so the
                    algorithm scales to large (I,J) instances.
drop_patience     : consecutive non-binding iterations before a
                    block is eligible for dropping               [2]
"""

import time
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from rcflp.subproblem import solve_subproblem_dual


# --------------------------------------------------------------------------- #
# Heuristic warm-start scenario generator                                      #
# --------------------------------------------------------------------------- #

def _heuristic_scenarios(inst, Hn: int, uncertainty_budget: float, n: int) -> list:
    """
    Generate up to n scenario dicts (eps_bar) without solving any optimisation.

    Strategy: sort facilities by (congestion_cost / avg_capacity) — those are
    most sensitive to degradation — and create one scenario per contiguous
    window of Γ facilities from the ranked list, each degraded to level H-1.
    """
    J   = inst["J"]
    R   = inst["R"]
    H   = list(range(Hn))
    Gamma = int(round(uncertainty_budget))  # number of facilities to degrade

    # Score: higher congestion & lower capacity → more critical
    avg_cap = {j: sum(inst["capacity"][j, r] for r in R) / len(R) for j in J}
    score   = {j: inst["congestion_cost"][j] / (avg_cap[j] + 1e-10) for j in J}
    ranked  = sorted(J, key=lambda j: -score[j])

    def _make(degraded):
        eps_bar = {}
        dset = set(degraded)
        for j in J:
            for h in H:
                eps_bar[j, h] = 0.0
            eps_bar[j, Hn - 1 if j in dset else 0] = 1.0
        return eps_bar

    scenarios = []
    n_j = len(ranked)

    # Sliding windows of size Gamma across ranked facilities
    for start in range(min(n, n_j)):
        degraded = [ranked[(start + k) % n_j] for k in range(min(Gamma, n_j))]
        s = _make(degraded)
        if s not in scenarios:
            scenarios.append(s)
        if len(scenarios) >= n:
            break

    return scenarios


# --------------------------------------------------------------------------- #
# Main solver                                                                   #
# --------------------------------------------------------------------------- #

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
    # --- Option A: scenario management ---
    n_warmstart: int = 0,
    max_active_blocks: int = None,
    drop_patience: int = 2,
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

    # Variables and constraints keyed by (scenario index s, ...)
    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    # Scenario management bookkeeping
    active_blocks      = []          # ordered list of live block indices
    block_vars         = {}          # s -> list of Gurobi vars
    block_constrs      = {}          # s -> list of Gurobi constrs (excl. opt_cut)
    opt_cut_constr     = {}          # s -> the nue >= ... constraint
    J_active_per_block = {}          # s -> list of active J for that block
    block_non_binding  = {}          # s -> consecutive non-binding count
    eps_bar_per_block  = {}          # s -> eps_bar dict (for re-add if needed)

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
        J_active_per_block[s] = J_active
        eps_bar_per_block[s]  = eps_bar

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
                    y[s, i, j, r] = master.addVar(lb=0, ub=1,
                                                   name=f"y_{s}_{i}_{j}_{r}")
                    bvars.append(y[s, i, j, r])

        oc = master.addConstr(
            nue >= gp.quicksum(coeff1[i, j] * y[s, i, j, r]
                               for i in I for j in J_active for r in R)
                 + gp.quicksum(coeff2[i] * Q[s, i] for i in I)
                 + gp.quicksum(congestion_cost[j] * C[s, j, r]
                               for j in J_active for r in R),
            name=f"opt_cut_{s}",
        )
        opt_cut_constr[s] = oc

        for i in I:
            c1 = master.addConstr(
                VV1[s, i] == gp.quicksum(y[s, i, j, r]
                                         for j in J_active for r in R)
            )
            c2 = master.addConstr(
                gp.quicksum(y[s, i, j, r]
                            for j in J_active for r in R) <= 1
            )
            bconstrs += [c1, c2]
            for j in J_active:
                c3 = master.addConstr(
                    gp.quicksum(y[s, i, j, r] for r in R) <= 1
                )
                bconstrs.append(c3)

        for j in J_active:
            c4 = master.addConstr(
                gp.quicksum(demand[i] * y[s, i, j, r] for i in I for r in R)
                <= eps_scalar[j] * gp.quicksum(capacity[j, r] * x[j, r] for r in R)
            )
            c5 = master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)
            bconstrs += [c4, c5]
            for r in R:
                lam = gp.quicksum(demand[i] * y[s, i, j, r] for i in I)
                c6 = master.addConstr(V1[s, j, r] == lam)
                c7 = master.addConstr(
                    V2[s, j, r] ==
                    eps_scalar[j] * capacity[j, r] * C[s, j, r] - lam
                )
                c8 = master.addConstr(
                    V3[s, j, r] ==
                    eps_scalar[j] * capacity[j, r] * x[j, r] - lam
                )
                bconstrs += [c6, c7, c8]
                for i in I:
                    c9 = master.addConstr(y[s, i, j, r] <= x[j, r])
                    bconstrs.append(c9)

        for i in I:
            master.addQConstr(VV1[s, i] ** 2 <= Q[s, i])

        for j in J_active:
            if congestion_cost[j] != 0:
                for r in R:
                    master.addQConstr(
                        V1[s, j, r] ** 2 <= V2[s, j, r] * V3[s, j, r]
                    )

        block_vars[s]        = bvars
        block_constrs[s]     = bconstrs
        block_non_binding[s] = 0
        active_blocks.append(s)
        s_counter += 1

    # ------------------------------------------------------------------ #
    # Helper: drop a non-binding block to keep master compact             #
    # ------------------------------------------------------------------ #
    def _drop_block(s):
        master.remove(block_vars[s])
        master.remove(block_constrs[s])
        master.remove([opt_cut_constr[s]])
        active_blocks.remove(s)
        # Clean up variable dicts to avoid stale references
        J_act = J_active_per_block[s]
        for i in I:
            del Q[s, i], VV1[s, i]
        for j in J_act:
            for r in R:
                del V1[s, j, r], V2[s, j, r], V3[s, j, r], C[s, j, r]
                for i in I:
                    del y[s, i, j, r]
        del block_vars[s], block_constrs[s], opt_cut_constr[s]
        del J_active_per_block[s], block_non_binding[s]

    # ------------------------------------------------------------------ #
    # Helper: update non-binding counters and drop if over cap            #
    # ------------------------------------------------------------------ #
    def _manage_blocks():
        if not active_blocks:
            return
        nue_val = nue.x
        for s in list(active_blocks):
            J_act = J_active_per_block[s]
            eps_scalar_s = {
                j: sum((1 - h / (Hn - 1)) * eps_bar_per_block[s][j, h] for h in H)
                for j in J_act
            }
            rhs = (sum(coeff1[i, j] * y[s, i, j, r].x
                       for i in I for j in J_act for r in R)
                   + sum(coeff2[i] * Q[s, i].x for i in I)
                   + sum(congestion_cost[j] * C[s, j, r].x
                         for j in J_act for r in R))
            slack = nue_val - rhs
            if slack > 1e-4 * (abs(nue_val) + 1.0):
                block_non_binding[s] += 1
            else:
                block_non_binding[s] = 0

        # Drop oldest non-binding blocks when over cap
        if max_active_blocks is not None:
            while len(active_blocks) > max_active_blocks:
                # Find the block with highest non-binding count (oldest inactive)
                candidate = max(
                    active_blocks,
                    key=lambda s: (block_non_binding[s], -s),
                )
                if block_non_binding[candidate] >= drop_patience:
                    if verbose:
                        print(f"    → dropping block {candidate} "
                              f"(non-binding for {block_non_binding[candidate]} iters)")
                    _drop_block(candidate)
                else:
                    break  # no block is eligible yet

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
            "iter":          n_iter,
            "LB":            L_ell if use_exploit else -U_j,
            "UB":            UB,
            "U_j":           U_j,
            "L_j":           L_j,
            "gap_pct":       round(true_gap * 100, 4),
            "inex_gap":      round(inex_gap * 100, 4),
            "mode":          mode,
            "ell":           ell,
            "eps_mp":        eps_mp,
            "tau":           tau,
            "n_blocks":      s_counter,
            "active_blocks": len(active_blocks),
            "t_sub":         round(t_sub,    3),
            "t_master":      round(t_master, 3),
            "elapsed":       time.time() - start,
        })
        if verbose:
            lb_show = -L_ell if use_exploit else -U_j
            print(
                f"  iter {n_iter:3d} [{mode:7s}] "
                f"| profit≥{-UB:10.2f} | LB={lb_show:10.2f} "
                f"| gap={true_gap*100:.2f}% "
                f"| blocks={len(active_blocks)} "
                f"| sub={t_sub:.1f}s master={t_master:.1f}s"
            )

    # ------------------------------------------------------------------ #
    # Warm-start: add heuristic scenario blocks before main loop          #
    # ------------------------------------------------------------------ #
    if n_warmstart > 0:
        warm_scenarios = _heuristic_scenarios(
            inst, Hn, uncertainty_budget, n_warmstart
        )
        for eps_bar in warm_scenarios:
            _add_block(eps_bar)
        if verbose:
            print(f"  warm-start: added {len(warm_scenarios)} heuristic blocks")

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # Loop order: subproblem → add block(s) → solve master                #
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

        # ---- Step 4: manage active blocks (drop non-binding) --------- #
        _manage_blocks()

        # LB for gap: use ObjBound (proven lower bound from Gurobi B&B)
        LB_now   = L_ell if use_exploit else L_j
        abs_UB   = abs(UB) + 1e-10
        true_gap = (UB - LB_now) / abs_UB
        inex_gap = (UB - U_j)   / abs_UB

        _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "explore")

        # ---- Step 5: convergence check ------------------------------- #
        if true_gap <= tol:
            converged = True
            break

        # ---- Step 6: exploit phase (only when eps_e > 0) ------------ #
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
                _manage_blocks()

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
        "x_jr":         x0,
        "LB":           best_LB,
        "UB":           UB,
        "profit_LB":    -UB,
        "n_iter":       n_iter,
        "n_blocks":     s_counter,
        "active_blocks": len(active_blocks),
        "runtime":      time.time() - start,
        "converged":    converged,
        "iter_log":     iter_log,
    }
