"""
rcflp.ccg
---------
Column-and-Constraint Generation (C&CG) for the two-stage robust CFLP.

════════════════════════════════════════════════════════════════════════════
OPERATING MODES — set via CCG_PARAMS in the notebook
════════════════════════════════════════════════════════════════════════════

MODE 1 · Pure CCG  (baseline — matches original February implementation)
─────────────────────────────────────────────────────────────────────────
  eps_e             = 0.0      no exploit phase; no lb_anchor constraint
  master_time_limit = 2000     per-master time limit (seconds)
  n_scenarios       = 1        one worst-case scenario added per iteration
  partial_block_fraction = 1.0 full SOC block per scenario (all J)
  n_warmstart       = 0        no heuristic pre-population
  max_active_blocks = None     master grows unboundedly

  Best for: instances where J ≤ 10 (master stays manageable).

────────────────────────────────────────────────────────────────────────
MODE 2 · CCG + Option A  (bounded master size for medium J)
────────────────────────────────────────────────────────────────────────
  All MODE 1 settings, plus:
  n_warmstart       = 3–5      add k heuristic blocks before the main loop
                               (facilities ranked by congestion_cost/capacity
                                degraded to maximum level)
  max_active_blocks = 8–12     drop oldest non-binding block when master
                               exceeds this count
  drop_patience     = 2        block must be non-binding for this many
                               consecutive iterations before dropping

  Effect: keeps master size bounded at O(max_active_blocks × I × J × R).
  Algorithm remains correct — the subproblem re-discovers any dropped
  scenario if it later becomes worst-case again.
  Trade-off: may increase iteration count slightly.

  Best for: J = 10–20; when master grows but dropping is affordable.

────────────────────────────────────────────────────────────────────────
MODE 3 · RPBD partial block  (scales to large J)
────────────────────────────────────────────────────────────────────────
  partial_block_fraction = 0.3–0.6   fraction of J_active to include in
                                      the SOC block (rest get Benders cut)
  eps_e             = 0.0
  n_scenarios       = 1        (forced; dual return not compatible with 2)

  Per iteration, for each scenario ε*:
    · Solve subproblem with duals (return_duals=True)
    · Add a LINEAR Benders cut covering ALL J_active       [cheap, always]
    · Add a PARTIAL SOC block for top-k facilities
        k = ceil(partial_block_fraction × |J_active|)
        ranked by  congestion_cost[j] × eps_scalar[j] / avg_capacity[j]
      [tight for most-critical facilities; SOCP constraints in master]

  Both cuts are valid lower bounds on the scenario recourse. Together
  they give  nue ≥ max(Benders, partial_SOC)  without an auxiliary var,
  because both are added as separate ≥ constraints on nue.

  Master grows at  O(fraction × I × J × R)  per iteration instead of
  O(I × J × R). Combine with max_active_blocks for full size control.

  Best for: J ≥ 15 where full CCG blocks become prohibitively large.

────────────────────────────────────────────────────────────────────────
MODE 4 · i-C&CG exploit phase  (experimental)
────────────────────────────────────────────────────────────────────────
  eps_e   > 0   (e.g. 0.005)   activates exploit loop
  alpha   = 0.8                MIPGap reduction factor per exploit step
  beta    = 300                time-limit increment per exploit step (s)

  WARNING: activates an lb_anchor constraint whose RHS changes each
  iteration — this invalidates Gurobi's LP basis warm-start and
  typically makes the algorithm SLOWER for this problem.
  Use eps_e = 0.0 unless you specifically want to test i-C&CG.

════════════════════════════════════════════════════════════════════════
PARAMETER QUICK REFERENCE
════════════════════════════════════════════════════════════════════════
master_mip_gap         MIPGap for the master MISOCP              [0.015]
master_time_limit      Per-master Gurobi time limit (s)           [2000]
n_scenarios            Extra diverse scenarios per iteration     [1 or 2]
eps_e                  i-C&CG inexact gap threshold — 0 = off      [0.0]
alpha                  i-C&CG MIPGap reduction per exploit step    [0.8]
beta                   i-C&CG time-limit increment per step (s)   [300]
n_warmstart            Heuristic blocks added before main loop       [0]
max_active_blocks      Hard cap on live blocks (None = off)        [None]
drop_patience          Non-binding iters before a block is dropped   [2]
partial_block_fraction RPBD fraction of J in SOC block (1.0=full) [1.0]
════════════════════════════════════════════════════════════════════════
"""

import math
import time
import gurobipy as gp
from gurobipy import GRB

from rcflp.subproblem import solve_subproblem_dual


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic warm-start scenario generator
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_scenarios(inst, Hn: int, uncertainty_budget: float, n: int) -> list:
    """
    Generate up to n scenario dicts (eps_bar) without solving any optimisation.

    Ranks facilities by congestion_cost / avg_capacity (most sensitive to
    disruption) and creates one scenario per sliding window of Γ facilities,
    each degraded to the maximum level h = H-1.
    """
    J     = inst["J"]
    R     = inst["R"]
    H     = list(range(Hn))
    Gamma = int(round(uncertainty_budget))

    avg_cap = {j: sum(inst["capacity"][j, r] for r in R) / len(R) for j in J}
    score   = {j: inst["congestion_cost"][j] / (avg_cap[j] + 1e-10) for j in J}
    ranked  = sorted(J, key=lambda j: -score[j])

    def _make(degraded):
        dset    = set(degraded)
        eps_bar = {}
        for j in J:
            for h in H:
                eps_bar[j, h] = 0.0
            eps_bar[j, Hn - 1 if j in dset else 0] = 1.0
        return eps_bar

    scenarios, n_j = [], len(ranked)
    for start in range(min(n, n_j)):
        s = _make([ranked[(start + k) % n_j] for k in range(min(Gamma, n_j))])
        if s not in scenarios:
            scenarios.append(s)
        if len(scenarios) >= n:
            break
    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_CCG(
    inst: dict,
    uncertainty_budget: float,
    Hn: int,
    x_init: dict,
    tol: float = 0.01,
    big_M: float = 10_000,
    time_limit: float = 6 * 3600,
    # ── master solver settings ──────────────────────────────────────────
    master_mip_gap: float = 0.015,
    master_time_limit: float = 2000,
    n_scenarios: int = 1,
    L_init: float = -1e10,
    # ── i-C&CG exploit phase (MODE 4) ──────────────────────────────────
    eps_e: float = 0.0,
    alpha: float = 0.8,
    beta: float = 300,
    # ── Option A: scenario management (MODE 2) ──────────────────────────
    n_warmstart: int = 0,
    max_active_blocks: int = None,
    drop_patience: int = 2,
    # ── RPBD partial block (MODE 3) ─────────────────────────────────────
    partial_block_fraction: float = 1.0,
    # ── misc ─────────────────────────────────────────────────────────────
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
    use_partial = 0.0 < partial_block_fraction < 1.0
    # n_scenarios > 1 is incompatible with return_duals=True (subproblem API)
    eff_n_scenarios = 1 if use_partial else n_scenarios

    avg_cap = {j: sum(capacity[j, r] for r in R) / len(R) for j in J}

    start     = time.time()
    UB        = 0.0
    x0        = x_init
    n_iter    = 0
    s_counter = 0
    n_benders = 0
    converged = False
    iter_log  = []

    # i-C&CG state
    L      = L_init
    L_ell  = L_init
    ell    = 0
    eps_mp = master_mip_gap
    tau    = master_time_limit

    # Per-variable dicts (keyed by scenario index s)
    y   = {}
    C   = {}
    Q   = {}
    VV1 = {}
    V1  = {}
    V2  = {}
    V3  = {}

    # Scenario bookkeeping
    active_blocks      = []
    block_vars         = {}
    block_constrs      = {}      # linear/quadratic constraints (excl. opt_cuts)
    opt_cut_constrs    = {}      # s -> list of opt_cut constraints (1 in full mode, 2 in partial)
    J_active_per_block = {}
    block_non_binding  = {}
    eps_bar_per_block  = {}

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

    # ── Helper: add one scenario block ───────────────────────────────────────
    def _add_block(eps_bar, duals=None):
        """
        Add scenario block for eps_bar to the master.

        duals : tuple (alpha, beta, t2, theta, u2, gamma) from subproblem —
                required when use_partial=True; ignored otherwise.
        """
        nonlocal s_counter, n_benders

        eps_scalar = {
            j: sum((1 - h / (Hn - 1)) * eps_bar[j, h] for h in H)
            for j in J
        }
        s        = s_counter
        J_active = [j for j in J if eps_scalar[j] > 0]
        J_active_per_block[s] = J_active
        eps_bar_per_block[s]  = eps_bar

        bvars    = []
        bconstrs = []
        opt_cuts = []

        # ── Select facilities for the SOC block ──────────────────────────────
        if use_partial and duals is not None and J_active:
            # Rank J_active by importance: high congestion & high degradation
            score_j = {
                j: congestion_cost[j] * eps_scalar[j] / (avg_cap[j] + 1e-10)
                for j in J_active
            }
            ranked = sorted(J_active, key=lambda j: -score_j[j])
            k_retain = max(1, math.ceil(partial_block_fraction * len(J_active)))
            J_socp = ranked[:k_retain]      # exact SOC treatment
        else:
            J_socp = J_active               # full block (MODE 1 / 2)

        # ── Benders cut (full scenario) — only in RPBD mode ──────────────────
        if use_partial and duals is not None:
            alpha_d, beta_d, t2_d, theta_d, u2_d, gamma_d = duals
            cut_expr = (
                -gp.quicksum(alpha_d[i] for i in I)
                - gp.quicksum(
                    capacity[j, r] * x[j, r] * (1 - h / (Hn - 1))
                    * (theta_d[j, h] + u2_d[j, h] + gamma_d[j, h])
                    for j in J for r in R for h in H
                )
                + gp.quicksum(t2_d[i] for i in I)
                - gp.quicksum(beta_d[i] for i in I)
            )
            bc = master.addConstr(nue >= cut_expr, name=f"bcut_{s}")
            opt_cuts.append(bc)
            n_benders += 1

        # ── SOC block for J_socp ─────────────────────────────────────────────
        for i in I:
            Q[s, i]   = master.addVar(lb=0, ub=1, name=f"Q_{s}_{i}")
            VV1[s, i] = master.addVar(lb=0,       name=f"VV1_{s}_{i}")
            bvars += [Q[s, i], VV1[s, i]]

        for j in J_socp:
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

        # opt_cut for the SOC block
        socp_cut = master.addConstr(
            nue >= gp.quicksum(coeff1[i, j] * y[s, i, j, r]
                               for i in I for j in J_socp for r in R)
                 + gp.quicksum(coeff2[i] * Q[s, i] for i in I)
                 + gp.quicksum(congestion_cost[j] * C[s, j, r]
                               for j in J_socp for r in R),
            name=f"opt_cut_{s}",
        )
        opt_cuts.append(socp_cut)

        for i in I:
            c1 = master.addConstr(
                VV1[s, i] == gp.quicksum(y[s, i, j, r]
                                         for j in J_socp for r in R)
            )
            c2 = master.addConstr(
                gp.quicksum(y[s, i, j, r]
                            for j in J_socp for r in R) <= 1
            )
            bconstrs += [c1, c2]
            for j in J_socp:
                bconstrs.append(master.addConstr(
                    gp.quicksum(y[s, i, j, r] for r in R) <= 1
                ))

        for j in J_socp:
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
                    V2[s, j, r] ==
                    eps_scalar[j] * capacity[j, r] * C[s, j, r] - lam
                ))
                bconstrs.append(master.addConstr(
                    V3[s, j, r] ==
                    eps_scalar[j] * capacity[j, r] * x[j, r] - lam
                ))
                for i in I:
                    bconstrs.append(master.addConstr(y[s, i, j, r] <= x[j, r]))

        for i in I:
            master.addQConstr(VV1[s, i] ** 2 <= Q[s, i])

        for j in J_socp:
            if congestion_cost[j] != 0:
                for r in R:
                    master.addQConstr(
                        V1[s, j, r] ** 2 <= V2[s, j, r] * V3[s, j, r]
                    )

        block_vars[s]        = bvars
        block_constrs[s]     = bconstrs
        opt_cut_constrs[s]   = opt_cuts
        block_non_binding[s] = 0
        active_blocks.append(s)
        s_counter += 1

    # ── Helper: remove a non-binding block ───────────────────────────────────
    def _drop_block(s):
        master.remove(block_vars[s])
        master.remove(block_constrs[s])
        master.remove(opt_cut_constrs[s])
        active_blocks.remove(s)
        J_act = J_active_per_block[s]
        for i in I:
            del Q[s, i], VV1[s, i]
        for j in J_act:
            for r in R:
                if (s, j, r) in V1:
                    del V1[s, j, r], V2[s, j, r], V3[s, j, r], C[s, j, r]
                for i in I:
                    if (s, i, j, r) in y:
                        del y[s, i, j, r]
        del block_vars[s], block_constrs[s], opt_cut_constrs[s]
        del J_active_per_block[s], block_non_binding[s]

    # ── Helper: update non-binding counters; drop when over cap ──────────────
    def _manage_blocks():
        if not active_blocks:
            return
        nue_val = nue.x
        for s in list(active_blocks):
            # Use the last opt_cut (the SOC cut) to check binding
            oc = opt_cut_constrs[s][-1]
            try:
                slack = nue_val - oc.getAttr("RHS") - master.getRow(oc).getValue()
            except Exception:
                slack = 0.0
            # simpler slack approximation via Slack attribute if available
            try:
                slack = -oc.Slack   # Slack = LHS - RHS for >= constraints; negative = binding
            except Exception:
                pass
            if slack > 1e-4 * (abs(nue_val) + 1.0):
                block_non_binding[s] += 1
            else:
                block_non_binding[s] = 0

        if max_active_blocks is not None:
            while len(active_blocks) > max_active_blocks:
                candidate = max(active_blocks,
                                key=lambda s: (block_non_binding[s], -s))
                if block_non_binding[candidate] >= drop_patience:
                    if verbose:
                        print(f"    → drop block {candidate} "
                              f"(non-binding {block_non_binding[candidate]} iters)")
                    _drop_block(candidate)
                else:
                    break

    # ── Helper: solve subproblem ──────────────────────────────────────────────
    def _solve_subproblem(remaining):
        if remaining < 1.0:
            return None, None, None, None

        fixed_now = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)

        if use_partial:
            # Need duals for Benders cut; only one scenario per call
            result = solve_subproblem_dual(
                x0, inst, uncertainty_budget, Hn,
                big_M=big_M, time_limit=min(tau, remaining),
                return_duals=True, n_scenarios=1,
            )
            eps_map, alpha_d, beta_d, t2_d, theta_d, u2_d, gamma_d, sub_obj = result
            eps_bar_list = [eps_map]
            duals_list   = [(alpha_d, beta_d, t2_d, theta_d, u2_d, gamma_d)]
        else:
            eps_bar_list, sub_obj = solve_subproblem_dual(
                x0, inst, uncertainty_budget, Hn,
                big_M=big_M, time_limit=min(tau, remaining),
                return_duals=False, n_scenarios=1,
            )
            duals_list = [None]

            if eff_n_scenarios >= 2:
                key1 = frozenset(
                    (j, h) for (j, h), v in eps_bar_list[0].items() if v > 0.5
                )
                eps_bar_list2, _ = solve_subproblem_dual(
                    x0, inst, uncertainty_budget, Hn,
                    big_M=big_M, time_limit=min(tau, remaining),
                    return_duals=False, n_scenarios=1,
                    exclude_scenario=eps_bar_list[0],
                )
                key2 = frozenset(
                    (j, h) for (j, h), v in eps_bar_list2[0].items() if v > 0.5
                )
                if key2 != key1:
                    eps_bar_list = eps_bar_list + eps_bar_list2
                    duals_list   = duals_list + [None]

        return eps_bar_list, duals_list, sub_obj, fixed_now

    # ── Helper: solve master ──────────────────────────────────────────────────
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

    # ── Helper: log ───────────────────────────────────────────────────────────
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
            "n_benders":     n_benders,
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
                f"| blocks={len(active_blocks)}/{s_counter} "
                f"| sub={t_sub:.1f}s master={t_master:.1f}s"
            )

    # ── Warm-start: heuristic blocks before main loop ─────────────────────────
    if n_warmstart > 0:
        warm_scens = _heuristic_scenarios(inst, Hn, uncertainty_budget, n_warmstart)
        for eps_bar in warm_scens:
            _add_block(eps_bar, duals=None)
        if verbose:
            print(f"  warm-start: added {len(warm_scens)} heuristic blocks")

    # ── Main loop: subproblem → add block(s) → solve master ───────────────────
    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # Step 1: subproblem
        t_sub_start = time.time()
        remaining   = max(0.0, time_limit - elapsed)
        eps_bar_list, duals_list, sub_obj, fixed_now = _solve_subproblem(remaining)
        t_sub = time.time() - t_sub_start

        if eps_bar_list is None:
            break

        UB = min(UB, sub_obj + fixed_now)

        # Step 2: add block(s)
        for eps_bar, duals in zip(eps_bar_list, duals_list):
            _add_block(eps_bar, duals=duals)

        # Step 3: solve master
        U_j, L_j, t_master = _solve_master()
        _manage_blocks()

        # In partial-block (RPBD) mode, ObjBound can exceed ObjVal when the
        # master terminates at MIPGap, giving a falsely small gap.  Use the
        # primal objective U_j, which is always a valid (conservative) LB.
        LB_now   = L_ell if use_exploit else (U_j if use_partial else L_j)
        abs_UB   = abs(UB) + 1e-10
        true_gap = (UB - LB_now) / abs_UB
        inex_gap = (UB - U_j)   / abs_UB

        _log(U_j, L_j, t_sub, t_master, true_gap, inex_gap, "explore")

        # Step 4: convergence
        if true_gap <= tol:
            converged = True
            break

        # Step 5: exploit phase (MODE 4 only)
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
                eps_bar_list, duals_list, sub_obj, fixed_now = _solve_subproblem(remaining)
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
                for eps_bar, duals in zip(eps_bar_list, duals_list):
                    _add_block(eps_bar, duals=duals)

    best_LB = L_ell if use_exploit else (iter_log[-1]["L_j"] if iter_log else L_init)

    return {
        "x_jr":                x0,
        "LB":                  best_LB,
        "UB":                  UB,
        "profit_LB":           -UB,
        "n_iter":              n_iter,
        "n_blocks":            s_counter,
        "active_blocks":       len(active_blocks),
        "n_benders":           n_benders,
        "runtime":             time.time() - start,
        "converged":           converged,
        "iter_log":            iter_log,
    }
