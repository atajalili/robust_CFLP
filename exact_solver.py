"""
exact_solver.py
===============

1. fast_lower_bound()
   -------------------
   Runs the full MIQCP (SOC kept) with a short time limit.
   Returns Gurobi's dual bound (m.ObjBound) — valid even when the solve
   is cut off early.  Tight in seconds because the SOC tightens the LP
   relaxation just as in the full solve.

2. warm_start_solver()
   --------------------
   Full MIQCP solver (identical to gurobi_solver) with an optional
   warm-start from any prior solution (e.g. SA).  The warm-start:
     (a) Seeds a good initial incumbent → Gurobi prunes the B&B tree
         more aggressively from the first node.
     (b) Tightens the MIP gap tolerance from the incumbent side,
         so optimality is proved faster.
   Reports LB, UB, gap and solution time.

Why not the iterative L-shaped / lazy-cut approach?
   Removing the SOC from the master weakens the LP bound drastically
   (e.g. 0.8 vs optimal 80.5), forcing tens of thousands of B&B nodes.
   The original MIQCP keeps the SOC in every LP relaxation node, which
   is exactly what makes Gurobi fast.  The correct strategy is to keep
   the SOC and reduce solve time via warm-starting.
"""

import time
import gurobipy as gp
from gurobipy import GRB


def _build_miqcp(obj, mode, C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn):
    """
    Build the full MIQCP (with SOC) — identical to gurobi_solver but
    returns the model and variable dict for external use.
    """
    rho        = sum(demand[c] for c in C) / (K * mu)
    coeff_beta = 1.0 / (2.0 * K * K * mu * mu * (1.0 - rho))

    m = gp.Model()
    m.Params.OutputFlag = 0

    x     = m.addVars(N, N, R, vtype='B')
    y     = m.addVars(C, R,    vtype='B')

    if mode == "Single":
        z_var = None
        z     = [1] * Rn
    elif mode == "Multiple":
        z_var = m.addVars(R, vtype=GRB.INTEGER, lb=0)
        zeta  = m.addVars(R, V, vtype='B')
        delta = m.addVars(R, V, vtype='C', lb=0)
        z     = z_var

    L     = m.addVars(R, vtype='C', lb=0)
    H     = m.addVars(R, vtype='C', lb=0)
    D     = m.addVars(R, vtype='C', lb=0)
    U     = m.addVars(C, R, vtype='C', lb=0)
    q     = m.addVars(C, R, vtype='C', lb=0)
    T     = m.addVars(N, R, vtype='C', lb=0)
    eta   = m.addVars(C, R, vtype='C', lb=0)
    theta = m.addVars(C, R, vtype='C')
    beta  = m.addVars(R, vtype='C', lb=0)

    TLT     = m.addVars(C, vtype='C', lb=0)
    max_L   = m.addVar(vtype='C', lb=0)
    max_TLT = m.addVar(vtype='C', lb=0)

    # ── Objective ─────────────────────────────────────────────────────────────
    if obj == "TRL_Min":
        m.setObjective(gp.quicksum(L[r] for r in R), GRB.MINIMIZE)
    elif obj == "MRL_Min":
        m.setObjective(max_L, GRB.MINIMIZE)
    elif obj == "ATAT_Min":
        m.setObjective(
            gp.quicksum(0.5*eta[i,r] + theta[i,r] for i in C for r in R) / Cn
            + (K-1)/(2*K*mu*rho)
            + coeff_beta * gp.quicksum(beta[r] for r in R)
            + SCV*rho/(2*mu*(1-rho)) + 1.0/mu,
            GRB.MINIMIZE)
    elif obj == "MTAT_Min":
        m.setObjective(
            max_TLT
            + (K-1)/(2*K*mu*rho)
            + coeff_beta * gp.quicksum(beta[r] for r in R)
            + SCV*rho/(2*mu*(1-rho)) + 1.0/mu,
            GRB.MINIMIZE)

    # ── Constraints ───────────────────────────────────────────────────────────
    for i in C:
        m.addConstr(gp.quicksum(x[i,j,r] for j in N for r in R) == 1)
        if obj == "MTAT_Min":
            m.addConstr(TLT[i] == gp.quicksum(0.5*eta[i,r] + theta[i,r] for r in R))
            m.addConstr(TLT[i] <= max_TLT)
        for r in R:
            m.addConstr(gp.quicksum(x[i,j,r] for j in N) ==
                        gp.quicksum(x[j,i,r] for j in N))
            m.addConstr(y[i,r] == gp.quicksum(x[i,j,r] for j in N))
            m.addConstr(q[i,r] == demand[i]*eta[i,r] + y[i,r])
            m.addConstr((y[i,r]==1) >> (eta[i,r]   == H[r]))
            m.addConstr((y[i,r]==0) >> (eta[i,r]   == 0))
            m.addConstr(U[i,r] == L[r] - T[i,r])
            m.addConstr((y[i,r]==1) >> (theta[i,r] == U[i,r]))
            m.addConstr((y[i,r]==0) >> (theta[i,r] == 0))

    for r in R:
        m.addConstr(gp.quicksum(x[0,j,r]    for j in N if j != 0)    == 1)
        m.addConstr(gp.quicksum(x[i,Cn+1,r] for i in N if i != Cn+1) == 1)
        m.addConstr(L[r] == gp.quicksum(t[i,j]*x[i,j,r] for i in N for j in N))

        if mode == "Single":
            m.addConstr(L[r] == H[r])
        elif mode == "Multiple":
            m.addConstr(z[r] == gp.quicksum(v*zeta[r,v] for v in V))
            m.addConstr(gp.quicksum(zeta[r,v] for v in V) <= 1)
            m.addConstr(L[r] == gp.quicksum(v*delta[r,v] for v in V))
            for v in V:
                m.addConstr((zeta[r,v]==1) >> (delta[r,v] == H[r]))
                m.addConstr((zeta[r,v]==0) >> (delta[r,v] == 0))

        m.addConstr(gp.quicksum(q[i,r] for i in C) <= Q)
        m.addConstr(T[0,r]    == 0)
        m.addConstr(T[Cn+1,r] == L[r])
        m.addConstr(D[r] == gp.quicksum(demand[i]*eta[i,r] for i in C))

        if obj == "MRL_Min":
            m.addConstr(L[r] <= max_L)

        for i in N:
            m.addConstr(x[i,i,r] == 0)
            for j in N:
                m.addConstr((x[i,j,r]==1) >> (T[i,r] + t[i,j] - T[j,r] == 0))

        # SOC — kept as a proper constraint (critical for tight LP relaxation)
        m.addQConstr(D[r]*D[r] <= H[r]*beta[r])

    m.addConstr(gp.quicksum(z[r] for r in R) <= Vn)

    for r in range(Rn - 1):
        m.addConstr(gp.quicksum(y[i,r] for i in C) >=
                    gp.quicksum(y[i,r+1] for i in C))

    vd = dict(x=x, y=y, z=z_var, L=L, H=H, D=D, U=U, q=q, T=T,
              eta=eta, theta=theta, beta=beta,
              TLT=TLT, max_L=max_L, max_TLT=max_TLT,
              rho=rho, coeff_beta=coeff_beta)
    if mode == "Multiple":
        vd.update(zeta=zeta, delta=delta)
    return m, vd


def _set_warm_start(m, vd, warm_start, C, N, R, mode, demand, V):
    """Inject MIP start hints from a prior solution (all variables)."""
    ws_x = warm_start[0]   # {(i,j,r): 1}
    ws_L = warm_start[1]   # {r: L_r}
    ws_T = warm_start[2]   # {(i,r): T_ir}
    ws_y = warm_start[3]   # {(i,r): 0/1}
    ws_z = warm_start[4]   # {r: z_r}
    ws_H = warm_start[5]   # {r: H_r}

    x, y, L, H, T = vd['x'], vd['y'], vd['L'], vd['H'], vd['T']
    eta, theta, beta, D, U, q = (vd['eta'], vd['theta'], vd['beta'],
                                  vd['D'], vd['U'], vd['q'])

    # Derived warm-start values
    ws_y_bin = {(i,r): float(ws_y.get((i,r), 0) > 0.5) for i in C for r in R}
    ws_eta   = {(i,r): ws_y_bin[i,r] * ws_H.get(r, 0)  for i in C for r in R}
    ws_U     = {(i,r): ws_L.get(r, 0) - ws_T.get((i,r), 0) for i in C for r in R}
    ws_theta = {(i,r): ws_y_bin[i,r] * ws_U[i,r]           for i in C for r in R}
    ws_D     = {r: sum(demand[i] * ws_eta[i,r] for i in C) for r in R}
    # Add a small slack so D²≤H·β is strictly satisfied in the MIP start check
    ws_beta  = {r: (ws_D[r]**2 / ws_H[r] * (1.0 + 1e-6) if ws_H.get(r, 0) > 1e-8 else 0.0)
                for r in R}
    ws_q     = {(i,r): demand[i] * ws_eta[i,r] + ws_y_bin[i,r]
                for i in C for r in R}

    for (i,j,r) in x.keys():
        x[i,j,r].Start = float(ws_x.get((i,j,r), 0))
    for (i,r) in y.keys():
        y[i,r].Start = ws_y_bin[i,r]
    for r in R:
        L[r].Start = float(ws_L.get(r, 0))
        H[r].Start = float(ws_H.get(r, 0))
        D[r].Start = ws_D[r]
        beta[r].Start = ws_beta[r]
    for (i,r) in T.keys():
        T[i,r].Start = float(ws_T.get((i,r), 0))
    for (i,r) in eta.keys():
        eta[i,r].Start   = ws_eta[i,r]
        theta[i,r].Start = ws_theta[i,r]
        U[i,r].Start     = ws_U[i,r]
        q[i,r].Start     = ws_q[i,r]

    if mode == "Multiple" and vd['z'] is not None:
        zeta  = vd['zeta']
        delta = vd['delta']
        for r in R:
            z_val = int(round(ws_z.get(r, 1)))
            vd['z'][r].Start = float(z_val)
            for v in V:
                zeta[r,v].Start  = float(v == z_val)
                delta[r,v].Start = (float(ws_H.get(r, 0))
                                    if v == z_val else 0.0)


# =============================================================================
# 1. Fast lower bound  (short-time-limit MIQCP)
# =============================================================================

def fast_lower_bound(
    obj, mode,
    C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,
    warm_start = None,
    time_limit = 30,
    verbose    = False,
):
    """
    Run the full MIQCP with a short time limit and return Gurobi's dual
    bound (m.ObjBound) as a valid lower bound.

    The SOC constraint is kept — this is what makes the LP relaxation
    tight, giving a useful bound even after just a few seconds.

    Parameters
    ----------
    warm_start : list, optional
        Prior solution (SA / any solver) used to seed a good incumbent,
        which tightens the bound faster.
    time_limit : float
        Seconds to run.  30 s is usually enough for a useful LB.

    Returns
    -------
    lb         : float  — valid lower bound on the optimal objective
    solve_time : float  — wall-clock seconds used
    """
    t0 = time.time()
    m, vd = _build_miqcp(obj, mode, C, N, R, t, demand, Q, V, K, mu,
                          SCV, Cn, Vn, Rn)
    if warm_start is not None:
        _set_warm_start(m, vd, warm_start, C, N, R, mode, demand, V)

    if verbose:
        m.Params.OutputFlag = 1

    m.Params.TimeLimit = time_limit
    m.update()
    m.optimize()

    solve_time = time.time() - t0
    lb         = m.ObjBound if m.SolCount >= 0 else -float('inf')
    ub         = m.ObjVal   if m.SolCount >  0 else  float('inf')

    if verbose or True:
        print(f"  fast_lower_bound | LB={lb:.4f}  "
              f"UB={ub:.4f}  time={solve_time:.1f}s")
    return lb, solve_time


# =============================================================================
# 2. Warm-start exact solver
# =============================================================================

def warm_start_solver(
    obj, mode,
    C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,
    warm_start = None,
    time_limit = 300,
    mip_gap    = 1e-4,
    verbose    = True,
):
    """
    Full MIQCP solver (identical model to gurobi_solver) with optional
    warm-start from any prior solution.

    The warm-start seeds Gurobi's MIP start with the SA (or any heuristic)
    solution, immediately providing a tight incumbent.  Gurobi then only
    needs to prove optimality, not discover the solution from scratch.

    Parameters
    ----------
    warm_start : list, optional
        14-element result list from metaheuristic_solver / gurobi_solver.
    mip_gap    : float
        Relative MIP gap tolerance (default 0.01%).

    Returns
    -------
    Standard 14-element list (same as gurobi_solver).
    """
    start_time = time.time()
    rho        = sum(demand[c] for c in C) / (K * mu)
    coeff_beta = 1.0 / (2.0 * K * K * mu * mu * (1.0 - rho))

    m, vd = _build_miqcp(obj, mode, C, N, R, t, demand, Q, V, K, mu,
                          SCV, Cn, Vn, Rn)

    if warm_start is not None:
        _set_warm_start(m, vd, warm_start, C, N, R, mode, demand, V)
        if verbose:
            print(f"  Warm-start loaded (SA obj = {warm_start[7]:.4f})")
        # Good incumbent available — focus on proving optimality, skip heuristics
        m.Params.MIPFocus   = 3
        m.Params.Heuristics = 0.0

    m.Params.TimeLimit = time_limit
    m.Params.MIPGap    = mip_gap
    if verbose:
        m.Params.OutputFlag = 1

    m.update()
    m.optimize()

    solve_time = time.time() - start_time

    if verbose and warm_start is not None:
        # Report whether Gurobi actually used the MIP start
        # (look for "MIP start 1 is feasible" in the log above)
        print(f"  MIP starts used: {m.SolCount}  "
              f"(start accepted if log shows 'MIP start 1 is feasible')")

    if m.SolCount == 0:
        raise RuntimeError("warm_start_solver found no feasible solution.")

    x, y, L, H, D = vd['x'], vd['y'], vd['L'], vd['H'], vd['D']
    T, eta, theta, beta = vd['T'], vd['eta'], vd['theta'], vd['beta']
    z_var = vd['z']

    L_val   = {r: L[r].x    for r in R}
    H_val   = {r: H[r].x    for r in R}
    eta_val = {(i,r): eta[i,r].x   for i in C for r in R}
    tht_val = {(i,r): theta[i,r].x for i in C for r in R}

    phi_actual = sum(D[r].x**2 / H_val[r] for r in R if H_val[r] > 1e-8)
    W_lab = ((K-1)/(2*K*mu*rho)
             + phi_actual * coeff_beta
             + SCV*rho/(2*mu*(1-rho))
             + 1.0/mu)

    TATs  = {i: sum(0.5*eta_val[i,r] + tht_val[i,r] for r in R) + W_lab
             for i in C}
    TRL   = sum(L_val.values())
    MRL   = max(L_val.values())
    ATAT  = sum(TATs.values()) / Cn
    MTAT  = max(TATs.values())
    LB    = m.ObjBound
    UB    = ATAT if obj == "ATAT_Min" else MTAT if obj == "MTAT_Min" else m.ObjVal
    gap   = abs(UB - LB) / abs(UB) * 100 if UB != 0 else 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  WARM-START SOLVER  obj={obj}  mode={mode}")
        print(f"  Objective  : {UB:.4f}")
        print(f"  Lower bound: {LB:.4f}   Gap: {gap:.4f}%")
        print(f"  ATAT       : {ATAT:.4f}   MTAT: {MTAT:.4f}")
        print(f"  W_lab      : {W_lab:.4f}")
        print(f"  Time       : {solve_time:.2f}s   Nodes: {int(m.NodeCount)}")
        print(f"{'='*60}")

    solution    = {(i,j,r): 1 for (i,j,r) in x.keys() if x[i,j,r].x > 0.5}
    start_times = {(i,r): T[i,r].x for i in N for r in R}
    allocations = {(i,r): y[i,r].x for i in C for r in R}
    num_veh     = ({r: z_var[r].x for r in R} if mode == "Multiple"
                   else {r: 1 for r in R})

    diagnostics = {
        "LB": LB, "UB": UB, "gap_pct": gap,
        "n_bb_nodes": int(m.NodeCount),
        "solution_time": solve_time,
        "gurobi_status": m.Status,
        "warm_started": warm_start is not None,
    }

    return [
        solution, L_val, start_times, allocations, num_veh,
        H_val, TATs, UB, solve_time, TRL, MRL, ATAT, MTAT,
        diagnostics,
    ]
