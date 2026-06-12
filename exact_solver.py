"""
exact_solver.py — Two tools for the HIV EID VRP
================================================

1. lp_lower_bound()
   -----------------
   Solves the continuous LP/SOCP relaxation (all binary/integer variables
   relaxed to continuous).  Runs in < 1 second and returns a valid lower
   bound on the optimal objective.  Use this to score SA solution quality.

2. branch_and_cut_solver()
   ------------------------
   Exact branch-and-cut solver.  The rotated SOC  D[r]² ≤ H[r]·β[r]  is
   handled by Gurobi *lazy-cut callbacks*: OA cuts are injected into the
   B&B tree the moment an integer node violates the SOC, rather than
   re-solving the full MILP from scratch after every cut (the iterative
   L-shaped approach).  This keeps one single B&B tree alive throughout,
   letting Gurobi reuse its incumbent, dual bounds, and branching history.

   Optionally accepts a warm-start solution (e.g. from the SA) to:
     (a) seed an initial incumbent,
     (b) add OA cuts at the warm-start point before the first solve,
         immediately tightening the LP relaxation.

Both functions return the same 14-element list as gurobi_solver().
"""

import time
import gurobipy as gp
from gurobipy import GRB


# =============================================================================
# Shared model builder (routing constraints, no SOC on β)
# =============================================================================

def _build_master(obj, mode, C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,
                  relax=False):
    """
    Build the routing MILP/LP with β[r] ≥ 0 (no SOC).

    Parameters
    ----------
    relax : bool
        If True, all binary / integer variables are relaxed to continuous
        (for the LP lower bound).

    Returns
    -------
    m, vars_dict
        Gurobi model and a dict of the key variable objects.
    """
    rho        = sum(demand[c] for c in C) / (K * mu)
    coeff_beta = 1.0 / (2.0 * K * K * mu * mu * (1.0 - rho))

    m = gp.Model()
    m.Params.OutputFlag = 0

    vtype_b = 'C' if relax else 'B'
    vtype_i = 'C' if relax else GRB.INTEGER

    x    = m.addVars(N, N, R, vtype=vtype_b, lb=0, ub=1)
    y    = m.addVars(C, R,    vtype=vtype_b, lb=0, ub=1)

    if mode == "Single":
        z_var = None
        z     = [1] * Rn
    elif mode == "Multiple":
        z_var = m.addVars(R, vtype=vtype_i, lb=0, ub=max(V))
        zeta  = m.addVars(R, V, vtype=vtype_b, lb=0, ub=1)
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
    beta  = m.addVars(R, vtype='C', lb=0)   # no SOC yet

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
            m.addConstr(U[i,r] == L[r] - T[i,r])
            if relax:
                # Big-M linearisation for LP (indicator constraints not valid
                # in LP mode; use explicit McCormick-style bounds instead)
                M_H = sum(t[i,j] for i in N for j in N) + 1
                m.addConstr(eta[i,r]   <= H[r])
                m.addConstr(eta[i,r]   <= M_H * y[i,r])
                m.addConstr(eta[i,r]   >= H[r] - M_H*(1 - y[i,r]))
                m.addConstr(theta[i,r] <= U[i,r])
                m.addConstr(theta[i,r] <= M_H * y[i,r])
                m.addConstr(theta[i,r] >= U[i,r] - M_H*(1 - y[i,r]))
            else:
                m.addConstr((y[i,r]==1) >> (eta[i,r]   == H[r]))
                m.addConstr((y[i,r]==0) >> (eta[i,r]   == 0))
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
                if relax:
                    M_H = sum(t[i,j] for i in N for j in N) + 1
                    m.addConstr(delta[r,v] <= H[r])
                    m.addConstr(delta[r,v] <= M_H * zeta[r,v])
                    m.addConstr(delta[r,v] >= H[r] - M_H*(1 - zeta[r,v]))
                else:
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
                if relax:
                    M_T = sum(t[i,j] for i in N for j in N) + 1
                    m.addConstr(T[i,r] + t[i,j] - T[j,r] <= M_T*(1 - x[i,j,r]))
                else:
                    m.addConstr((x[i,j,r]==1) >> (T[i,r] + t[i,j] - T[j,r] == 0))

        # SOC as actual constraint for LP relaxation (SOCP → still convex)
        if relax:
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


# =============================================================================
# 1. LP / SOCP lower bound
# =============================================================================

def lp_lower_bound(
    obj, mode,
    C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,
    time_limit = 30,
    verbose    = False,
):
    """
    Solve the continuous LP/SOCP relaxation to get a fast valid lower bound.

    All binary and integer variables are relaxed to [0,1] / [0, max(V)].
    The SOC  D[r]²  ≤  H[r]·β[r]  is kept as a convex constraint, so the
    relaxation is a SOCP (solved in milliseconds by Gurobi).

    Returns
    -------
    lb : float
        Valid lower bound on the optimal objective.
    solve_time : float
        Wall-clock solve time in seconds.
    """
    t0 = time.time()
    m, vd = _build_master(obj, mode, C, N, R, t, demand, Q, V, K, mu,
                           SCV, Cn, Vn, Rn, relax=True)
    if verbose:
        m.Params.OutputFlag = 1
    m.Params.TimeLimit = time_limit
    m.optimize()

    solve_time = time.time() - t0

    if m.SolCount == 0:
        return -float('inf'), solve_time

    lb = m.ObjVal
    if verbose:
        print(f"  LP lower bound: {lb:.4f}  (solved in {solve_time:.2f}s)")
    return lb, solve_time


# =============================================================================
# 2. Branch-and-cut with lazy OA callbacks
# =============================================================================

def branch_and_cut_solver(
    obj, mode,
    C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,
    warm_start  = None,    # result list from SA / any prior solver
    time_limit  = 300,
    tol         = 1e-4,
    verbose     = True,
):
    """
    Exact branch-and-cut solver using Gurobi lazy-cut callbacks.

    The SOC  D[r]² ≤ H[r]·β[r]  is enforced by injecting OA cuts
    *inside the running B&B tree* every time Gurobi finds an integer node
    that violates the SOC — no separate MILP re-solve needed.

    Compared with the iterative L-shaped:
      • One B&B tree lives throughout (Gurobi reuses bounds/heuristics).
      • Cuts are added exactly where violations occur.
      • Warm-start + initial cuts tighten the LP relaxation immediately.

    Parameters
    ----------
    warm_start : list or None
        14-element result list (e.g. from metaheuristic_solver).
        Used to seed an MIP start and add pre-emptive OA cuts.
    """
    start_time = time.time()
    rho        = sum(demand[c] for c in C) / (K * mu)
    coeff_beta = 1.0 / (2.0 * K * K * mu * mu * (1.0 - rho))

    m, vd = _build_master(obj, mode, C, N, R, t, demand, Q, V, K, mu,
                           SCV, Cn, Vn, Rn, relax=False)

    x, y, L, H, D, T = vd['x'], vd['y'], vd['L'], vd['H'], vd['D'], vd['T']
    eta, theta, beta  = vd['eta'], vd['theta'], vd['beta']
    max_TLT           = vd['max_TLT']
    z_var             = vd['z']

    # ── Pre-emptive OA cuts + MIP warm-start from external solution ──────────
    if warm_start is not None:
        ws_H   = warm_start[5]   # time_between_visits {r: H_r}
        ws_y   = warm_start[3]   # allocations {(i,r): 0/1}
        ws_L   = warm_start[1]   # lengthes {r: L_r}
        ws_sol = warm_start[0]   # solution {(i,j,r): 1}
        ws_z   = warm_start[4]   # num_vehicles {r: z_r}
        ws_T   = warm_start[2]   # start_times {(i,r): T_ir}
        ws_eta = {(i,r): ws_y.get((i,r),0)*ws_H.get(r,0) for i in C for r in R}

        # Compute D* from warm-start
        ws_D = {r: sum(demand[i]*ws_eta[i,r] for i in C) for r in R}

        n_pre = 0
        for r in R:
            H_r = ws_H.get(r, 0.0)
            D_r = ws_D[r]
            if H_r > 1e-8:
                slope = 2.0 * D_r / H_r
                curv  = (D_r / H_r) ** 2
                m.addConstr(beta[r] >= slope*D[r] - curv*H[r],
                            name=f"oa_ws_r{r}")
                n_pre += 1

        # MIP start hints
        for (i,j,r), val in ws_sol.items():
            x[i,j,r].Start = float(val)
        for (i,r), val in ws_y.items():
            y[i,r].Start = float(val > 0.5)
        for r in R:
            L[r].Start = ws_L.get(r, 0.0)
            H[r].Start = ws_H.get(r, 0.0)
            D[r].Start = ws_D[r]
        for (i,r), val in ws_T.items():
            if i in [nd for nd in N]:
                T[i,r].Start = float(val)
        if mode == "Multiple" and z_var is not None:
            for r in R:
                z_var[r].Start = float(ws_z.get(r, 1))

        if verbose:
            print(f"  Warm-start loaded | {n_pre} pre-emptive OA cuts added.")

    # ── Lazy-cut callback ────────────────────────────────────────────────────
    m.Params.LazyConstraints = 1
    m.Params.TimeLimit        = time_limit

    cut_counter = [0]   # mutable counter accessible inside closure

    def _callback(where):
        if where != GRB.Callback.MIPSOL:
            return

        D_v    = m.cbGetSolution([D[r]    for r in R])
        H_v    = m.cbGetSolution([H[r]    for r in R])
        beta_v = m.cbGetSolution([beta[r] for r in R])

        for idx, r in enumerate(R):
            H_r = H_v[idx];  D_r = D_v[idx];  b_r = beta_v[idx]
            if H_r < 1e-8:
                continue
            f_r = D_r * D_r / H_r
            if f_r - b_r > tol:
                slope = 2.0 * D_r / H_r
                curv  = (D_r / H_r) ** 2
                m.cbLazy(beta[r] >= slope*D[r] - curv*H[r])
                cut_counter[0] += 1

    if verbose:
        m.Params.OutputFlag = 1
        print(f"\n{'─'*65}")
        print(f"  BRANCH-AND-CUT  |  obj={obj}  mode={mode}  "
              f"Cn={Cn}  Rn={Rn}  Vn={Vn}")
        print(f"  ρ={rho:.4f}   K={K}   μ={mu}   SCV={SCV}")
        print(f"{'─'*65}")

    m.optimize(_callback)

    solve_time = time.time() - start_time

    if m.SolCount == 0:
        raise RuntimeError(
            "Branch-and-cut found no feasible solution.\n"
            "Try increasing time_limit or check instance parameters."
        )

    # ── Extract best solution ────────────────────────────────────────────────
    L_val   = {r: L[r].x    for r in R}
    H_val   = {r: H[r].x    for r in R}
    D_val   = {r: D[r].x    for r in R}
    eta_val = {(i,r): eta[i,r].x   for i in C for r in R}
    tht_val = {(i,r): theta[i,r].x for i in C for r in R}

    # Compute exact W_lab from recovered solution
    phi_actual = sum(D_val[r]**2 / H_val[r] for r in R if H_val[r] > 1e-8)
    W_lab = ((K-1)/(2*K*mu*rho)
             + phi_actual * coeff_beta
             + SCV*rho/(2*mu*(1-rho))
             + 1.0/mu)

    TATs = {i: sum(0.5*eta_val[i,r] + tht_val[i,r] for r in R) + W_lab
            for i in C}

    TRL  = sum(L_val.values())
    MRL  = max(L_val.values())
    ATAT = sum(TATs.values()) / Cn
    MTAT = max(TATs.values())

    # Use actual objective (SOC satisfied by cuts → ObjVal ≈ true obj)
    if obj in ("ATAT_Min", "MTAT_Min"):
        obj_val = ATAT if obj == "ATAT_Min" else MTAT
    else:
        obj_val = m.ObjVal

    LB  = m.ObjBound
    gap = abs(obj_val - LB) / abs(obj_val) * 100 if obj_val != 0 else 0.0

    if verbose:
        print(f"\n{'='*65}")
        print(f"  B&C RESULT   obj={obj}  mode={mode}")
        print(f"  Objective      : {obj_val:.4f}")
        print(f"  Lower bound    : {LB:.4f}")
        print(f"  Gap            : {gap:.4f}%")
        print(f"  ATAT           : {ATAT:.4f}")
        print(f"  MTAT           : {MTAT:.4f}")
        print(f"  TRL            : {TRL:.4f}")
        print(f"  MRL            : {MRL:.4f}")
        print(f"  W_lab          : {W_lab:.4f}")
        print(f"  Lazy OA cuts   : {cut_counter[0]}")
        print(f"  Time           : {solve_time:.2f}s")
        print(f"  B&B nodes      : {int(m.NodeCount)}")
        print(f"{'='*65}")

    solution   = {(i,j,r): 1 for (i,j,r) in x.keys() if x[i,j,r].x > 0.5}
    start_times= {(i,r): T[i,r].x for i in N for r in R}
    allocations= {(i,r): y[i,r].x for i in C for r in R}
    num_veh    = ({r: z_var[r].x for r in R} if mode=="Multiple"
                  else {r: 1 for r in R})

    diagnostics = {
        "LB":              LB,
        "UB":              obj_val,
        "gap_pct":         gap,
        "n_lazy_cuts":     cut_counter[0],
        "n_bb_nodes":      int(m.NodeCount),
        "solution_time":   solve_time,
        "gurobi_status":   m.Status,
    }

    return [
        solution,     # 0
        L_val,        # 1
        start_times,  # 2
        allocations,  # 3
        num_veh,      # 4
        H_val,        # 5
        TATs,         # 6
        obj_val,      # 7
        solve_time,   # 8
        TRL,          # 9
        MRL,          # 10
        ATAT,         # 11
        MTAT,         # 12
        diagnostics,  # 13
    ]
