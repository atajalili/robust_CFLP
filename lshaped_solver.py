"""
L-Shaped (Outer Approximation) solver for HIV Early Infant Diagnosis VRP
========================================================================
Replaces the rotated SOC constraint  D[r]² ≤ H[r]·β[r]  with iterative
supporting-hyperplane (tangent) cuts, converting the MIQCP into a sequence
of MILPs that Gurobi solves much faster per iteration.

Theory
------
The variable part of W_lab is  φ = Σ_r D_r²/H_r.
The function f(D,H) = D²/H is convex in (D,H) for H>0.
Its supporting hyperplane at (D*, H*) is:

    β[r] ≥  2·(D*/H*) · D[r]  −  (D*/H*)² · H[r]          (OA cut)

This is LINEAR in the master variables (D[r] and H[r] are already model
variables linked via  D[r] = Σ demand[i]·η[i,r]  and the vehicle-count
constraints).  Each cut is globally valid and the master objective value
is a non-decreasing lower bound as cuts accumulate.

Algorithm
---------
1.  Build master MIP (routing constraints identical to gurobi_solver,
    but β[r] ≥ 0 with NO SOC — starts as a pure MILP).
2.  Solve master  →  extract (D_r*, H_r*, β_r*).
3.  Compute actual  f_r = D_r*² / H_r*  for each route.
4.  UB = min(UB, actual objective);  LB = max(LB, master ObjBound).
5.  If  max_r(f_r − β_r*) ≤ tol  and  (UB−LB)/UB ≤ gap_tol:  STOP.
6.  For each route r where  f_r > β_r* + tol:
        add  β[r] ≥ 2·(D_r*/H_r*)·D[r] − (D_r*/H_r*)²·H[r]
7.  Go to 2.

Output
------
Same 13-element list as gurobi_solver() plus diagnostics dict at index 13.
Compatible with the notebook's check_feasibility and objective_breakdown cells.
"""

import time
import gurobipy as gp
from gurobipy import GRB


def lshaped_solver(
    obj, mode,
    C, N, R, t, demand, Q, V, K, mu, SCV, Cn, Vn, Rn,
    max_iter   = 50,
    tol        = 1e-4,     # SOC violation tolerance
    gap_tol    = 1e-3,     # relative optimality gap (0.1 %)
    time_limit = 300,      # seconds
    verbose    = True,
):
    """
    Drop-in replacement for gurobi_solver() for ATAT_Min and MTAT_Min.
    TRL_Min / MRL_Min fall back to a single Gurobi solve (no SOC in those
    objectives so L-shaped gives no benefit).
    """
    start_time = time.time()
    rho        = sum(demand[c] for c in C) / (K * mu)
    coeff_beta = 1.0 / (2.0 * K * K * mu * mu * (1.0 - rho))

    # =========================================================================
    # 1.  Build master model
    # =========================================================================
    m = gp.Model()
    m.Params.OutputFlag = 0           # suppress Gurobi log; we print our own

    # --- Decision variables (identical to gurobi_solver) ---------------------
    x    = m.addVars(N, N, R, vtype='B')
    y    = m.addVars(C, R, vtype='B')

    if mode == "Single":
        z = [1] * Rn
    elif mode == "Multiple":
        z     = m.addVars(R, vtype=GRB.INTEGER, lb=0)
        zeta  = m.addVars(R, V, vtype='B')
        delta = m.addVars(R, V, vtype='C', lb=0)

    L     = m.addVars(R, vtype='C', lb=0)
    H     = m.addVars(R, vtype='C', lb=0)
    D     = m.addVars(R, vtype='C', lb=0)
    U     = m.addVars(C, R, vtype='C', lb=0)
    q     = m.addVars(C, R, vtype='C', lb=0)
    T     = m.addVars(N, R, vtype='C', lb=0)
    eta   = m.addVars(C, R, vtype='C', lb=0)
    theta = m.addVars(C, R, vtype='C')

    # β[r] ≥ 0 — no SOC, grows tighter with each OA cut
    beta  = m.addVars(R, vtype='C', lb=0)

    TLT     = m.addVars(C, vtype='C', lb=0)
    max_L   = m.addVar(vtype='C', lb=0)
    max_TLT = m.addVar(vtype='C', lb=0)

    # --- Objective -----------------------------------------------------------
    if obj == "TRL_Min":
        m.setObjective(gp.quicksum(L[r] for r in R), GRB.MINIMIZE)

    elif obj == "MRL_Min":
        m.setObjective(max_L, GRB.MINIMIZE)

    elif obj == "ATAT_Min":
        m.setObjective(
            gp.quicksum(0.5*eta[i,r] + theta[i,r] for i in C for r in R) / Cn
            + (K - 1) / (2*K*mu*rho)
            + coeff_beta * gp.quicksum(beta[r] for r in R)
            + SCV*rho / (2*mu*(1-rho))
            + 1.0 / mu,
            GRB.MINIMIZE,
        )

    elif obj == "MTAT_Min":
        m.setObjective(
            max_TLT
            + (K - 1) / (2*K*mu*rho)
            + coeff_beta * gp.quicksum(beta[r] for r in R)
            + SCV*rho / (2*mu*(1-rho))
            + 1.0 / mu,
            GRB.MINIMIZE,
        )

    # --- Constraints (identical to gurobi_solver) ----------------------------
    for i in C:
        m.addConstr(gp.quicksum(x[i,j,r] for j in N for r in R) == 1)   # (6)

        if obj == "MTAT_Min":
            m.addConstr(TLT[i] == gp.quicksum(0.5*eta[i,r] + theta[i,r] for r in R))
            m.addConstr(TLT[i] <= max_TLT)

        for r in R:
            m.addConstr(gp.quicksum(x[i,j,r] for j in N) ==
                        gp.quicksum(x[j,i,r] for j in N))                # (9)
            m.addConstr(y[i,r] == gp.quicksum(x[i,j,r] for j in N))     # (1)
            m.addConstr(q[i,r] == demand[i]*eta[i,r] + y[i,r])           # (27)
            m.addConstr((y[i,r] == 1) >> (eta[i,r]   == H[r]))
            m.addConstr((y[i,r] == 0) >> (eta[i,r]   == 0))
            m.addConstr(U[i,r] == L[r] - T[i,r])                         # (16)
            m.addConstr((y[i,r] == 1) >> (theta[i,r] == U[i,r]))
            m.addConstr((y[i,r] == 0) >> (theta[i,r] == 0))

    for r in R:
        m.addConstr(gp.quicksum(x[0,j,r]     for j in N if j != 0)    == 1)  # (7)
        m.addConstr(gp.quicksum(x[i,Cn+1,r]  for i in N if i != Cn+1) == 1)  # (8)
        m.addConstr(L[r] == gp.quicksum(t[i,j]*x[i,j,r] for i in N for j in N))  # (2)

        if mode == "Single":
            m.addConstr(L[r] == H[r])                                    # (3)
        elif mode == "Multiple":
            m.addConstr(z[r] == gp.quicksum(v*zeta[r,v] for v in V))    # (24)
            m.addConstr(gp.quicksum(zeta[r,v] for v in V) <= 1)          # (25)
            m.addConstr(L[r] == gp.quicksum(v*delta[r,v] for v in V))   # (26)
            for v in V:
                m.addConstr((zeta[r,v] == 1) >> (delta[r,v] == H[r]))
                m.addConstr((zeta[r,v] == 0) >> (delta[r,v] == 0))

        m.addConstr(gp.quicksum(q[i,r] for i in C) <= Q)                # (11)
        m.addConstr(T[0,r]    == 0)                                      # (13)
        m.addConstr(T[Cn+1,r] == L[r])                                   # (14)
        m.addConstr(D[r] == gp.quicksum(demand[i]*eta[i,r] for i in C)) # (28)

        if obj == "MRL_Min":
            m.addConstr(L[r] <= max_L)

        for i in N:
            m.addConstr(x[i,i,r] == 0)
            for j in N:
                m.addConstr((x[i,j,r] == 1) >> (T[i,r] + t[i,j] - T[j,r] == 0))

    m.addConstr(gp.quicksum(z[r] for r in R) <= Vn)                     # (12)

    for r in range(Rn - 1):                                              # symmetry
        m.addConstr(gp.quicksum(y[i,r] for i in C) >=
                    gp.quicksum(y[i,r+1] for i in C))

    # =========================================================================
    # 2.  L-shaped main loop
    # =========================================================================
    LB           = -float('inf')
    UB           =  float('inf')
    best_snap    = None          # snapshot of best feasible solution found
    iter_log     = []
    n_cuts_total = 0
    gap          = float('inf')

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  L-SHAPED SOLVER  |  obj={obj}  mode={mode}  "
              f"Cn={Cn}  Rn={Rn}  Vn={Vn}")
        print(f"  ρ={rho:.4f}   K={K}   μ={mu}   SCV={SCV}")
        print(f"{'─'*65}")
        print(f"  {'iter':>4}  {'LB':>10}  {'UB':>10}  {'gap%':>7}  "
              f"{'φ_mstr':>9}  {'φ_true':>9}  {'SOC_viol':>9}  {'cuts':>5}")

    for k in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            if verbose:
                print(f"\n  Time limit reached at iteration {k+1}.")
            break

        m.Params.TimeLimit = max(1.0, time_limit - elapsed)
        m.update()
        m.optimize()

        status = m.Status
        if m.SolCount == 0:
            if verbose:
                print(f"  Master returned no solution (status={status}). Stopping.")
            break

        # --- Lower bound from master -----------------------------------------
        # ObjBound is the tightest LB from B&B; equals ObjVal when optimal.
        master_LB = m.ObjBound if hasattr(m, 'ObjBound') else m.ObjVal
        LB = max(LB, master_LB)

        # --- Extract master solution -----------------------------------------
        D_val    = {r: D[r].x    for r in R}
        H_val    = {r: H[r].x    for r in R}
        beta_val = {r: beta[r].x for r in R}

        # --- Compute exact φ = Σ D_r²/H_r ------------------------------------
        f_val = {}
        for r in R:
            f_val[r] = (D_val[r]**2 / H_val[r]) if H_val[r] > 1e-8 else 0.0

        phi_master = sum(beta_val.values())
        phi_actual = sum(f_val.values())

        # --- Compute actual (true) objective at this routing -----------------
        eta_val   = {(i,r): eta[i,r].x   for i in C for r in R}
        theta_val = {(i,r): theta[i,r].x for i in C for r in R}

        if obj == "ATAT_Min":
            routing_part = (sum(0.5*eta_val[i,r] + theta_val[i,r]
                                for i in C for r in R) / Cn)
        elif obj == "MTAT_Min":
            routing_part = max_TLT.x
        elif obj == "TRL_Min":
            routing_part = sum(L[r].x for r in R)
        else:  # MRL_Min
            routing_part = max_L.x

        W_lab_actual = ((K-1)/(2*K*mu*rho)
                        + phi_actual * coeff_beta
                        + SCV*rho/(2*mu*(1-rho))
                        + 1.0/mu)

        if obj in ("ATAT_Min", "MTAT_Min"):
            actual_obj = routing_part + W_lab_actual
        else:
            actual_obj = routing_part   # SOC irrelevant for TRL/MRL

        # --- Update best upper bound ------------------------------------------
        if actual_obj < UB:
            UB = actual_obj
            best_snap = {
                "solution":  {(i,j,r): 1 for (i,j,r) in x.keys()
                              if x[i,j,r].x > 0.5},
                "L":         {r: L[r].x    for r in R},
                "H":         {r: H_val[r]  for r in R},
                "T":         {(i,r): T[i,r].x for i in N for r in R},
                "y":         {(i,r): y[i,r].x for i in C for r in R},
                "z":         ({r: z[r].x   for r in R}
                              if mode == "Multiple" else {r: 1 for r in R}),
                "eta":       eta_val,
                "theta":     theta_val,
                "W_lab":     W_lab_actual,
            }

        # --- Convergence check -----------------------------------------------
        soc_viol = max(f_val[r] - beta_val[r] for r in R)
        gap = (UB - LB) / abs(UB) * 100 if UB not in (0.0, float('inf')) else float('inf')

        if verbose:
            print(f"  {k+1:4d}  {LB:10.4f}  {UB:10.4f}  {gap:7.3f}%  "
                  f"{phi_master:9.4f}  {phi_actual:9.4f}  "
                  f"{soc_viol:9.4f}  {n_cuts_total:5d}")

        iter_log.append({
            "iter": k+1, "LB": LB, "UB": UB, "gap_pct": gap,
            "phi_master": phi_master, "phi_actual": phi_actual,
            "soc_violation": soc_viol, "n_cuts_total": n_cuts_total,
        })

        if soc_viol <= tol and gap <= gap_tol * 100:
            if verbose:
                print(f"\n  ✓ Converged at iteration {k+1}  "
                      f"(SOC_viol={soc_viol:.2e}, gap={gap:.4f}%)")
            break

        # --- Generate OA cuts for violated routes ----------------------------
        n_new = 0
        for r in R:
            if H_val[r] < 1e-8:
                continue                    # empty route — no cut needed
            if f_val[r] - beta_val[r] <= tol:
                continue                    # already satisfied

            slope = 2.0 * D_val[r] / H_val[r]          # 2·D*/H*
            curv  = (D_val[r] / H_val[r]) ** 2          # (D*/H*)²
            # β[r] ≥ slope·D[r] − curv·H[r]
            m.addConstr(beta[r] >= slope * D[r] - curv * H[r])
            n_new += 1

        n_cuts_total += n_new

        if n_new == 0:
            if verbose:
                print(f"\n  No new cuts generated but gap not closed "
                      f"(gap={gap:.4f}%). Stopping.")
            break

    # =========================================================================
    # 3.  Build output (same format as gurobi_solver)
    # =========================================================================
    solution_time = time.time() - start_time

    if best_snap is None:
        raise RuntimeError(
            "L-shaped solver found no feasible solution.\n"
            "Try increasing time_limit or check instance parameters."
        )

    br    = best_snap
    W_lab = br["W_lab"]

    TATs  = {
        i: sum(0.5*br["eta"][i,r] + br["theta"][i,r] for r in R) + W_lab
        for i in C
    }
    TRL  = sum(br["L"].values())
    MRL  = max(br["L"].values())
    ATAT = sum(TATs.values()) / Cn
    MTAT = max(TATs.values())

    if verbose:
        print(f"\n{'='*65}")
        print(f"  FINAL   obj={obj}  mode={mode}")
        print(f"  UB (objective) : {UB:.4f}")
        print(f"  LB             : {LB:.4f}")
        print(f"  Gap            : {gap:.4f}%")
        print(f"  ATAT           : {ATAT:.4f}")
        print(f"  MTAT           : {MTAT:.4f}")
        print(f"  TRL            : {TRL:.4f}")
        print(f"  MRL            : {MRL:.4f}")
        print(f"  W_lab          : {W_lab:.4f}")
        print(f"  OA cuts added  : {n_cuts_total}")
        print(f"  Iterations     : {len(iter_log)}")
        print(f"  Time           : {solution_time:.2f}s")
        print(f"{'='*65}")

    diagnostics = {
        "iter_log":      iter_log,
        "n_iter":        len(iter_log),
        "n_cuts":        n_cuts_total,
        "LB":            LB,
        "UB":            UB,
        "gap_pct":       gap,
        "solution_time": solution_time,
    }

    return [
        br["solution"],                                         # 0  solution
        br["L"],                                                # 1  lengthes
        {(i, r): br["T"][i, r] for i in N for r in R},        # 2  start_times
        br["y"],                                                # 3  allocations
        br["z"],                                                # 4  num_vehicles
        br["H"],                                                # 5  time_between_visits
        TATs,                                                   # 6  TATs
        UB,                                                     # 7  objective_value
        solution_time,                                          # 8  solution_time
        TRL,                                                    # 9  TRL
        MRL,                                                    # 10 MRL
        ATAT,                                                   # 11 ATAT
        MTAT,                                                   # 12 MTAT
        diagnostics,                                            # 13 diagnostics
    ]
