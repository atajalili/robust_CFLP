"""
rcflp.bdcp
----------
Benders Decomposition Cutting Plane (BDCP) algorithm.

At each iteration:
  1. Solve the subproblem (separation) to find the worst-case disruption ε*
     and the dual solution (α, β, t2, θ̃, ũ2, γ̃).
     Update the upper bound.
  2. Add a single linear Benders optimality cut to the master problem.
  3. Solve the master problem and update the lower bound.
  4. Terminate when the absolute gap ≤ tol or time limit is reached.

The master problem grows by only one linear constraint per iteration,
making individual master solves cheap — but typically many more
iterations are needed compared to C&CG.

Key design note on the Benders cut
-----------------------------------
The cut has the form:

  η ≥ -Σ_i α_i
      - Σ_{j,r,h} capacity[j,r] * x[j,r] * (1 - h/(H-1)) * θ̃[j,h]
      - Σ_{j,r,h} capacity[j,r] * x[j,r] * (1 - h/(H-1)) * ũ2[j,h]
      - Σ_{j,r,h} capacity[j,r] * x[j,r] * (1 - h/(H-1)) * γ̃[j,h]
      + Σ_i t2_i - Σ_i β_i

where θ̃[j,h], ũ2[j,h], γ̃[j,h] are the linearised products
ε[j,h] * θ[j], ε[j,h] * u2[j], ε[j,h] * γ[j] from the subproblem.
For facilities not open in the current x, these take value 0 (not big-M).
"""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from rcflp.subproblem import solve_subproblem_dual


def solve_BDCP(
    inst: dict,
    uncertainty_budget: float,
    Hn: int,
    x_init: dict,
    tol: float = 0.01,
    big_M: float = 10_000,
    time_limit: float = 6 * 3600,
    master_time_limit: float = 2000,
    verbose: bool = False,
) -> dict:
    """
    Solve the robust CFLP with the Benders Decomposition Cutting Plane algorithm.

    Parameters
    ----------
    inst               : instance dict from instancemaker()
    uncertainty_budget : Γ
    Hn                 : number of disruption levels
    x_init             : {(j,r): float}  warm-start (e.g. nominal solution)
    tol                : relative optimality gap tolerance for termination (e.g. 0.01 = 1%)
    big_M              : big-M for the subproblem bilinear linearisation
    time_limit         : total wall-clock time limit (seconds)
    master_time_limit  : time limit for each individual master solve
    verbose            : print iteration log

    Returns
    -------
    dict with keys:
        x_jr        – {(j,r): float}  optimal first-stage solution
        LB          – float           final lower bound  (minimisation obj)
        UB          – float           final upper bound  (minimisation obj)
        profit_LB   – float           = -UB  (best upper bound on profit)
        n_iter      – int             number of BDCP iterations
        runtime     – float           total wall-clock time (seconds)
        converged   – bool            True if gap tolerance was met
        iter_log    – list of dicts   per-iteration details
    """
    I  = inst["I"]
    J  = inst["J"]
    R  = inst["R"]
    H  = list(range(Hn))
    demand     = inst["demand"]
    capacity   = inst["capacity"]
    fixed_cost = inst["fixed_cost"]

    start  = time.time()
    LB     = -np.inf
    UB     =  np.inf
    x0     = x_init
    n_iter = 0
    converged = False
    iter_log  = []

    # ------------------------------------------------------------------ #
    #  Initialise master problem                                           #
    # ------------------------------------------------------------------ #
    master = gp.Model("BDCP_master")
    master.Params.OutputFlag = 0
    master.Params.TimeLimit  = master_time_limit

    x   = master.addVars(J, R, vtype=GRB.BINARY,    name="x")
    # nue is unbounded below — the Benders cuts provide the effective lower bound
    nue = master.addVar(lb=-GRB.INFINITY,            name="nue")

    master.setObjective(
        gp.quicksum(fixed_cost[j, r] * x[j, r] for j in J for r in R) + nue,
        GRB.MINIMIZE,
    )
    for j in J:
        master.addConstr(gp.quicksum(x[j, r] for r in R) <= 1)

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    while True:
        elapsed = time.time() - start
        if elapsed > time_limit:
            break

        # ---- Step 1: subproblem — full dual solution needed -----------
        t_sub_start = time.time()
        (eps_bar, alpha_bar, beta_bar, t2_bar,
         theta_bar, u2_bar, gamma_bar, sub_obj) = solve_subproblem_dual(
            x0, inst, uncertainty_budget, Hn,
            big_M=big_M,
            time_limit=min(master_time_limit, time_limit - elapsed),
            return_duals=True,
        )
        t_sub = time.time() - t_sub_start

        fixed_now = sum(fixed_cost[j, r] * x0[j, r] for j in J for r in R)
        UB = min(UB, sub_obj + fixed_now)

        # ---- Step 2: Benders optimality cut --------------------------
        #
        # The cut is linear in x[j,r].
        # theta_bar[j,h], u2_bar[j,h], gamma_bar[j,h] are constants
        # (dual values from the subproblem); they are 0 for non-open j.
        master.addConstr(
            nue >= - gp.quicksum(alpha_bar[i] for i in inst["I"])
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
                   + gp.quicksum(t2_bar[i]   for i in inst["I"])
                   - gp.quicksum(beta_bar[i] for i in inst["I"]),
            name=f"benders_cut_{n_iter}",
        )

        # ---- Step 3: solve master ------------------------------------
        master.update()
        t_master_start = time.time()
        master.optimize()
        t_master = time.time() - t_master_start

        x0  = {(j, r): x[j, r].x for j in J for r in R}
        LB  = max(master.ObjVal, LB)
        n_iter += 1

        diff = 100 * (round(UB, 2) - round(LB, 2)) / (abs(UB) + 0.000001)

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
                f"| gap={diff:.2f}% | sub={t_sub:.1f}s | master={t_master:.1f}s"
            )

        if diff <= 1:
            converged = True
            break

    return {
        "x_jr":      x0,
        "LB":        LB,
        "UB":        UB,
        "profit_LB": -UB,
        "n_iter":    n_iter,
        "runtime":   time.time() - start,
        "converged": converged,
        "iter_log":  iter_log,
    }
