"""
rcflp.evaluation
----------------
Post-hoc risk evaluation of a first-stage solution x_jr over a sample of
disruption scenarios.

Workflow
--------
1. Call ``evaluate_recourse`` for each scenario to obtain the per-scenario
   profit (fixed costs + optimal recourse given the disruption).
2. Call ``compute_risk_metrics`` on the resulting profit vector.

Risk metrics returned
---------------------
mean_profit   : sample mean of per-scenario profits
min_profit    : worst realised profit across all scenarios
pct5_profit   : 5th-percentile profit  (VaR at 95 % confidence)
cvar5         : CVaR at 5 % — mean of the worst 5 % of outcomes
cvar10        : CVaR at 10 % — mean of the worst 10 % of outcomes
prob_loss     : fraction of scenarios with negative profit
mean_regret   : mean regret vs. best-case scenario for this x
max_regret    : worst-case regret vs. best-case scenario for this x
"""

import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ------------------------------------------------------------------ #
# Recourse solver                                                      #
# ------------------------------------------------------------------ #

def evaluate_recourse(
    x_jr: dict,
    inst: dict,
    eps_bar: dict,
    Hn: int,
) -> float:
    """
    Solve the second-stage recourse problem for a fixed x_jr and scenario.

    The recourse model exactly mirrors the scenario block added by the C&CG
    master (_add_block), with x fixed to x_jr values.  The rotated-cone SOCP
    constraints model:
      * price-sensitive revenue   : (sum_r y[i,j,r])^2  <=  Q[i]
      * M/M/1 congestion waiting  : lambda^2  <=  (eps*mu*C - lambda)(eps*mu*x - lambda)

    Parameters
    ----------
    x_jr    : {(j,r): float}  first-stage binary solution (values ≈ 0 or 1)
    inst    : instance dict from instancemaker()
    eps_bar : {(j,h): float}  disruption scenario (one h active per facility)
    Hn      : number of disruption levels

    Returns
    -------
    float : total profit = -(fixed_cost + optimal recourse cost)
            Returns -inf if the recourse is infeasible (should not occur for
            a valid first-stage solution and valid scenario).
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

    # Fraction of capacity remaining after disruption
    eps_scalar = {
        j: sum((1.0 - h / (Hn - 1)) * eps_bar[j, h] for h in H)
        for j in J
    }

    # Fixed first-stage cost (scenario-independent)
    fixed = sum(fixed_cost[j, r] * x_jr[j, r] for j in J for r in R)

    # Facilities that can still serve customers
    J_active = [
        j for j in J
        if eps_scalar[j] > 1e-9 and any(x_jr[j, r] > 0.5 for r in R)
    ]

    m = gp.Model("recourse")
    m.Params.OutputFlag = 0

    y   = m.addVars(I, J_active, R, lb=0.0, ub=1.0, name="y")
    Q   = m.addVars(I,           lb=0.0, ub=1.0,    name="Q")
    C   = m.addVars(J_active, R, lb=0.0,             name="C")
    VV1 = m.addVars(I,           lb=0.0,             name="VV1")
    V1  = m.addVars(J_active, R, lb=0.0,             name="V1")
    V2  = m.addVars(J_active, R, lb=0.0,             name="V2")
    V3  = m.addVars(J_active, R, lb=0.0,             name="V3")

    # Objective: minimise recourse cost (fixed cost is a constant offset)
    m.setObjective(
        gp.quicksum(coeff1[i, j] * y[i, j, r]
                    for i in I for j in J_active for r in R)
        + gp.quicksum(coeff2[i] * Q[i] for i in I)
        + gp.quicksum(congestion_cost[j] * C[j, r]
                      for j in J_active for r in R),
        GRB.MINIMIZE,
    )

    # --- customer assignment constraints ---
    for i in I:
        m.addConstr(
            VV1[i] == gp.quicksum(y[i, j, r] for j in J_active for r in R)
        )
        # Revenue SOCP: (sum_r y[i,j,r])^2 <= Q[i]
        m.addQConstr(VV1[i] ** 2 <= Q[i])
        m.addConstr(
            gp.quicksum(y[i, j, r] for j in J_active for r in R) <= 1
        )
        for j in J_active:
            m.addConstr(gp.quicksum(y[i, j, r] for r in R) <= 1)
            for r in R:
                m.addConstr(y[i, j, r] <= x_jr[j, r])

    # --- facility-level constraints ---
    for j in J_active:
        eff_cap_j = eps_scalar[j] * sum(
            capacity[j, r] * x_jr[j, r] for r in R
        )
        m.addConstr(
            gp.quicksum(demand[i] * y[i, j, r] for i in I for r in R)
            <= eff_cap_j
        )
        for r in R:
            x_val   = x_jr[j, r]           # fixed binary value
            eff_mu  = eps_scalar[j] * capacity[j, r] * x_val
            lam     = gp.quicksum(demand[i] * y[i, j, r] for i in I)

            m.addConstr(V1[j, r] == lam)
            m.addConstr(V2[j, r] == eff_mu * C[j, r] - lam)
            m.addConstr(V3[j, r] == eff_mu - lam)

            # Queue stability: effective capacity >= load
            m.addConstr(eff_mu >= lam)

            # Congestion SOCP: lambda^2 <= (eff_mu*C - lambda)(eff_mu - lambda)
            if congestion_cost[j] != 0 and eff_mu > 1e-9:
                m.addQConstr(V1[j, r] ** 2 <= V2[j, r] * V3[j, r])

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return -(fixed + m.ObjVal)

    return -math.inf


# ------------------------------------------------------------------ #
# Risk metrics                                                         #
# ------------------------------------------------------------------ #

def compute_risk_metrics(profits: list) -> dict:
    """
    Compute risk metrics from a list of per-scenario profits.

    Parameters
    ----------
    profits : list or array of floats — one value per scenario

    Returns
    -------
    dict with keys:
        mean_profit, min_profit, pct5_profit,
        cvar5, cvar10, prob_loss, mean_regret, max_regret
    """
    p = np.array([v for v in profits if not math.isinf(v)])
    if len(p) == 0:
        raise ValueError("No feasible scenario profits to evaluate.")

    n   = len(p)
    k5  = max(1, math.ceil(0.05 * n))
    k10 = max(1, math.ceil(0.10 * n))

    sorted_p    = np.sort(p)           # ascending — worst first at left
    best_profit = float(sorted_p[-1])

    return {
        "mean_profit":  float(np.mean(p)),
        "min_profit":   float(sorted_p[0]),
        "pct5_profit":  float(np.percentile(p, 5)),
        "cvar5":        float(np.mean(sorted_p[:k5])),
        "cvar10":       float(np.mean(sorted_p[:k10])),
        "prob_loss":    float(np.mean(p < 0)),
        "mean_regret":  float(np.mean(best_profit - p)),
        "max_regret":   float(np.max(best_profit - p)),
    }


# ------------------------------------------------------------------ #
# Convenience wrapper                                                  #
# ------------------------------------------------------------------ #

def evaluate_solution(
    x_jr: dict,
    inst: dict,
    scenarios: list,
    Hn: int,
) -> dict:
    """
    Evaluate a first-stage solution across a list of scenarios and return
    both the per-scenario profits and the aggregated risk metrics.

    Parameters
    ----------
    x_jr      : {(j,r): float}  first-stage solution
    inst      : instance dict from instancemaker()
    scenarios : list of {(j,h): float} dicts from sample_scenarios()
    Hn        : number of disruption levels

    Returns
    -------
    dict with keys:
        profits      : list of per-scenario profits (length = len(scenarios))
        risk_metrics : dict from compute_risk_metrics()
    """
    profits = [
        evaluate_recourse(x_jr, inst, eps, Hn)
        for eps in scenarios
    ]
    return {
        "profits":      profits,
        "risk_metrics": compute_risk_metrics(profits),
    }
