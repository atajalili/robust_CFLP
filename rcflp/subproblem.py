"""
rcflp.subproblem
----------------
Solves the recourse problem (RP) — the separation oracle shared by both
the C&CG and BDCP algorithms.

The RP maximises simultaneously over the worst-case disruption scenario
ε ∈ Ξ and over the dual variables of the inner SOCP recourse problem
(model 11-12 in the paper).

Bilinear products  u2[j] * ε[j,h]  etc. are linearised with auxiliary
variables and big-M constraints (12c–12h in the paper).

`return_duals=False`  →  used by C&CG  (only needs ε* and obj value)
`return_duals=True`   →  used by BDCP  (needs full dual solution for the cut)
"""

import gurobipy as gp
from gurobipy import GRB


def solve_subproblem_dual(
    x_jr: dict,
    inst: dict,
    uncertainty_budget: float,
    Hn: int,
    big_M: float = 10_000,
    time_limit: float = 3600,
    return_duals: bool = False,
    n_scenarios: int = 1,
    exclude_scenario: dict = None,
) -> tuple:
    """
    Solve the recourse / separation problem.

    Parameters
    ----------
    x_jr               : {(j,r): float}  current first-stage solution
    inst               : instance dict from instancemaker()
    uncertainty_budget : Γ parameter
    Hn                 : number of disruption levels  (H = {0, …, Hn-1})
    big_M              : big-M constant for bilinear linearisation
    time_limit         : Gurobi time limit in seconds
    return_duals       : whether to return full dual solution (for BDCP)
    n_scenarios        : ignored (kept for API compatibility, always 1 internally)
    exclude_scenario   : optional {(j,h): float} epsilon map to exclude via an
                         integer cut.  When provided, the subproblem is forced to
                         find the best scenario *different* from the one given.
                         Only used when return_duals=False.

    Returns (return_duals=False)
    ----------------------------
    (epsilon_maps, obj_val)
        epsilon_maps : list of {(j,h): float}  — length n_scenarios (or fewer if
                       solution pool has fewer distinct solutions)
        obj_val      : float  — best (worst-case) subproblem objective value

    Returns (return_duals=True)
    ---------------------------
    (epsilon_map, alpha, beta, t2, theta_map, u2_map, gamma_map, obj_val)
        alpha, beta, t2 : {i: float}     dual variables for customer constraints
        theta_map       : {(j,h): float} linearised  ε * θ  values
        u2_map          : {(j,h): float} linearised  ε * u2 values
        gamma_map       : {(j,h): float} linearised  ε * γ  values
    """
    I  = inst["I"]
    J  = inst["J"]
    R  = inst["R"]
    H  = list(range(Hn))
    demand          = inst["demand"]
    capacity        = inst["capacity"]
    coeff1          = inst["coeff1"]
    coeff2          = inst["coeff2"]
    congestion_cost = inst["congestion_cost"]

    # Derived first-stage quantities
    x0  = {j: sum(x_jr[j, r] for r in R) for j in J}
    mu0 = {j: sum(x_jr[j, r] * capacity[j, r] for r in R) for j in J}
    fJ  = [j for j in J if x0[j] > 0.5]   # open facilities only

    d = gp.Model("subproblem_dual")
    d.Params.OutputFlag = 0
    d.Params.TimeLimit  = time_limit

    # --- dual variables of the inner SOCP ---
    alpha = d.addVars(I,   vtype=GRB.CONTINUOUS, name="alpha")
    u1    = d.addVars(fJ,  vtype=GRB.CONTINUOUS, name="u1")
    u2    = d.addVars(fJ,  vtype=GRB.CONTINUOUS, name="u2")
    gamma = d.addVars(fJ,  vtype=GRB.CONTINUOUS, name="gamma")
    t1    = d.addVars(I,   vtype=GRB.CONTINUOUS, name="t1")
    t2    = d.addVars(I,   vtype=GRB.CONTINUOUS, name="t2")
    beta  = d.addVars(I,   vtype=GRB.CONTINUOUS, name="beta")
    theta = d.addVars(fJ,  vtype=GRB.CONTINUOUS, name="theta")
    eta   = d.addVars(fJ,  vtype=GRB.CONTINUOUS, name="eta")

    # --- disruption scenario variables ---
    epsilon   = d.addVars(fJ, H, vtype=GRB.BINARY,     name="epsilon")

    # --- linearisation auxiliaries:  aux[j,h] = base[j] * epsilon[j,h] ---
    eps_gamma = d.addVars(fJ, H, vtype=GRB.CONTINUOUS, lb=0, name="eps_gamma")
    eps_theta = d.addVars(fJ, H, vtype=GRB.CONTINUOUS, lb=0, name="eps_theta")
    eps_eta   = d.addVars(fJ, H, vtype=GRB.CONTINUOUS, lb=0, name="eps_eta")
    eps_u2    = d.addVars(fJ, H, vtype=GRB.CONTINUOUS, lb=0, name="eps_u2")

    # --- objective (11a in paper) ---
    d.setObjective(
        - gp.quicksum(alpha[i] for i in I)
        - gp.quicksum(
            mu0[j] * gp.quicksum((1 - h / (Hn - 1)) * eps_theta[j, h] for h in H)
            for j in fJ)
        - gp.quicksum(
            mu0[j] * gp.quicksum((1 - h / (Hn - 1)) * eps_u2[j, h] for h in H)
            for j in fJ)
        - gp.quicksum(
            mu0[j] * gp.quicksum((1 - h / (Hn - 1)) * eps_gamma[j, h] for h in H)
            for j in fJ)
        + gp.quicksum(t2[i]   for i in I)
        - gp.quicksum(beta[i] for i in I),
        GRB.MAXIMIZE,
    )

    # --- budget constraint (8) ---
    d.addConstr(
        gp.quicksum((h / (Hn - 1)) * epsilon[j, h] for j in fJ for h in H)
        <= uncertainty_budget,
        name="budget"
    )

    # --- dual feasibility constraints (11b–11g) ---
    for i in I:
        # (11c)
        d.addConstr(coeff2[i] - t2[i] - beta[i] >= 0)
        # (11f)
        d.addQConstr(t1[i] ** 2 + t2[i] ** 2 <= beta[i] ** 2)
        for j in fJ:
            # (11b)
            d.addConstr(
                coeff1[i, j] + alpha[i]
                + demand[i] * eta[j] + demand[i] * theta[j]
                + 2 * t1[i]
                + 2 * demand[i] * u1[j]
                + 2 * demand[i] * gamma[j] >= 0
            )

    for j in fJ:
        # (11d) — written with eps_* since ε_j appears in product with μ_j
        d.addConstr(
            congestion_cost[j]
            - mu0[j] * gp.quicksum((1 - h / (Hn - 1)) * eps_eta[j, h]   for h in H)
            + mu0[j] * gp.quicksum((1 - h / (Hn - 1)) * eps_u2[j, h]    for h in H)
            - mu0[j] * gp.quicksum((1 - h / (Hn - 1)) * eps_gamma[j, h] for h in H)
            >= 0
        )
        # (11e)
        d.addQConstr(u1[j] ** 2 + u2[j] ** 2 <= gamma[j] ** 2)

        # exactly one scenario per facility (from Ξ definition)
        d.addConstr(gp.quicksum(epsilon[j, h] for h in H) == 1)

        # (12c–12h) big-M linearisation for each bilinear product
        for h in H:
            for aux, base in [
                (eps_gamma, gamma),
                (eps_theta, theta),
                (eps_eta,   eta),
                (eps_u2,    u2),
            ]:
                d.addConstr(aux[j, h] <= base[j])
                d.addConstr(aux[j, h] <= big_M * epsilon[j, h])
                d.addConstr(aux[j, h] >= base[j] - big_M * (1 - epsilon[j, h]))

    # --- integer exclusion cut: forbid the given scenario (for second call) ---
    if exclude_scenario is not None and not return_duals:
        # Identify which (j,h) pairs are active (epsilon=1) in the excluded scenario,
        # but only for open facilities (the only ones that appear in the model).
        active = [(j, h) for j in fJ for h in H
                  if exclude_scenario.get((j, h), 0) > 0.5]
        if active:
            # Standard integer cut: sum of active vars <= |active| - 1
            # This forbids the exact same combination of disrupted (j,h) pairs.
            d.addConstr(
                gp.quicksum(epsilon[j, h] for (j, h) in active) <= len(active) - 1,
                name="exclude_scenario"
            )

    d.optimize()

    obj_val = d.ObjVal

    if not return_duals:
        # Return the single best scenario found
        e_map = {}
        for j in J:
            for h in H:
                if j in fJ:
                    e_map[j, h] = round(epsilon[j, h].x)
                else:
                    e_map[j, h] = 1.0 if h == 0 else 0.0
        return [e_map], obj_val

    # --- build full epsilon map over ALL j for dual return path ---
    epsilon_map = {}
    for j in J:
        for h in H:
            if j in fJ:
                epsilon_map[j, h] = epsilon[j, h].x
            else:
                epsilon_map[j, h] = 1.0 if h == 0 else 0.0
    # For non-open facilities, use big_M (not 0).
    # This gives the Benders cut a large negative coefficient on x[j,r]
    # for facilities not currently open, which penalises the master for
    # opening them and forces it to seek genuinely better solutions.
    # This matches the original implementation exactly.
    def _map(var_jh, default):
        """Return {(j,h): val} over all J, default for non-open facilities."""
        out = {}
        for j in J:
            for h in H:
                out[j, h] = var_jh[j, h].x if j in fJ else default
        return out

    return (
        epsilon_map,
        {i: alpha[i].x for i in I},
        {i: beta[i].x  for i in I},
        {i: t2[i].x    for i in I},
        _map(eps_theta, big_M),
        _map(eps_u2,    big_M),
        _map(eps_gamma, big_M),
        obj_val,
    )
