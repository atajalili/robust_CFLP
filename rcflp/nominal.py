"""
rcflp.nominal
-------------
Solves the nominal (no-disruption) MISOCP — model (3) in the paper.

The SOCP reformulation of the congestion and revenue terms follows
Theorem 1: two sets of rotated second-order cone constraints.

Returns the optimal first-stage solution x[j,r] together with
summary statistics, which are used as the warm-start for both
the C&CG and BDCP algorithms.
"""

import gurobipy as gp
from gurobipy import GRB


def solve_nominal(inst: dict, mip_gap: float = 0.01, time_limit: float = 3600) -> dict:
    """
    Solve the nominal MISOCP.

    Parameters
    ----------
    inst       : instance dict from instancemaker()
    mip_gap    : Gurobi MIPGap tolerance
    time_limit : Gurobi time limit in seconds

    Returns
    -------
    dict with keys:
        x_jr      – {(j,r): float}  optimal binary location-capacity variables
        obj_val   – float            optimal objective value (minimisation form)
        profit    – float            = -obj_val
        runtime   – float            solver wall-clock time (seconds)
        status    – int              Gurobi status code
    """
    I  = inst["I"]
    J  = inst["J"]
    R  = inst["R"]
    demand         = inst["demand"]
    capacity       = inst["capacity"]
    fixed_cost     = inst["fixed_cost"]
    coeff1         = inst["coeff1"]
    coeff2         = inst["coeff2"]
    congestion_cost = inst["congestion_cost"]

    m = gp.Model("nominal")
    m.Params.OutputFlag = 0
    m.Params.MIPGap     = mip_gap
    m.Params.TimeLimit  = time_limit

    # --- decision variables ---
    xx  = m.addVars(J, R, vtype=GRB.BINARY,     name="x")
    yy  = m.addVars(I, J, R, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
    CC  = m.addVars(J, R, vtype=GRB.CONTINUOUS, lb=0, name="C")
    QQ  = m.addVars(I,    vtype=GRB.CONTINUOUS, lb=0, name="Q")

    # auxiliary variables for SOCP constraints
    V1  = m.addVars(J, R, vtype=GRB.CONTINUOUS, name="V1")
    V2  = m.addVars(J, R, vtype=GRB.CONTINUOUS, name="V2")
    V3  = m.addVars(J, R, vtype=GRB.CONTINUOUS, name="V3")
    VV1 = m.addVars(I,    vtype=GRB.CONTINUOUS, name="VV1")
    VV2 = m.addVars(I,    vtype=GRB.CONTINUOUS, name="VV2")
    VV3 = m.addVars(I,    vtype=GRB.CONTINUOUS, name="VV3")

    # --- objective ---
    m.setObjective(
        gp.quicksum(fixed_cost[j, r] * xx[j, r] for j in J for r in R)
        + gp.quicksum(coeff1[i, j] * yy[i, j, r] for i in I for j in J for r in R)
        + gp.quicksum(coeff2[i] * QQ[i] for i in I)
        + gp.quicksum(congestion_cost[j] * CC[j, r] for j in J for r in R),
        GRB.MINIMIZE,
    )

    # --- constraints ---
    for i in I:
        # Revenue SOCP: ||[2*sum_y, 1-Q]||_2 <= 1+Q
        # equivalently: VV1 = 2*sum_y, VV2 = 1-Q, VV3 = 1+Q
        m.addConstr(VV1[i] == 2 * gp.quicksum(yy[i, j, r] for j in J for r in R))
        m.addConstr(VV2[i] == 1 - QQ[i])
        m.addConstr(VV3[i] == 1 + QQ[i])
        m.addQConstr(VV1[i] ** 2 + VV2[i] ** 2 <= VV3[i] ** 2)

        m.addConstr(gp.quicksum(yy[i, j, r] for j in J for r in R) <= 1)
        for j in J:
            m.addConstr(gp.quicksum(yy[i, j, r] for r in R) <= 1)
            for r in R:
                m.addConstr(yy[i, j, r] <= xx[j, r])

    for j in J:
        m.addConstr(
            gp.quicksum(demand[i] * yy[i, j, r] for i in I for r in R)
            <= gp.quicksum(capacity[j, r] * xx[j, r] for r in R)
        )
        m.addConstr(gp.quicksum(xx[j, r] for r in R) <= 1)

        for r in R:
            # queue stability
            m.addConstr(
                capacity[j, r] * CC[j, r]
                - gp.quicksum(demand[i] * yy[i, j, r] for i in I) >= 0
            )
            m.addConstr(
                capacity[j, r] * xx[j, r]
                - gp.quicksum(demand[i] * yy[i, j, r] for i in I) >= 0
            )
            # Congestion SOCP: ||[2*lambda, mu*C - mu]||_2 <= mu*C + mu - 2*lambda
            lam = gp.quicksum(demand[i] * yy[i, j, r] for i in I)
            m.addConstr(V1[j, r] == 2 * lam)
            m.addConstr(V2[j, r] == -capacity[j, r] * xx[j, r] + capacity[j, r] * CC[j, r])
            m.addConstr(
                V3[j, r] == capacity[j, r] * xx[j, r]
                + capacity[j, r] * CC[j, r] - 2 * lam
            )
            m.addQConstr(V1[j, r] ** 2 + V2[j, r] ** 2 <= V3[j, r] ** 2)

    m.optimize()

    return {
        "x_jr":    {(j, r): xx[j, r].x for j in J for r in R},
        "obj_val": m.ObjVal,
        "profit":  -m.ObjVal,
        "runtime": m.Runtime,
        "status":  m.Status,
    }
