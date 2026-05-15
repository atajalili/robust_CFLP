"""
rcflp.instance
--------------
Builds problem instance data from the Daskin (2013) 49-node dataset.

All cost/demand parameters follow the conventions in the paper:
  - distance[i,j]  : transportation cost per unit demand  (= 20 * Euclidean)
  - demand[i]      : potential arrival rate at node i      (population / 100000)
  - capacity[j,r]  : processing rate at facility j, level r
  - fixed_cost[j,r]: fixed opening cost rate
  - value_max[i]   : maximum willingness-to-pay at node i  (set externally)
  - coeff1[i,j]    : demand[i] * (distance[i,j] - value_max[i])
  - coeff2[i]      : demand[i] * value_max[i]
"""

import numpy as np
import pandas as pd
from math import sqrt


def instancemaker(
    In: int,
    Jn: int,
    Rn: int,
    value_max_scale: float,
    congestion_cost_val: float,
    data_path: str = "dataset.xlsx",
) -> dict:
    """
    Build a problem instance.

    Parameters
    ----------
    In : number of customer nodes  (first In rows of dataset)
    Jn : number of candidate facilities (first Jn rows of dataset)
    Rn : number of capacity levels
    value_max_scale : multiplier on max distance to set willingness-to-pay v
    congestion_cost_val : congestion cost rate w (uniform across facilities)
    data_path : path to dataset.xlsx

    Returns
    -------
    dict with keys:
        I, J, R          – index lists
        distance         – {(i,j): float}
        demand           – {i: float}
        capacity         – {(j,r): float}
        fixed_cost       – {(j,r): float}
        value_max        – {i: float}
        congestion_cost  – {j: float}
        coeff1           – {(i,j): float}   linear obj coefficient for y
        coeff2           – {i: float}       linear obj coefficient for Q
    """
    I = list(range(In))
    J = list(range(Jn))
    R = list(range(Rn))

    df = pd.read_excel(data_path)

    lons = df["lon"].tolist()
    lats = df["lat"].tolist()

    def euc(i, j):
        return sqrt((lons[i] - lons[j]) ** 2 + (lats[i] - lats[j]) ** 2)

    distance = {(i, j): 20 * round(euc(i, j), 1) for i in I for j in J}
    demand   = {i: round(df["population"].iloc[i] / 100_000, 0) for i in I}

    cap_raw  = df["capacity"].tolist()
    cost_raw = df["cost"].tolist()

    if Rn == 1:
        capacity   = {(j, r): round(cap_raw[j],  1) for j in J for r in R}
        fixed_cost = {(j, r): round(cost_raw[j], 0) for j in J for r in R}
    else:
        capacity = {
            (j, r): round(np.linspace(cap_raw[j] * 0.25, cap_raw[j], Rn)[r], 1)
            for j in J for r in R
        }
        fixed_cost = {
            (j, r): round(np.linspace(cost_raw[j] * 0.25, cost_raw[j], Rn)[r], 0)
            for j in J for r in R
        }

    d_max         = max(distance.values())
    value_max     = {i: value_max_scale * d_max for i in I}
    cong_cost     = {j: congestion_cost_val for j in J}

    coeff1 = {(i, j): demand[i] * (distance[i, j] - value_max[i]) for i in I for j in J}
    coeff2 = {i: demand[i] * value_max[i] for i in I}

    return {
        "I": I,
        "J": J,
        "R": R,
        "distance":        distance,
        "demand":          demand,
        "capacity":        capacity,
        "fixed_cost":      fixed_cost,
        "value_max":       value_max,
        "congestion_cost": cong_cost,
        "coeff1":          coeff1,
        "coeff2":          coeff2,
    }
