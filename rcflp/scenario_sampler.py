"""
rcflp.scenario_sampler
----------------------
Generates random disruption scenarios from the polyhedral uncertainty set Ξ.

A scenario is a dict {(j, h): float} where exactly one h per facility is 1.0
and the budget constraint

    sum_{j,h}  (h / (Hn - 1)) * epsilon[j, h]  <=  uncertainty_budget

holds.  The sampler draws each facility's disruption level sequentially in a
random order, clipping to the remaining budget at each step.  This produces a
non-uniform but valid distribution that covers the full feasible region.
"""

import numpy as np


def sample_scenarios(
    inst: dict,
    uncertainty_budget: float,
    Hn: int,
    n_samples: int = 200,
    seed: int = None,
) -> list:
    """
    Sample random scenarios from the uncertainty set Ξ.

    Parameters
    ----------
    inst               : instance dict from instancemaker()
    uncertainty_budget : Γ — total disruption budget
    Hn                 : number of disruption levels  (H = {0, …, Hn-1})
    n_samples          : number of scenarios to generate
    seed               : optional random seed for reproducibility

    Returns
    -------
    list of n_samples dicts, each mapping (j, h) -> {0.0, 1.0}
    """
    rng = np.random.default_rng(seed)
    J   = inst["J"]
    H   = list(range(Hn))

    scenarios = []
    for _ in range(n_samples):
        remaining = float(uncertainty_budget)
        h_chosen  = {}

        for j in rng.permutation(J).tolist():
            # Maximum disruption level that still fits in the remaining budget
            max_h = min(Hn - 1, int(remaining * (Hn - 1)))
            h = int(rng.integers(0, max_h + 1))   # uniform in {0, …, max_h}
            h_chosen[j] = h
            remaining  -= h / (Hn - 1)

        eps = {(j, h): 1.0 if h_chosen[j] == h else 0.0
               for j in J for h in H}
        scenarios.append(eps)

    return scenarios
