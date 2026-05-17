"""
rcflp — Robust Congested Facility Location Problem
====================================================
Solvers and utilities for the two-stage robust CFLP with
price-sensitive demand and M/M/1 congestion (Jalili Marand et al.).

Public API
----------
from rcflp.instance   import instancemaker
from rcflp.nominal    import solve_nominal
from rcflp.subproblem import solve_subproblem_dual
from rcflp.ccg        import solve_CCG
from rcflp.bdcp       import solve_BDCP
from rcflp.warmstart  import solve_robust_warmstart
from rcflp.evaluate   import (evaluate_second_stage, evaluate_fixed_price,
                               compute_prices, compute_cost_breakdown,
                               compute_epsilon_scalar, sample_disruptions,
                               worst_case_disruption, no_disruption_scenario)
"""

from rcflp.instance   import instancemaker
from rcflp.nominal    import solve_nominal
from rcflp.subproblem import solve_subproblem_dual
from rcflp.ccg        import solve_CCG
from rcflp.bdcp       import solve_BDCP
from rcflp.warmstart  import solve_robust_warmstart
from rcflp.evaluate   import (
    evaluate_second_stage,
    evaluate_fixed_price,
    compute_prices,
    compute_cost_breakdown,
    compute_epsilon_scalar,
    sample_disruptions,
    worst_case_disruption,
    no_disruption_scenario,
)

__all__ = [
    "instancemaker",
    "solve_nominal",
    "solve_subproblem_dual",
    "solve_CCG",
    "solve_BDCP",
    "solve_robust_warmstart",
    "evaluate_second_stage",
    "evaluate_fixed_price",
    "compute_prices",
    "compute_cost_breakdown",
    "compute_epsilon_scalar",
    "sample_disruptions",
    "worst_case_disruption",
    "no_disruption_scenario",
]
