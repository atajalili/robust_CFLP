"""
rcflp — Robust Congested Facility Location Problem
====================================================
Solvers and utilities for the two-stage robust CFLP with
price-sensitive demand and M/M/1 congestion (Jalili Marand et al.).

Public API
----------
from rcflp.instance          import instancemaker
from rcflp.nominal           import solve_nominal
from rcflp.subproblem        import solve_subproblem_dual
from rcflp.ccg               import solve_CCG
from rcflp.bdcp              import solve_BDCP
from rcflp.warmstart         import solve_robust_warmstart
from rcflp.scenario_sampler  import sample_scenarios
from rcflp.evaluation        import evaluate_recourse, evaluate_solution, compute_risk_metrics
"""

from rcflp.instance          import instancemaker
from rcflp.nominal           import solve_nominal
from rcflp.subproblem        import solve_subproblem_dual
from rcflp.ccg               import solve_CCG
from rcflp.bdcp              import solve_BDCP
from rcflp.warmstart         import solve_robust_warmstart
from rcflp.scenario_sampler  import sample_scenarios
from rcflp.evaluation        import evaluate_recourse, evaluate_solution, compute_risk_metrics

__all__ = [
    "instancemaker",
    "solve_nominal",
    "solve_subproblem_dual",
    "solve_CCG",
    "solve_BDCP",
    "solve_robust_warmstart",
    "sample_scenarios",
    "evaluate_recourse",
    "evaluate_solution",
    "compute_risk_metrics",
]
