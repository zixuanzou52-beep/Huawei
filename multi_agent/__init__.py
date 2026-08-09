"""Huawei home-control multi-agent implementation.

The package owns the Planner, Repair, Executor, Monitor, and Verifier roles.
tau-bench supplies only the generic benchmark runtime and lightweight adapters.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Executor",
    "Monitor",
    "Plan",
    "PlanError",
    "PlannerAgent",
    "PlannerRepairAgent",
    "RepairAgent",
    "evaluate_home_acceptance",
]

_EXPORTS = {
    "Executor": ("multi_agent.executor", "Executor"),
    "Monitor": ("multi_agent.monitor", "Monitor"),
    "Plan": ("multi_agent.planner", "Plan"),
    "PlanError": ("multi_agent.planner", "PlanError"),
    "PlannerAgent": ("multi_agent.planner", "PlannerAgent"),
    "PlannerRepairAgent": ("multi_agent.orchestrator", "PlannerRepairAgent"),
    "RepairAgent": ("multi_agent.repair", "RepairAgent"),
    "evaluate_home_acceptance": (
        "multi_agent.verifier",
        "evaluate_home_acceptance",
    ),
}


def __getattr__(name: str) -> Any:
    """Load a role only when its dependency graph is ready."""
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), symbol)
    globals()[name] = value
    return value
