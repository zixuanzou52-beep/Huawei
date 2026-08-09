"""Repair role for bounded home-control planning."""

from multi_agent.planner import PlannerAgent


class RepairAgent(PlannerAgent):
    """LLM role that replaces a failed plan with a corrected bounded plan."""

    role_name = "repair"
