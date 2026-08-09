"""Deterministic plan dispatch for the home-control workflow."""

from __future__ import annotations

import json
from typing import Any, Optional

from multi_agent.planner import Plan
from tau_bench.envs.base import Env


class Executor:
    """Dispatch a validated plan in order and record each runtime outcome."""

    def __init__(self, executable_names: set[str]) -> None:
        self.executable_names = executable_names

    def execute(
        self,
        env: Env,
        plan: Plan,
        remaining_steps: int,
        executed: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> Optional[str]:
        if len(plan.actions) > remaining_steps:
            return "planner plan exceeds the remaining Agent Loop budget"

        for action in plan.actions:
            if action.name not in self.executable_names:
                return f"planner selected an unavailable tool: {action.name}"
            response = env.step(action)
            event = {
                "action": action.model_dump(),
                "observation": response.observation,
                "done": response.done,
            }
            executed.append(event)
            messages.append(
                {"role": "executor", "content": json.dumps(event, ensure_ascii=False)}
            )
            if response.done:
                return "execution was intercepted or terminated"
            if response.observation.startswith("Error:"):
                return response.observation

        return None
