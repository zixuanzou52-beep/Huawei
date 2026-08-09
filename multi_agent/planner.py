"""LLM-backed planning primitives for the home-control workflow."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from litellm import completion

from tau_bench.types import Action, RESPOND_ACTION_NAME


YUNWU_API_KEY_ENV = "YUNWU_API_KEY"
YUNWU_API_BASE_ENV = "YUNWU_API_BASE"
DEFAULT_YUNWU_API_BASE = "https://yunwu.ai/v1"
MODEL_REQUEST_TIMEOUT_SECONDS = 60
MODEL_REQUEST_MAX_ATTEMPTS = 3
MODEL_RETRY_DELAY_SECONDS = 1

PLANNING_RULES = (
    "Return JSON only.",
    "Return only tool actions; do not include a respond action.",
    "Use only available_tools and observed device identifiers.",
    "Every action must advance the user's requested final device state.",
    "For a request to control equipment, return at least one required write action; never return an empty plan.",
    "A state query is only a prerequisite for a control request, not its completion; include the required write action in the plan.",
    "Treat a question about whether to perform a stated device action as a request to perform that action after any required confirmation.",
    "Use create_alert only when the user explicitly requests a persistent household alert.",
    "Never use create_alert to explain a plan, request confirmation, or acknowledge a request.",
)


@dataclass
class Plan:
    actions: list[Action]
    raw_content: str


class PlanError(ValueError):
    """Raised when a model response cannot be used as a tool plan."""


class UsageCost(float):
    """Response cost with provider-reported token usage attached."""

    def __new__(cls, value: float, token_usage: dict[str, int]):
        instance = float.__new__(cls, value)
        instance.token_usage = token_usage
        return instance


class PlannerAgent:
    """LLM role that turns a request and public state into a tool-only plan."""

    role_name = "planner"

    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.seed = seed

    def create_plan(
        self,
        instruction: str,
        state: dict[str, Any],
        tools: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[Plan, dict[str, Any], UsageCost]:
        prompt = {
            "role": self.role_name,
            "instruction": instruction,
            "current_state": state,
            "available_tools": tools,
            "context": context or {},
            "output_schema": {
                "actions": [
                    {"name": "tool_name", "arguments": {"tool_argument": "value"}}
                ]
            },
            "rules": list(PLANNING_RULES),
        }
        response = self._complete_with_retry(prompt)
        message = response.choices[0].message.model_dump()
        plan = self.parse_plan(message.get("content") or "")
        usage = getattr(response, "usage", None) or response.get("usage")
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if usage is not None:
            for key in token_usage:
                value = getattr(usage, key, None)
                if value is None and isinstance(usage, dict):
                    value = usage.get(key)
                if isinstance(value, (int, float)):
                    token_usage[key] = int(value)
        return (
            plan,
            message,
            UsageCost(response._hidden_params.get("response_cost") or 0.0, token_usage),
        )

    def _complete_with_retry(self, prompt: dict[str, Any]) -> Any:
        """Retry provider failures that cannot be corrected by replanning."""
        for attempt in range(MODEL_REQUEST_MAX_ATTEMPTS):
            try:
                return completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are the planning role in a bounded home-control workflow.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(prompt, ensure_ascii=False),
                        },
                    ],
                    temperature=self.temperature,
                    seed=self.seed,
                    response_format={"type": "json_object"},
                    timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
                    **self._provider_connection_options(),
                )
            except Exception as error:
                if (
                    attempt + 1 == MODEL_REQUEST_MAX_ATTEMPTS
                    or not self._is_transient_model_error(error)
                ):
                    raise
                time.sleep(MODEL_RETRY_DELAY_SECONDS * (attempt + 1))
        raise RuntimeError("model completion retry loop ended unexpectedly")

    @staticmethod
    def _is_transient_model_error(error: Exception) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "timeout",
                "timed out",
                "connection error",
                "connection reset",
                "temporarily unavailable",
                "rate limit",
                "too many requests",
                "service unavailable",
                "status code 429",
                "status code 500",
                "status code 502",
                "status code 503",
                "status code 504",
            )
        )

    def _provider_connection_options(self) -> dict[str, str]:
        """Pass Yunwu credentials to its OpenAI-compatible client."""
        if self.provider != "openai":
            return {}
        api_key = os.environ.get(YUNWU_API_KEY_ENV)
        if not api_key:
            return {}
        return {
            "api_key": api_key,
            "api_base": os.environ.get(YUNWU_API_BASE_ENV, DEFAULT_YUNWU_API_BASE),
        }

    @staticmethod
    def parse_plan(content: str) -> Plan:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise PlanError(f"planner returned invalid JSON: {error.msg}") from error
        actions = parsed.get("actions")
        if not isinstance(actions, list):
            raise PlanError("planner output must contain an actions list")

        parsed_actions: list[Action] = []
        for item in actions:
            if not isinstance(item, dict):
                raise PlanError("each planned action must be an object")
            name = item.get("name")
            arguments = item.get("arguments", item.get("kwargs"))
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise PlanError("each action requires string name and object arguments")
            if name == RESPOND_ACTION_NAME:
                raise PlanError("planner must not emit a respond action")
            parsed_actions.append(Action(name=name, kwargs=arguments))
        return Plan(actions=parsed_actions, raw_content=content)
