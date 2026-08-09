"""Strict white-box execution gate for tau-bench tool agents.

The stock benchmark grades primarily from the final environment state. This
wrapper rejects malformed or out-of-contract actions before they reach the
environment and adds action-trace validation at termination.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from tau_bench.envs.base import Env, consistent_hash, to_hashable
from tau_bench.types import Action, EnvInfo, EnvResponse, EnvResetResponse, RESPOND_ACTION_NAME


# tau-bench task traces are inconsistent about read-only prerequisites.  For
# example, retail's policy requires identity authentication, but many training
# tasks list only the later database mutation.  These are the only tools whose
# execution is consequential and therefore must match the reference trace.
STATE_MUTATING_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_pending_order",
        "cancel_reservation",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)
AUTHENTICATION_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})


class GuardedEnv:
    """Fail-closed wrapper around a tau-bench environment.

    This is intentionally a white-box evaluator: it reads ``task.actions`` to
    build a reference execution contract. It is suitable for development and
    fault-interception tests, not as a black-box production policy.
    """

    def __init__(self, env: Env) -> None:
        self.env = env
        self.expected_actions: list[Action] = []
        self.expected_mutations: list[Action] = []
        self.expected_terminal_hash = ""
        self.executed_actions: list[Action] = []
        self.executed_mutations: list[Action] = []
        self.intercepted = False
        self.interception_reason: str | None = None
        self.events: list[dict[str, Any]] = []
        self.authenticated = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, task_index: int | None = None) -> EnvResetResponse:
        response = self.env.reset(task_index=task_index)
        self.expected_actions = [
            action for action in self.env.task.actions if action.name != RESPOND_ACTION_NAME
        ]
        self.expected_mutations = [
            action for action in self.expected_actions if action.name in STATE_MUTATING_TOOLS
        ]
        self.expected_terminal_hash = self._reference_terminal_hash()
        self.executed_actions = []
        self.executed_mutations = []
        self.intercepted = False
        self.interception_reason = None
        self.events = []
        self.authenticated = False
        return response

    def step(self, action: Action) -> EnvResponse:
        if self.intercepted:
            return self._blocked_response("execution already intercepted")

        reason = self._validate_action(action)
        if reason is not None:
            return self._blocked_response(reason)

        if action.name in STATE_MUTATING_TOOLS:
            if not self.authenticated:
                return self._blocked_response(
                    "state-changing tool call before successful user authentication"
                )
            if len(self.executed_mutations) >= len(self.expected_mutations):
                return self._blocked_response("unexpected state-changing tool call after task actions completed")
            next_expected = self.expected_mutations[len(self.executed_mutations)]
            if not self._actions_match(action, next_expected):
                return self._blocked_response(
                    "unexpected state-changing action before execution: "
                    f"expected {next_expected.model_dump()}, got {action.model_dump()}"
                )

        state_hash_before_step = self.env.get_data_hash()
        response = self.env.step(action)
        if action.name != RESPOND_ACTION_NAME:
            if (
                action.name in STATE_MUTATING_TOOLS
                and response.observation.startswith("Error:")
            ):
                return self._blocked_response(f"tool execution failed: {response.observation}")
            self.executed_actions.append(action)
            if (
                action.name in AUTHENTICATION_TOOLS
                and response.observation == self.env.task.user_id
            ):
                self.authenticated = True
            if action.name in STATE_MUTATING_TOOLS:
                self.executed_mutations.append(action)
            self.events.append(
                {
                    "action": action.model_dump(),
                    "state_hash_after": self.env.get_data_hash(),
                    "recoverable_error": response.observation
                    if response.observation.startswith("Error:")
                    else None,
                }
            )

        if response.done:
            reason = self._terminal_failure_reason(action, state_hash_before_step)
            if reason is not None:
                return self._blocked_response(reason)
        return response

    def guard_summary(self) -> dict[str, Any]:
        return {
            "intercepted": self.intercepted,
            "interception_reason": self.interception_reason,
            "expected_action_count": len(self.expected_actions),
            "executed_action_count": len(self.executed_actions),
            "expected_mutation_count": len(self.expected_mutations),
            "executed_mutation_count": len(self.executed_mutations),
            "authenticated": self.authenticated,
            "events": self.events,
        }

    def abort(self, reason: str) -> EnvResponse:
        """Fail a non-terminating trajectory without retaining partial writes."""
        self.env.data = self.env.data_load_func()
        return self._blocked_response(reason)

    def _reference_terminal_hash(self) -> str:
        reference_data = self.env.data_load_func()
        for action in self.expected_actions:
            if action.name not in self.env.tools_map:
                raise ValueError(f"task references unknown tool: {action.name}")
            self.env.tools_map[action.name].invoke(data=reference_data, **action.kwargs)
        return consistent_hash(to_hashable(reference_data))

    @staticmethod
    def _actions_match(actual: Action, expected: Action) -> bool:
        if actual == expected:
            return True
        # The return tool stores item ids in sorted order, so their ordering is
        # not semantically meaningful. Other item-change tools pair old and
        # new ids positionally and must retain exact list ordering.
        if (
            actual.name != "return_delivered_order_items"
            or expected.name != "return_delivered_order_items"
        ):
            return False
        actual_kwargs = dict(actual.kwargs)
        expected_kwargs = dict(expected.kwargs)
        return (
            actual_kwargs.pop("item_ids", None) is not None
            and expected_kwargs.pop("item_ids", None) is not None
            and actual_kwargs == expected_kwargs
            and Counter(actual.kwargs["item_ids"]) == Counter(expected.kwargs["item_ids"])
        )

    def _validate_action(self, action: Action) -> str | None:
        if action.name == RESPOND_ACTION_NAME:
            content = action.kwargs.get("content")
            if not isinstance(content, str):
                return "respond.content must be a string"
            return None
        tool = self.env.tools_map.get(action.name)
        if tool is None:
            return f"tool not allowed: {action.name}"
        schema = tool.get_info()["function"]["parameters"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        supplied = set(action.kwargs)
        missing = required - supplied
        unknown = supplied - set(properties)
        if missing:
            return f"missing required arguments: {sorted(missing)}"
        if unknown:
            return f"unknown arguments: {sorted(unknown)}"
        for name, value in action.kwargs.items():
            expected_type = properties[name].get("type")
            if expected_type == "string" and not isinstance(value, str):
                return f"argument {name} must be a string"
            if expected_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                return f"argument {name} must be an integer"
            if expected_type == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                return f"argument {name} must be a number"
            if expected_type == "boolean" and not isinstance(value, bool):
                return f"argument {name} must be a boolean"
            if expected_type == "array" and not isinstance(value, list):
                return f"argument {name} must be an array"
        return None

    def _terminal_failure_reason(self, action: Action, state_hash_before_step: str) -> str | None:
        if action.name != RESPOND_ACTION_NAME:
            return "task terminated without a final response"
        if self.executed_mutations != self.expected_mutations:
            return "terminal state-changing action trace does not satisfy the task contract"
        if state_hash_before_step != self.expected_terminal_hash:
            return "terminal state hash does not satisfy the task contract"
        content = action.kwargs["content"].lower().replace(",", "")
        missing_outputs = [
            output for output in self.env.task.outputs if output.lower() not in content
        ]
        if missing_outputs:
            return f"final response is missing required outputs: {missing_outputs}"
        return None

    def _blocked_response(self, reason: str) -> EnvResponse:
        if not self.intercepted:
            self.intercepted = True
            self.interception_reason = reason
            self.events.append({"intercepted": True, "reason": reason})
            # A guarded task is transactional: no failed trajectory may leave
            # a prior state-changing action committed in the mock environment.
            self.env.data = self.env.data_load_func()
        return EnvResponse(
            observation=f"Execution intercepted: {self.interception_reason}",
            reward=0.0,
            done=True,
            info=EnvInfo(task=self.env.task, source="failure_interceptor"),
        )
