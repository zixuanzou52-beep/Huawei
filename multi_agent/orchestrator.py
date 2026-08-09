"""Orchestrate the Planner, Repair, Executor, Monitor, and Verifier roles."""

from __future__ import annotations

import json
from typing import Any, Optional

from tau_bench.agents.base import Agent
from multi_agent.executor import Executor
from multi_agent.planner import (
    PLANNING_RULES,
    Plan,
    PlanError,
    PlannerAgent,
    UsageCost,
)
from multi_agent.repair import RepairAgent
from tau_bench.envs.base import Env
from tau_bench.types import Action, RESPOND_ACTION_NAME, SolveResult


__all__ = [
    "Plan",
    "PlanError",
    "PLANNING_RULES",
    "PlannerAgent",
    "PlannerRepairAgent",
    "RepairAgent",
    "UsageCost",
]


class PlannerRepairAgent(Agent):
    """Coordinate LLM planning/repair with deterministic execution and safety."""

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_repairs: int = 1,
        planner: Optional[PlannerAgent] = None,
        repairer: Optional[RepairAgent] = None,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.max_repairs = max(0, max_repairs)
        self.planner = planner or PlannerAgent(model, provider, temperature, seed)
        self.repairer = repairer or RepairAgent(model, provider, temperature, seed)
        self.executor = Executor(self._execution_tool_names())

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        reset = env.reset(task_index=task_index)
        planner_tools = self._planner_tools()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": reset.observation}
        ]
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_cost = 0.0
        info = reset.info.model_dump()
        executed: list[dict[str, Any]] = []
        failure: Optional[str] = None
        remaining_steps = max_num_steps
        instruction = reset.observation
        conversation_context: dict[str, Any] = {}

        for repair_round in range(self.max_repairs + 1):
            context = {
                "repair_round": repair_round,
                "previous_failure": failure,
                "executed_actions": executed,
                **conversation_context,
            }
            role = self.planner if repair_round == 0 else self.repairer
            try:
                plan, planner_message, cost = role.create_plan(
                    instruction=instruction,
                    state=self._state_snapshot(env),
                    tools=planner_tools,
                    context=context,
                )
            except Exception as error:
                failure = f"{role.role_name} failed: {error}"
                messages.append({"role": role.role_name, "content": failure})
                continue

            total_cost += cost
            for key, value in getattr(cost, "token_usage", {}).items():
                token_usage[key] += value
            messages.append(
                {
                    "role": role.role_name,
                    "content": plan.raw_content,
                    "raw": planner_message,
                }
            )

            plan_error = self._plan_error(plan)
            if plan_error:
                failure = plan_error
                messages.append({"role": "verifier", "content": plan_error})
                continue

            if self._confirmation_required_for(env, plan.actions):
                confirmation = self._request_confirmation(env, info, messages)
                info = confirmation[0]
                if confirmation[1]:
                    return self._finished_result(
                        confirmation[2], messages, info, total_cost, token_usage
                    )
                conversation_context["user_confirmation"] = confirmation[3]
                if self._confirmation_required_for(env, plan.actions):
                    return self._finish_failure(
                        env,
                        "user confirmation was not provided for the planned write actions",
                        info,
                        messages,
                        total_cost,
                        token_usage,
                    )

            failure = self.executor.execute(
                env=env,
                plan=plan,
                remaining_steps=remaining_steps,
                executed=executed,
                messages=messages,
            )
            remaining_steps -= len(plan.actions)
            if failure is None:
                response = env.step(
                    Action(name=RESPOND_ACTION_NAME, kwargs={"content": "已完成。"})
                )
                info = {**info, **response.info.model_dump()}
                messages.append(
                    {
                        "role": "verifier",
                        "content": "all planned actions completed without an execution error",
                    }
                )
                messages.append({"role": "executor", "content": response.observation})
                return self._finished_result(
                    response, messages, info, total_cost, token_usage
                )
            if getattr(env, "intercepted", False):
                return self._finish_intercepted(
                    env, info, messages, total_cost, token_usage
                )
            if remaining_steps <= 0:
                break

        return self._finish_failure(
            env,
            failure or "repair budget exhausted",
            info,
            messages,
            total_cost,
            token_usage,
        )

    @staticmethod
    def _plan_error(plan: Plan) -> Optional[str]:
        if not plan.actions:
            return "planner returned an empty plan"
        for action in plan.actions:
            if (
                action.name == "create_alert"
                and action.kwargs.get("alert_type") == "confirmation"
            ):
                return "create_alert cannot be used to request user confirmation"
        return None

    @staticmethod
    def _request_confirmation(
        env: Env, info: dict[str, Any], messages: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], bool, Any, str]:
        """Request a scripted confirmation without exposing policy metadata."""
        prompt = "这项操作涉及家庭安全或隐私。请确认是否现在执行？"
        response = env.step(
            Action(name=RESPOND_ACTION_NAME, kwargs={"content": prompt})
        )
        updated_info = {**info, **response.info.model_dump()}
        messages.extend(
            [
                {"role": "assistant", "content": prompt},
                {"role": "user", "content": response.observation},
            ]
        )
        return updated_info, response.done, response, response.observation

    @staticmethod
    def _finished_result(
        response: Any,
        messages: list[dict[str, Any]],
        info: dict[str, Any],
        total_cost: float,
        token_usage: dict[str, int],
    ) -> SolveResult:
        return SolveResult(
            reward=response.reward,
            messages=messages,
            info={**info, "token_usage": token_usage},
            total_cost=total_cost,
        )

    def _finish_intercepted(
        self,
        env: Env,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        total_cost: float,
        token_usage: dict[str, int],
    ) -> SolveResult:
        summary = env.guard_summary() if hasattr(env, "guard_summary") else {}
        messages.append(
            {"role": "monitor", "content": json.dumps(summary, ensure_ascii=False)}
        )
        return SolveResult(
            reward=0.0,
            messages=messages,
            info={**info, "failure_interceptor": summary, "token_usage": token_usage},
            total_cost=total_cost,
        )

    def _finish_failure(
        self,
        env: Env,
        reason: str,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        total_cost: float,
        token_usage: dict[str, int],
    ) -> SolveResult:
        messages.append({"role": "verifier", "content": reason})
        if hasattr(env, "abort"):
            response = env.abort(reason)
        else:
            response = env.step(
                Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={"content": f"任务未完成：{reason}"},
                )
            )
        info = {**info, **response.info.model_dump()}
        if hasattr(env, "guard_summary"):
            info["failure_interceptor"] = env.guard_summary()
        return self._finished_result(response, messages, info, total_cost, token_usage)

    def _planner_tools(self) -> list[dict[str, Any]]:
        """Expose the global registry without leaking task-specific gold tools."""
        return [
            tool
            for tool in self.tools_info
            if tool["function"]["name"] not in {"think", "transfer_to_human_agents"}
        ]

    def _execution_tool_names(self) -> set[str]:
        return {tool["function"]["name"] for tool in self._planner_tools()}

    @staticmethod
    def _confirmation_required_for(env: Env, actions: list[Action]) -> bool:
        """Ask the runtime Monitor for a narrow confirmation decision."""
        checker = getattr(env, "confirmation_required_for", None)
        return bool(checker(actions)) if callable(checker) else False

    @staticmethod
    def _state_snapshot(env: Env) -> dict[str, Any]:
        state = getattr(env, "data", {})
        return json.loads(json.dumps(state, ensure_ascii=False))
