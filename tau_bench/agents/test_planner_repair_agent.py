from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

from tau_bench.agents.planner_repair_agent import (
    PLANNING_RULES,
    Plan,
    PlannerAgent,
    PlannerRepairAgent,
)
from tau_bench.envs.home.env import MockHomeDomainEnv, evaluate_home_acceptance
from tau_bench.guards.home_execution_gate import Monitor
from tau_bench.types import Action


class ScriptedRole:
    def __init__(self, role_name: str, plans: list[Plan]) -> None:
        self.role_name = role_name
        self.plans = plans
        self.calls = 0
        self.last_tools: list[dict[str, Any]] = []
        self.last_kwargs: dict[str, Any] = {}

    def create_plan(self, **kwargs: Any):
        self.last_tools = kwargs["tools"]
        self.last_kwargs = kwargs
        plan = self.plans[self.calls]
        self.calls += 1
        return plan, {"role": self.role_name, "content": plan.raw_content}, 0.0


class PlannerRepairAgentTest(unittest.TestCase):
    def test_yunwu_credentials_use_dedicated_environment_variables(self) -> None:
        with patch.dict(
            os.environ,
            {"YUNWU_API_KEY": "test-key", "YUNWU_API_BASE": "https://example.test/v1"},
            clear=True,
        ):
            agent = PlannerAgent(model="test-model", provider="openai")
            self.assertEqual(
                agent._provider_connection_options(),
                {"api_key": "test-key", "api_base": "https://example.test/v1"},
            )

    def test_yunwu_default_base_only_applies_to_openai_provider(self) -> None:
        with patch.dict(os.environ, {"YUNWU_API_KEY": "test-key"}, clear=True):
            self.assertEqual(
                PlannerAgent(
                    model="test-model", provider="openai"
                )._provider_connection_options(),
                {"api_key": "test-key", "api_base": "https://yunwu.ai/v1"},
            )
            self.assertEqual(
                PlannerAgent(
                    model="test-model", provider="ollama"
                )._provider_connection_options(),
                {},
            )

    def test_planning_rules_require_control_plans_to_include_writes(self) -> None:
        rules = " ".join(PLANNING_RULES).lower()
        self.assertIn("never return an empty plan", rules)
        self.assertIn("not its completion", rules)

    def test_only_transient_model_errors_are_retryable(self) -> None:
        self.assertTrue(
            PlannerAgent._is_transient_model_error(RuntimeError("Request timed out"))
        )
        self.assertTrue(
            PlannerAgent._is_transient_model_error(RuntimeError("Connection error"))
        )
        self.assertTrue(
            PlannerAgent._is_transient_model_error(RuntimeError("status code 429"))
        )
        self.assertFalse(
            PlannerAgent._is_transient_model_error(ValueError("invalid JSON schema"))
        )

    def make_light_task(self):
        env = MockHomeDomainEnv(task_split="all")
        task_index = next(
            index
            for index, task in enumerate(env.tasks)
            if task.category == "routine"
            and task.expected_state["devices"]
            == {"living_room_air_purifier": {"power": True}}
            and not task.policy.get("confirmation_required")
        )
        task = env.tasks[task_index]
        action = Action(
            name="turn_on", kwargs={"device_id": "living_room_air_purifier"}
        )
        return env, task_index, action

    def make_agent(
        self, planner: ScriptedRole, repairer: ScriptedRole, max_repairs: int = 1
    ):
        return PlannerRepairAgent(
            tools_info=MockHomeDomainEnv(task_split="all").tools_info,
            wiki="",
            model="test-model",
            provider="openai",
            planner=planner,
            repairer=repairer,
            max_repairs=max_repairs,
        )

    def test_planner_plan_is_executed_then_verified(self) -> None:
        env, task_index, action = self.make_light_task()
        planner = ScriptedRole("planner", [Plan([action], '{"actions": ["valid"]}')])
        repairer = ScriptedRole("repair", [])
        guarded = Monitor(env)
        result = self.make_agent(planner, repairer).solve(
            guarded, task_index=task_index
        )

        self.assertEqual(result.reward, 0.0)
        self.assertEqual(evaluate_home_acceptance(guarded)["reward"], 1.0)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(repairer.calls, 0)
        self.assertFalse(guarded.intercepted)
        self.assertGreater(
            len(planner.last_tools), len(env.tasks[task_index].allowed_tools)
        )
        self.assertIn("verifier", [message["role"] for message in result.messages])

    def test_repair_agent_replaces_a_plan_rejected_by_the_verifier(self) -> None:
        env, task_index, action = self.make_light_task()
        initial_plan = Plan(
            [Action(name="unavailable_tool", kwargs={})],
            '{"actions": ["invalid"]}',
        )
        repair_plan = Plan([action], '{"actions": ["corrected"]}')
        planner = ScriptedRole("planner", [initial_plan])
        repairer = ScriptedRole("repair", [repair_plan])

        result = self.make_agent(planner, repairer).solve(env, task_index=task_index)

        self.assertEqual(result.reward, 0.0)
        self.assertEqual(evaluate_home_acceptance(env)["reward"], 1.0)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(repairer.calls, 1)
        self.assertEqual(result.messages[-2]["role"], "verifier")

    def test_unavailable_tool_is_not_executed(self) -> None:
        env, task_index, _ = self.make_light_task()
        invalid_plan = Plan(
            [Action(name="unavailable_tool", kwargs={})],
            '{"actions": ["invalid"]}',
        )
        planner = ScriptedRole("planner", [invalid_plan])
        repairer = ScriptedRole("repair", [])
        guarded = Monitor(env)
        result = self.make_agent(planner, repairer, max_repairs=0).solve(
            guarded, task_index=task_index
        )

        summary = result.info["failure_interceptor"]
        self.assertEqual(result.reward, 0.0)
        self.assertTrue(summary["intercepted"])
        self.assertEqual(summary["executed_action_count"], 0)

    def test_confirmation_alert_is_replanned_without_execution(self) -> None:
        env, task_index, action = self.make_light_task()
        invalid_plan = Plan(
            [
                Action(
                    name="create_alert",
                    kwargs={
                        "alert_type": "confirmation",
                        "message": "请确认是否执行。",
                    },
                )
            ],
            '{"actions": ["invalid confirmation alert"]}',
        )
        planner = ScriptedRole("planner", [invalid_plan])
        repairer = ScriptedRole("repair", [Plan([action], '{"actions": ["valid"]}')])
        guarded = Monitor(env)

        result = self.make_agent(planner, repairer).solve(
            guarded, task_index=task_index
        )

        self.assertEqual(evaluate_home_acceptance(guarded)["reward"], 1.0)
        self.assertEqual(repairer.calls, 1)
        self.assertEqual(guarded.guard_summary()["executed_action_count"], 1)
        self.assertFalse(
            any(
                "create_alert" in message["content"]
                for message in result.messages
                if message["role"] == "executor"
            )
        )

    def test_policy_confirmation_does_not_depend_on_instruction_keywords(self) -> None:
        env = MockHomeDomainEnv(task_split="all")
        task_index = next(
            index
            for index, task in enumerate(env.tasks)
            if task.category == "confirmation"
            and task.expected_state["devices"] == {"front_door_lock": {"locked": True}}
        )
        original_instruction = "请把入户门锁好。"
        env.tasks[task_index].instruction = original_instruction
        planner = ScriptedRole(
            "planner",
            [
                Plan(
                    [Action(name="lock_door", kwargs={"device_id": "front_door_lock"})],
                    '{"actions": ["lock"]}',
                )
            ],
        )
        repairer = ScriptedRole("repair", [])
        guarded = Monitor(env)

        result = self.make_agent(planner, repairer).solve(
            guarded, task_index=task_index
        )

        self.assertEqual(result.reward, 0.0)
        self.assertEqual(evaluate_home_acceptance(guarded)["reward"], 1.0)
        self.assertEqual(planner.last_kwargs["instruction"], original_instruction)
        self.assertNotIn("user_confirmation", planner.last_kwargs["context"])
        self.assertIn(
            "这项操作涉及家庭安全或隐私。请确认是否现在执行？",
            [message["content"] for message in result.messages],
        )
        self.assertTrue(env.current_policy["confirmation_provided"])

    def test_read_only_plan_does_not_request_confirmation(self) -> None:
        env = MockHomeDomainEnv(task_split="all")
        task_index = next(
            index
            for index, task in enumerate(env.tasks)
            if task.category == "confirmation"
            and task.expected_state["devices"] == {"front_door_lock": {"locked": True}}
        )
        planner = ScriptedRole(
            "planner",
            [
                Plan(
                    [Action(name="get_state", kwargs={"device_id": "front_door_lock"})],
                    '{"actions": ["read"]}',
                )
            ],
        )
        repairer = ScriptedRole("repair", [])
        guarded = Monitor(env)

        result = self.make_agent(planner, repairer).solve(
            guarded, task_index=task_index
        )

        self.assertFalse(guarded.intercepted)
        self.assertFalse(env.current_policy["confirmation_provided"])
        self.assertNotIn(
            "这项操作涉及家庭安全或隐私。请确认是否现在执行？",
            [message["content"] for message in result.messages],
        )


if __name__ == "__main__":
    unittest.main()
