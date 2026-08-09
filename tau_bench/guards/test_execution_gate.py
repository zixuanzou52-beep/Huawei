from __future__ import annotations

import unittest

from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.guards.execution_gate import GuardedEnv
from tau_bench.types import Action, RESPOND_ACTION_NAME


class StopUser:
    def reset(self, instruction: str | None = None) -> str:
        return "请帮我处理这个请求。"

    def step(self, content: str) -> str:
        return "###STOP###"

    def get_total_cost(self) -> float:
        return 0.0


class ExecutionGateTest(unittest.TestCase):
    def make_env(self, task_index: int, task_split: str = "test") -> GuardedEnv:
        env = MockRetailDomainEnv(
            user_strategy="human", task_index=task_index, task_split=task_split
        )
        env.user = StopUser()
        guarded = GuardedEnv(env)
        guarded.reset(task_index=task_index)
        return guarded

    def test_mutation_before_authentication_is_blocked_before_state_change(
        self,
    ) -> None:
        guarded = self.make_env(0)
        initial_hash = guarded.get_data_hash()
        response = guarded.step(
            Action(
                name="cancel_pending_order",
                kwargs={"order_id": "#W2378156", "reason": "ordered by mistake"},
            )
        )
        self.assertTrue(response.done)
        self.assertEqual(response.reward, 0.0)
        self.assertEqual(guarded.get_data_hash(), initial_hash)
        self.assertIn(
            "before successful user authentication", guarded.interception_reason or ""
        )

    def test_read_only_authentication_is_allowed_before_mutation(self) -> None:
        guarded = self.make_env(0)
        authentication = guarded.expected_actions[0]
        self.assertIn(
            authentication.name, {"find_user_id_by_email", "find_user_id_by_name_zip"}
        )
        initial_hash = guarded.get_data_hash()
        response = guarded.step(authentication)
        self.assertFalse(response.done)
        self.assertFalse(guarded.intercepted)
        self.assertEqual(guarded.get_data_hash(), initial_hash)
        self.assertEqual(guarded.executed_mutations, [])

    def test_wrong_mutation_is_blocked_after_authentication(self) -> None:
        guarded = self.make_env(0)
        guarded.step(guarded.expected_actions[0])
        initial_hash = guarded.get_data_hash()
        response = guarded.step(
            Action(
                name="cancel_pending_order",
                kwargs={"order_id": "#W2378156", "reason": "ordered by mistake"},
            )
        )
        self.assertTrue(response.done)
        self.assertEqual(guarded.get_data_hash(), initial_hash)
        self.assertIn(
            "unexpected state-changing action", guarded.interception_reason or ""
        )

    def test_abort_rolls_back_partial_mutation(self) -> None:
        guarded = self.make_env(0)
        initial_hash = guarded.get_data_hash()
        for action in guarded.expected_actions:
            guarded.step(action)
        self.assertNotEqual(guarded.get_data_hash(), initial_hash)
        response = guarded.abort("agent exceeded the maximum number of steps")
        self.assertTrue(response.done)
        self.assertEqual(response.reward, 0.0)
        self.assertEqual(guarded.get_data_hash(), initial_hash)
        self.assertTrue(guarded.intercepted)

    def test_return_item_order_is_semantically_equivalent(self) -> None:
        guarded = self.make_env(0, task_split="train")
        guarded.step(
            Action(
                name="find_user_id_by_name_zip",
                kwargs={"first_name": "Omar", "last_name": "Anderson", "zip": "19031"},
            )
        )
        expected = guarded.expected_mutations[0]
        response = guarded.step(
            Action(
                name=expected.name,
                kwargs={
                    **expected.kwargs,
                    "item_ids": list(reversed(expected.kwargs["item_ids"])),
                },
            )
        )
        self.assertFalse(response.done)
        self.assertFalse(guarded.intercepted)
        self.assertEqual(len(guarded.executed_mutations), 1)

    def test_read_only_error_is_returned_for_recovery(self) -> None:
        guarded = self.make_env(0)
        response = guarded.step(
            Action(name="get_order_details", kwargs={"order_id": "#W0000000"})
        )
        self.assertFalse(response.done)
        self.assertEqual(response.observation, "Error: order not found")
        self.assertFalse(guarded.intercepted)

    def test_interception_rolls_back_prior_mutation(self) -> None:
        guarded = self.make_env(0)
        initial_hash = guarded.get_data_hash()
        for action in guarded.expected_actions:
            guarded.step(action)
        self.assertNotEqual(guarded.get_data_hash(), initial_hash)
        response = guarded.step(guarded.expected_mutations[0])
        self.assertTrue(response.done)
        self.assertEqual(guarded.get_data_hash(), initial_hash)

    def test_reference_tool_trace_and_terminal_response_pass(self) -> None:
        guarded = self.make_env(0)
        for action in guarded.expected_actions:
            response = guarded.step(action)
            self.assertFalse(response.done)
        response = guarded.step(
            Action(name=RESPOND_ACTION_NAME, kwargs={"content": "处理完成"})
        )
        self.assertTrue(response.done)
        self.assertEqual(response.reward, 1.0)
        self.assertFalse(guarded.intercepted)

    def test_premature_terminal_response_is_intercepted_from_pre_reset_snapshot(
        self,
    ) -> None:
        guarded = self.make_env(0)
        response = guarded.step(
            Action(name=RESPOND_ACTION_NAME, kwargs={"content": "已完成"})
        )
        self.assertTrue(response.done)
        self.assertEqual(response.reward, 0.0)
        self.assertTrue(guarded.intercepted)
        self.assertIn(
            "terminal state-changing action trace", guarded.interception_reason or ""
        )


if __name__ == "__main__":
    unittest.main()
