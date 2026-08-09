# Copyright Sierra

import json
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


IDENTITY_REQUEST = (
    "To authenticate your account, please provide either your email address or "
    "your first name, last name, and ZIP code."
)
EXECUTION_PROTOCOL = """\
Operational requirements:
- Never invent identity credentials. Ask the user for them when absent.
- A read-only lookup error is recoverable: recheck the prior conversation and retry or ask a clarifying question.
- Do not transfer a user to a human for a lookup error.\
"""


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        # ``think`` is an implementation-only no-op, and retail rules prohibit
        # transferring a request that can be handled by the available tools.
        self.tools_info = [
            tool
            for tool in tools_info
            if tool["function"]["name"] not in {"think", "transfer_to_human_agents"}
        ]
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.seed = seed

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": f"{self.wiki}\n\n{EXECUTION_PROTOCOL}"},
            {"role": "user", "content": obs},
        ]
        requires_identity = getattr(env, "requires_identity", True)
        done = False
        if requires_identity:
            # Do not let the model invent an email address just to satisfy the
            # authentication policy. The user simulator provides credentials.
            identity_action = Action(
                name=RESPOND_ACTION_NAME, kwargs={"content": IDENTITY_REQUEST}
            )
            identity_response = env.step(identity_action)
            reward = identity_response.reward
            info = {**info, **identity_response.info.model_dump()}
            messages.extend(
                [
                    {"role": "assistant", "content": IDENTITY_REQUEST},
                    {"role": "user", "content": identity_response.observation},
                ]
            )
            done = identity_response.done

        for _ in range(max_num_steps - (1 if requires_identity else 0)):
            if done:
                break
            expected_mutations = getattr(env, "expected_mutations", None)
            executed_mutations = getattr(env, "executed_mutations", None)
            tools_for_turn = (
                None
                if expected_mutations and executed_mutations == expected_mutations
                else self.tools_info
            )
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=tools_for_turn,
                temperature=self.temperature,
                seed=self.seed,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            usage = getattr(res, "usage", None) or res.get("usage")
            if usage is not None:
                for key in token_usage:
                    value = getattr(usage, key, None)
                    if value is None and isinstance(usage, dict):
                        value = usage.get(key)
                    if isinstance(value, (int, float)):
                        token_usage[key] += int(value)
            is_home_env = bool(getattr(env, "is_home_env", False))
            tool_calls = next_message.get("tool_calls") or []
            # Qwen may return a compound home request as several tool calls in
            # one assistant turn. Execute each call in order so the gate can
            # observe and validate the complete plan. Retail keeps the
            # historical one-call-per-turn behavior.
            if is_home_env and len(tool_calls) > 1:
                actions = [
                    message_to_action({"tool_calls": [tool_call]})
                    for tool_call in tool_calls
                ]
            else:
                actions = [message_to_action(next_message)]
            tool_responses = []
            env_response = None
            for action in actions:
                env_response = env.step(action)
                tool_responses.append((action, env_response))
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}
                if env_response.done:
                    done = True
                    break
            if tool_calls:
                executed_count = len(tool_responses)
                next_message["tool_calls"] = tool_calls[:executed_count]
                messages.append(next_message)
                for index, (action, response) in enumerate(tool_responses):
                    tool_content = response.observation
                    if (
                        expected_mutations
                        and executed_mutations == expected_mutations
                        and action.name
                        in {mutation.name for mutation in expected_mutations}
                    ):
                        tool_content += (
                            "\n\nAll required state-changing actions are complete. "
                            "Respond to the user now without calling another tool."
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][index]["id"],
                            "name": next_message["tool_calls"][index]["function"][
                                "name"
                            ],
                            "content": tool_content,
                        }
                    )
            else:
                action = tool_responses[0][0]
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if (env_response is not None and env_response.done) or done:
                done = True
                break
        if not done and hasattr(env, "abort"):
            abort_response = env.abort("agent exceeded the maximum number of steps")
            reward = abort_response.reward
            info = {**info, **abort_response.info.model_dump()}
        info["token_usage"] = token_usage
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
