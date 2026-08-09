from __future__ import annotations

import argparse

from litellm import provider_list

from tau_bench.envs.user import UserStrategy
from tau_bench.run import run
from tau_bench.types import RunConfig


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline", "home"], default="home"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model-provider", type=str, choices=provider_list, required=True
    )
    parser.add_argument("--user-model", type=str, default="gpt-4o")
    parser.add_argument(
        "--user-model-provider", type=str, choices=provider_list, default="openai"
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot", "planner-repair"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--task-split",
        type=str,
        default="all",
        choices=["train", "test", "dev", "all"],
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument(
        "--user-strategy",
        type=str,
        default="human",
        choices=[item.value for item in UserStrategy],
    )
    parser.add_argument("--few-shot-displays-path", type=str)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        guarded_execution=args.monitor,
        max_repairs=args.max_repairs,
    )


if __name__ == "__main__":
    run(parse_args())
