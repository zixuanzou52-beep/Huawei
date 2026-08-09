# Home-Control Multi-Agent Benchmark

This project is a smart-home tool-use benchmark built on a local `tau-bench` runtime. It evaluates whether a language model can translate Chinese home-control requests into correct tool calls, and compares a single-agent baseline with a monitored Planner/Repair multi-agent workflow.

The project runs against a stateful simulated home environment. It does not control physical devices. The benchmark includes lighting, curtains, air conditioners, air purifiers, locks, gas valves, cameras, and other household devices.

## Architecture

The multi-agent strategy separates five responsibilities:

```text
User instruction -> Planner -> Monitor -> Executor -> Home environment
                              ^                       |
                              |                       v
                           Repair <----------- Execution feedback
                                       |
                                       v
                                   Verifier
```

- `Planner`: Uses an LLM to turn an instruction, public state, and tool schemas into a JSON action plan.
- `Repair`: Replans after observable execution failures, invalid plans, or parameter errors.
- `Executor`: Deterministically dispatches planned tool calls in order.
- `Monitor`: Validates schemas, confirmation and safety policies, tool failures, and state changes. It rolls the task state back on interception.
- `Verifier`: Scores the completed trajectory against the task contract after the agent stops.

The `tool-calling` strategy is the single-agent baseline. In both modes, the score depends on final state and interception outcomes, not on the natural-language completion alone.

## Dataset and Evaluation

Each task contains an initial state, a natural-language instruction, reference operations, policy and fault conditions, and an evaluation contract. During dataset construction, `expected_state` is derived by applying the reference operations to the initial device state. Fault tasks mainly use `expected_outcome: "intercepted"` as their contract.

The model does not receive `expected_state`, `expected_outcome`, `fault_injection`, or the reference operations. The Verifier reads those fields only after the agent has finished.

- Completed task: the agent must not be intercepted, must terminate, and the final runtime state must contain every field in `expected_state`.
- Fault or policy task: an appropriate safety/failure interception must be observed. Without Monitor, a limited safe-stop recovery path is supported.
- All other cases, including saying "done" without reaching the expected state, receive a reward of `0`.


## Repository Layout

| Path | Purpose |
| --- | --- |
| `run_benchmark.py` | Command-line entry point. Parses model, strategy, task-range, and Monitor options, then runs an evaluation. |
| `multi_agent/` | Core multi-agent implementation: planning, repair, execution, monitoring, and verification. |
| `tau_bench/` | Local tau-bench runtime, including agents, environments, tool definitions, guards, and model adapters. |
| `tau_bench/envs/home/` | Home environment, device tools, public wiki, verifier adapter, and home-environment tests. |
| `tau_bench/agents/` | Agent strategies. `planner_repair_agent.py` is the multi-agent compatibility entry point and contains related tests. |
| `tau_bench/guards/` | Guard compatibility interfaces. The home environment uses `multi_agent.monitor.Monitor`. |
| `tau_bench/model_utils/` | LiteLLM and provider-specific model invocation utilities. |
| `home/` | Task data, authored task cards, and dataset generation/validation scripts. `tasks_expanded.json` is the runtime task dataset. |
| `results/` | Saved baseline and multi-agent result JSON files. |


## Prerequisites

Run commands from the `Huawei/` directory with Python 3. The environment must include the project dependencies, notably `litellm`.

For an OpenAI-compatible Yunwu endpoint, set the credentials before running a benchmark:

```sh
export YUNWU_API_KEY='your-api-key'
export YUNWU_API_BASE='https://yunwu.ai/v1'
```

`YUNWU_API_BASE` is optional and defaults to `https://yunwu.ai/v1` when `YUNWU_API_KEY` is set.

## Run the Benchmark

Run the monitored Planner/Repair multi-agent benchmark:

```sh
cd /home/zouzixuan/code-mm/Huawei

python run_benchmark.py \
  --env home \
  --model '<model-name>' \
  --model-provider openai \
  --agent-strategy planner-repair \
  --monitor \
  --task-split all \
  --max-repairs 1 \
  --log-dir results
```

Run the single-agent baseline:

```sh
cd /home/zouzixuan/code-mm/Huawei

python run_benchmark.py \
  --env home \
  --model '<model-name>' \
  --model-provider openai \
  --agent-strategy tool-calling \
  --task-split all \
  --log-dir results
```

Run a small task range while debugging:

```sh
python run_benchmark.py \
  --env home \
  --model '<model-name>' \
  --model-provider openai \
  --agent-strategy planner-repair \
  --monitor \
  --start-index 0 \
  --end-index 5 \
  --log-dir results
```

Use `--task-ids 0 5 12` to select explicit task indices. Output is written as a timestamped JSON file under `--log-dir`.

## Tests

Run the home environment and Planner/Repair test suites:

```sh
cd /home/zouzixuan/code-mm/Huawei

python -m unittest \
  tau_bench.envs.home.test_home_env \
  tau_bench.agents.test_planner_repair_agent
```

These tests cover task counts and fault distribution, gold-label isolation from the agent boundary, confirmation flows, interception rollback, fault recovery, and Planner/Repair execution behavior.

