"""Deterministic home-control environment backed by Huawei/home/tasks_expanded.json."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Optional, Union

from tau_bench.envs.base import Env
from tau_bench.envs.home.tools import ALL_TOOLS, MODE_VALUES
from tau_bench.envs.home.verifier import evaluate_home_acceptance, subset_matches
from tau_bench.envs.home.wiki import WIKI
from tau_bench.envs.user import UserStrategy
from tau_bench.types import (
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RewardActionInfo,
    RewardResult,
    Task,
    RESPOND_ACTION_NAME,
)


HOME_DATA_DIR = Path(__file__).resolve().parents[3] / "home"
HOME_TASK_PATH = HOME_DATA_DIR / "tasks_expanded.json"
HOME_SEED_TASK_PATH = HOME_DATA_DIR / "tasks.json"

# Kept for external scripts that imported the historical private helper.
_subset_matches = subset_matches


class HomeTask(Task):
    task_id: str
    split: Optional[str] = None
    template_family: Optional[str] = None
    category: str
    difficulty: str
    initial_state: dict[str, Any]
    expected_state: dict[str, Any]
    allowed_tools: list[str]
    max_agent_loops: int
    fault_injection: list[dict[str, Any]]
    policy: dict[str, Any]
    expected_outcome: str
    # Optional deterministic follow-up messages.  These model the staged
    # clarification/confirmation turns that make retail and airline tasks
    # interactive, without exposing task gold labels to an Agent.
    dialogue: list[dict[str, Any]] = []


def load_home_tasks() -> list[HomeTask]:
    task_path = HOME_TASK_PATH if HOME_TASK_PATH.exists() else HOME_SEED_TASK_PATH
    raw = json.loads(task_path.read_text(encoding="utf-8"))["tasks"]
    return [
        HomeTask(
            user_id=task["policy"].get("user_role", "resident"),
            actions=[],
            outputs=[],
            **task,
        )
        for task in raw
    ]


def split_home_tasks(tasks: list[HomeTask], split: str) -> list[HomeTask]:
    """Return the unified benchmark, with legacy fallback for seed tasks."""
    if any(task.split == "all" for task in tasks):
        if split != "all":
            raise ValueError("The expanded home benchmark uses the single 'all' split")
        return tasks
    explicit_tasks = [task for task in tasks if task.split == split]
    if explicit_tasks:
        return explicit_tasks
    offsets = {"train": range(0, 5), "dev": range(5, 7), "test": range(7, 10)}
    if split not in offsets:
        raise ValueError(f"Unknown task split: {split}")
    selected: list[HomeTask] = []
    for start in range(0, len(tasks), 10):
        selected.extend(tasks[start + offset] for offset in offsets[split])
    return selected


FIELD_BY_TOOL = {
    "turn_on": ("power", True),
    "turn_off": ("power", False),
    "set_brightness": ("brightness", "brightness"),
    "set_temperature": ("target_temperature", "temperature"),
    "set_position": ("position", "position"),
    "set_mode": ("mode", "mode"),
    "set_level": ("level", "level"),
    "set_volume": ("volume", "volume"),
    "lock_door": ("locked", True),
    "unlock_door": ("locked", False),
    "open_gas_valve": ("open", True),
    "close_gas_valve": ("open", False),
    "enable_camera": ("enabled", True),
    "disable_camera": ("enabled", False),
    "start_recording": ("recording", True),
    "stop_recording": ("recording", False),
    "close_privacy_shutter": ("privacy_shutter", True),
}


class MockHomeDomainEnv(Env):
    """A stateful home domain with deterministic user input and fault injection."""

    is_home_env = True
    requires_identity = False

    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = None,
        user_seed: Optional[int] = None,
        task_split: str = "all",
        task_index: Optional[int] = None,
    ) -> None:
        tasks = split_home_tasks(load_home_tasks(), task_split)
        # Home tasks provide their own deterministic user utterance. Avoid an
        # unnecessary LLM user simulation while retaining the Env interface.
        super().__init__(
            data_load_func=lambda: {"devices": {}, "sensors": {}},
            tools=ALL_TOOLS,
            tasks=tasks,
            wiki=WIKI,
            rules=[],
            user_strategy=UserStrategy.HUMAN,
            user_model=user_model,
            user_provider=user_provider,
            user_seed=user_seed,
            # Env.__init__ uses randint with an inclusive upper bound when no
            # index is supplied. HomeEnv selects its task safely in reset().
            task_index=0 if task_index is None else task_index,
        )
        self.task_split = task_split
        self.initial_data: dict[str, Any] = {}
        self.fault_triggered: list[str] = []
        self.fault_trigger_action_index: Optional[int] = None
        self.tool_attempts: dict[str, int] = {}
        self.dialogue_index = 0
        self.current_policy: dict[str, Any] = {}

    @property
    def home_task(self) -> HomeTask:
        return self.task  # type: ignore[return-value]

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        self.task_index = (
            random.randrange(len(self.tasks)) if task_index is None else task_index
        )
        self.task = self.tasks[self.task_index]
        self.actions = []
        self.data = copy.deepcopy(self.home_task.initial_state)
        self.data.setdefault("devices", {})
        self.data.setdefault("sensors", {})
        self.data.setdefault("alerts", [])
        self.data["version"] = 0
        self.initial_data = copy.deepcopy(self.data)
        self.fault_triggered = []
        self.fault_trigger_action_index = None
        self.tool_attempts = {}
        self.dialogue_index = 0
        self.current_policy = copy.deepcopy(self.home_task.policy)
        return EnvResetResponse(
            observation=self.home_task.instruction,
            info=self.public_env_info(source="user"),
        )

    def public_env_info(self, source: Optional[str] = None) -> EnvInfo:
        """Return only runtime-facing task data across the Agent boundary."""
        return EnvInfo(
            task=Task(
                user_id=self.home_task.user_id,
                actions=[],
                instruction=self.home_task.instruction,
                outputs=[],
            ),
            source=source,
        )

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        if action.name == RESPOND_ACTION_NAME:
            dialogue = self.home_task.dialogue
            if self.dialogue_index < len(dialogue):
                turn = dialogue[self.dialogue_index]
                requires_question = bool(turn.get("requires_question"))
                content = str(action.kwargs.get("content", ""))
                if (
                    requires_question
                    and "?" not in content
                    and "？" not in content
                    and "确认" not in content
                ):
                    return EnvResponse(
                        observation="请先向用户说明需要确认的操作，再继续。",
                        reward=0.0,
                        done=False,
                        info=self.public_env_info(source="user"),
                    )
                self.dialogue_index += 1
                policy_update = turn.get("policy_update")
                if isinstance(policy_update, dict):
                    self.current_policy.update(policy_update)
                return EnvResponse(
                    observation=str(turn["content"]),
                    reward=0.0,
                    done=False,
                    info=self.public_env_info(source="user"),
                )
            return EnvResponse(
                observation="###STOP###",
                # Gold labels are intentionally unavailable while the Agent is
                # running. The external Harness evaluates this terminal trace.
                reward=0.0,
                done=True,
                info=self.public_env_info(source="user"),
            )
        if action.name not in self.tools_map:
            observation = f"Error: unknown tool {action.name}"
        else:
            observation = self._execute(action.name, action.kwargs)
        return EnvResponse(
            observation=observation,
            reward=0.0,
            done=False,
            info=self.public_env_info(source=action.name),
        )

    def calculate_reward(self) -> RewardResult:
        # Retained for the Env interface. Production runs invoke this only
        # from the post-run Harness, never from an Agent action.
        acceptance = evaluate_home_acceptance(self)
        satisfied = acceptance["reward"] == 1.0
        return RewardResult(
            reward=1.0 if satisfied else 0.0,
            info=RewardActionInfo(
                r_actions=1.0 if satisfied else 0.0, gt_data_hash="home_expected_state"
            ),
            actions=self.actions,
        )

    def fault_recovery_satisfied(self) -> bool:
        """Harness-only check for an unguarded agent stopping after a fault."""
        if not self.fault_triggered or self.fault_trigger_action_index is None:
            return False
        if not _subset_matches(self.home_task.expected_state, self.data):
            return False

        follow_up = self.actions[self.fault_trigger_action_index + 1 :]
        if len(follow_up) != 1 or follow_up[0].name != RESPOND_ACTION_NAME:
            return False

        content = str(follow_up[0].kwargs.get("content", "")).lower()
        failure_markers = (
            "无法",
            "失败",
            "错误",
            "故障",
            "离线",
            "超时",
            "拒绝",
            "权限",
            "冲突",
            "不支持",
            "未完成",
            "不能",
            "请检查",
            "停止",
            "unable",
            "failed",
            "error",
            "offline",
            "timeout",
            "denied",
        )
        return any(marker in content for marker in failure_markers)

    def rollback(self) -> None:
        self.data = copy.deepcopy(self.initial_data)

    def _execute(self, name: str, args: dict[str, Any]) -> str:
        self.tool_attempts[name] = self.tool_attempts.get(name, 0) + 1
        if name == "get_state":
            device_id = args.get("device_id")
            if device_id is None:
                return json.dumps(
                    {"devices": self.data["devices"], "version": self.data["version"]},
                    ensure_ascii=False,
                )
            if isinstance(device_id, str):
                device_id = self._resolve_device_id(device_id)
            device = self.data["devices"].get(device_id)
            return (
                f"Error: device not found: {device_id}"
                if device is None
                else json.dumps(
                    {
                        "device_id": device_id,
                        "state": device,
                        "version": self.data["version"],
                    },
                    ensure_ascii=False,
                )
            )
        if name == "get_sensor":
            sensor_id = args.get("sensor_id")
            if sensor_id not in self.data["sensors"]:
                return f"Error: sensor not found: {sensor_id}"
            return json.dumps(
                {"sensor_id": sensor_id, "value": self.data["sensors"][sensor_id]},
                ensure_ascii=False,
            )
        if name == "create_alert":
            self.data["alerts"].append(copy.deepcopy(args))
            self.data["version"] += 1
            return json.dumps({"accepted": True, "version": self.data["version"]})

        device_id = args.get("device_id")
        if not isinstance(device_id, str):
            return "Error: device_id is required"
        device_id = self._resolve_device_id(device_id)
        fault = self._trigger_fault(name, device_id, args)
        if fault is not None:
            return fault
        device = self.data["devices"].get(device_id)
        if device is None:
            return f"Error: device not found: {device_id}"
        if not device.get("online", True):
            return f"Error: device offline: {device_id}"
        if name not in FIELD_BY_TOOL:
            return f"Error: unsupported home tool {name}"
        field, value = FIELD_BY_TOOL[name]
        if field not in device:
            return f"Error: unsupported capability {field} for {device_id}"
        if isinstance(value, str):
            value = args.get(value)
        if name == "set_temperature" and not 16 <= value <= 30:
            return "Error: temperature must be between 16 and 30"
        if (
            name in {"set_brightness", "set_position", "set_volume"}
            and not 0 <= value <= 100
        ):
            return f"Error: {field} must be between 0 and 100"
        if name == "set_level" and not 1 <= value <= 5:
            return "Error: level must be between 1 and 5"
        if name == "set_mode" and value not in MODE_VALUES.get(device["type"], ()):
            return f"Error: unsupported mode {value} for {device['type']}"
        device[field] = value
        self.data["version"] += 1
        return json.dumps(
            {
                "accepted": True,
                "device_id": device_id,
                "state": device,
                "version": self.data["version"],
            },
            ensure_ascii=False,
        )

    def _resolve_device_id(self, device_id: str) -> str:
        """Accept the two conventional English identifiers for the study room."""
        if device_id.startswith("study_room_"):
            alternate = "study_" + device_id.removeprefix("study_room_")
        elif device_id.startswith("study_"):
            alternate = "study_room_" + device_id.removeprefix("study_")
        else:
            return device_id
        return (
            alternate
            if device_id not in self.data["devices"]
            and alternate in self.data["devices"]
            else device_id
        )

    def _trigger_fault(
        self, name: str, device_id: str, args: dict[str, Any]
    ) -> str | None:
        for fault in self.home_task.fault_injection:
            target = str(fault.get("target", ""))
            target_device = target.split(".")[0]
            trigger_tool = fault.get("trigger_tool")
            matches = (not trigger_tool or trigger_tool == name) and (
                not target_device or target_device == device_id
            )
            if not matches:
                continue
            kind = fault["type"]
            if kind == "invalid_parameter":
                parameter = fault.get("parameter")
                if not isinstance(parameter, str) or args.get(parameter) != fault.get(
                    "value"
                ):
                    continue
            if kind == "planner_invalid_plan" and self.tool_attempts[name] > fault.get(
                "fail_first_n_loops", 1
            ):
                continue
            self.fault_triggered.append(kind)
            if self.fault_trigger_action_index is None:
                self.fault_trigger_action_index = len(self.actions) - 1
            if kind == "silent_state_failure":
                return json.dumps(
                    {
                        "accepted": True,
                        "version": self.data["version"],
                        "warning": "success_but_state_unchanged",
                    }
                )
            if kind == "partial_failure":
                # Model a device applying the first write before reporting a
                # terminal error. Monitor must roll this mutation back.
                device = self.data["devices"].get(device_id)
                field, value = FIELD_BY_TOOL.get(name, (None, None))
                if device is not None and field in device:
                    if isinstance(value, str):
                        value = args.get(value)
                    device[field] = value
                    self.data["version"] += 1
            messages = {
                "device_offline": "Error: injected device offline",
                "tool_timeout": "Error: injected tool timeout",
                "device_not_found": "Error: injected device not found",
                "invalid_parameter": "Error: injected invalid parameter",
                "state_conflict": "Error: injected state conflict",
                "unsupported_capability": "Error: injected unsupported capability",
                "partial_failure": "Error: injected partial failure",
                "planner_invalid_plan": "Error: injected planner failure",
                "permission_denied": "Error: injected permission denied",
            }
            return messages[kind]
        return None
