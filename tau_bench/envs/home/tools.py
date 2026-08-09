"""Tool schemas for the deterministic home-control environment."""

from __future__ import annotations

from typing import Any

from tau_bench.envs.tool import Tool


def _parameters(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


DEVICE_ID = {
    "device_id": {
        "type": "string",
        "description": "房间可定位的设备标识，格式为 room_device_type，如 living_room_light、bedroom_air_conditioner；门锁为 front_door_lock，燃气阀为 kitchen_gas_valve。",
    }
}
MODE_VALUES = {
    "air_conditioner": ("auto", "cool", "heat", "dry", "fan", "sleep"),
    "air_purifier": ("auto", "low", "medium", "high", "sleep"),
    "dehumidifier": ("auto", "low", "medium", "high", "sleep"),
}
MODE_DESCRIPTION = (
    "空调可用 auto/cool/heat/dry/fan/sleep；"
    "空气净化器和除湿机可用 auto/low/medium/high/sleep。"
)

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_state": _parameters({**DEVICE_ID}),
    "get_sensor": _parameters({"sensor_id": {"type": "string"}}, ["sensor_id"]),
    "turn_on": _parameters({**DEVICE_ID}, ["device_id"]),
    "turn_off": _parameters({**DEVICE_ID}, ["device_id"]),
    "set_brightness": _parameters(
        {**DEVICE_ID, "brightness": {"type": "integer"}}, ["device_id", "brightness"]
    ),
    "set_temperature": _parameters(
        {**DEVICE_ID, "temperature": {"type": "number"}}, ["device_id", "temperature"]
    ),
    "set_position": _parameters(
        {**DEVICE_ID, "position": {"type": "integer"}}, ["device_id", "position"]
    ),
    "set_mode": _parameters(
        {**DEVICE_ID, "mode": {"type": "string", "description": MODE_DESCRIPTION}},
        ["device_id", "mode"],
    ),
    "set_level": _parameters(
        {**DEVICE_ID, "level": {"type": "integer"}}, ["device_id", "level"]
    ),
    "set_volume": _parameters(
        {**DEVICE_ID, "volume": {"type": "integer"}}, ["device_id", "volume"]
    ),
    "lock_door": _parameters({**DEVICE_ID}, ["device_id"]),
    "unlock_door": _parameters({**DEVICE_ID}, ["device_id"]),
    "open_gas_valve": _parameters({**DEVICE_ID}, ["device_id"]),
    "close_gas_valve": _parameters({**DEVICE_ID}, ["device_id"]),
    "enable_camera": _parameters({**DEVICE_ID}, ["device_id"]),
    "disable_camera": _parameters({**DEVICE_ID}, ["device_id"]),
    "start_recording": _parameters({**DEVICE_ID}, ["device_id"]),
    "stop_recording": _parameters({**DEVICE_ID}, ["device_id"]),
    "close_privacy_shutter": _parameters({**DEVICE_ID}, ["device_id"]),
    "create_alert": _parameters(
        {"alert_type": {"type": "string"}, "message": {"type": "string"}},
        ["alert_type", "message"],
    ),
}


def _make_tool(name: str, schema: dict[str, Any]) -> type[Tool]:
    description = f"家庭控制工具：{name}。只能操作任务允许的设备。"
    if name == "create_alert":
        description = (
            "创建持久的家庭告警。仅当用户明确要求创建告警或通知时使用；"
            "不得用于解释计划、请求确认或确认收到请求。"
        )
    info = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }

    def invoke(data: dict[str, Any], **kwargs: Any) -> str:
        # Home execution needs task-local faults and policies, so it is owned
        # by MockHomeDomainEnv.step rather than a stateless Tool class.
        raise RuntimeError("home tools must be executed by MockHomeDomainEnv")

    return type(
        "".join(part.title() for part in name.split("_")),
        (Tool,),
        {
            "invoke": staticmethod(invoke),
            "get_info": staticmethod(lambda info=info: info),
        },
    )


ALL_TOOLS = [_make_tool(name, schema) for name, schema in TOOL_SCHEMAS.items()]
