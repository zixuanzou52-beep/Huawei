"""Compatibility adapter for Huawei's Monitor implementation."""

from multi_agent.monitor import Monitor

MMonitor = Monitor
HomeGuardedEnv = Monitor

__all__ = ["HomeGuardedEnv", "MMonitor", "Monitor"]
