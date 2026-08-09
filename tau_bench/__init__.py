# Copyright Sierra

import sys
from pathlib import Path

# The project-specific multi-agent implementation deliberately lives outside
# the vendored tau-bench package. Keep the source-tree runner importable
# without requiring an editable installation of the Huawei project package.
HUAWEI_ROOT = Path(__file__).resolve().parents[1]
if str(HUAWEI_ROOT) not in sys.path:
    sys.path.insert(0, str(HUAWEI_ROOT))

from tau_bench.agents.base import Agent as Agent
from tau_bench.envs.base import Env as Env
