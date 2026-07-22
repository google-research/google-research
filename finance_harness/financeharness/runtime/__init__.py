# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The agent runtime: agent loop, never-raise dispatch, registries,

reference-chaining, recovery policy, context budget, config.
"""

from financeharness.runtime.agent import Agent
from financeharness.runtime.chaining import resolve_references
from financeharness.runtime.config import RuntimeConfig, load_runtime_config
from financeharness.runtime.dispatch import DispatchResult, dispatch, dispatch_json_args
from financeharness.runtime.prompts import build_system_prompt
from financeharness.runtime.tool_registry import (
    ToolRegistry,
    ToolResponse,
    ToolSessionState,
    ToolSpec,
)

__all__ = [
    "Agent",
    "DispatchResult",
    "RuntimeConfig",
    "ToolRegistry",
    "ToolResponse",
    "ToolSessionState",
    "ToolSpec",
    "build_system_prompt",
    "dispatch",
    "dispatch_json_args",
    "load_runtime_config",
    "resolve_references",
]
