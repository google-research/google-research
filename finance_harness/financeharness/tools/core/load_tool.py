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

"""load_tool — fetch deferred-tool schemas on demand (the ToolSearch pattern).

The model sees deferred tools as catalog lines only. Calling load_tool marks
them loaded in the session state (so the next turn's `tools=` includes their
schemas) and returns the schemas inline this turn.
"""

from __future__ import annotations

import json

from financeharness.runtime.tool_registry import (
    ToolRegistry,
    ToolResponse,
    ToolSessionState,
    ToolSpec,
)
from pydantic import BaseModel, Field


class LoadToolRequest(BaseModel):
  """Input for loading deferred tool schemas into the current run."""

  tools: list[str] = Field(
      Ellipsis,
      min_length=1,
      description=(
          "Deferred tool wire-names to load, e.g. ['data_equity_reference']."
          " Batch them."
      ),
  )


def _spec_md(spec):
  schema = json.dumps(spec.request_schema.model_json_schema(), indent=2)
  return (
      f"### `{spec.name}` ({spec.display_name})\n{spec.description}\n\n"
      f"Arguments schema:\n```json\n{schema}\n```"
  )


def build_load_tool_spec(
    session_state, registry
):
  """Build the per-run `load_tool` tool bound to the run's session state."""

  async def handler(req):
    loaded, unknown = session_state.load(req.tools, registry)
    parts = [_spec_md(s) for s in loaded]
    if unknown:
      parts.append(
          f"Unknown tool name(s): {unknown}. Known deferred tools: "
          f"{sorted(s.name for s in registry.deferred_tools())}"
      )
    return ToolResponse(
        markdown="\n\n---\n\n".join(parts) if parts else "no tools loaded",
        structured={"loaded": [s.name for s in loaded], "unknown": unknown},
        meta={"count": len(loaded)},
    )

  return ToolSpec(
      name="load_tool",
      display_name="core.load_tool",
      tier="core",
      description=(
          "Load schemas for one or more deferred tools (use the underscored"
          " catalog names) so they can be called on the next turn. Batch all"
          " you need into one call."
      ),
      request_schema=LoadToolRequest,
      handler=handler,
      tags=("core", "meta"),
  )
