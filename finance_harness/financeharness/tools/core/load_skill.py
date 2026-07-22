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

"""load_skill — pull a workflow recipe into context + auto-load its tools.

Returns each requested skill's SKILL.md body (the recipe the model now follows)
and auto-loads the schemas of every tool in the skill's `requires_tools`, so the
model doesn't need a separate load_tool call.
"""

from __future__ import annotations

from financeharness.runtime.skill_registry import SkillRegistry, SkillSessionState, SkillSpec
from financeharness.runtime.tool_registry import (
    ToolRegistry,
    ToolResponse,
    ToolSessionState,
    ToolSpec,
)
from pydantic import BaseModel, Field


class LoadSkillRequest(BaseModel):
  """Input for loading workflow recipes into the current run."""

  skills: list[str] = Field(
      Ellipsis,
      min_length=1,
      description=(
          "Skill names to load (kebab-case from the skills catalog), "
          "e.g. ['dcf-valuation']. Batch them."
      ),
  )


def _render(spec, auto_loaded):
  parts = [f"## Skill: `{spec.name}`", "", spec.body]
  if auto_loaded:
    names = ", ".join(f"`{n}`" for n in auto_loaded)
    parts += ["", "---", f"Auto-loaded tools (callable now): {names}."]
  return "\n".join(parts)


def build_load_skill_spec(
    skill_state,
    tool_state,
    skill_registry,
    tool_registry,
):
  """Build the per-run `load_skill` tool bound to the run's skill + tool state."""

  async def handler(req):
    parts: list[str] = []
    loaded: list[str] = []
    unknown: list[str] = []
    for name in req.skills:
      spec = skill_registry.get(name)
      if spec is None:
        unknown.append(name)
        continue
      skill_state.loaded.add(name)
      auto: list[str] = []
      if spec.requires_tools:
        # auto-load required tools; silently skip any not registered
        known = [
            t for t in spec.requires_tools if tool_registry.get(t) is not None
        ]
        if known:
          tool_state.load(known, tool_registry)
          auto = known
      parts.append(_render(spec, auto))
      loaded.append(name)
    if unknown:
      parts.append(
          f"Unknown skill(s): {unknown}. Available: {skill_registry.names()}"
      )
    return ToolResponse(
        markdown="\n\n---\n\n".join(parts) if parts else "no skills loaded",
        structured={"loaded": loaded, "unknown": unknown},
        meta={"count": len(loaded)},
    )

  return ToolSpec(
      name="load_skill",
      display_name="core.load_skill",
      tier="core",
      description=(
          "Load one or more workflow-recipe skills (kebab-case from the skills"
          " catalog) when the question matches. Each skill's required tools are"
          " auto-loaded."
      ),
      request_schema=LoadSkillRequest,
      handler=handler,
      tags=("core", "meta"),
  )
