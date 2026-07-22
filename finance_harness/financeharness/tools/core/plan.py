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

"""update_plan — a live research plan the model maintains for multi-step work.

The model resends the full ordered list of steps (each with a status) whenever
the
plan changes; the tool emits a ``plan`` event so the UI can render a checklist
that
advances as the run proceeds (transparency on long deep-dive trajectories). Pure
+
stateless — the model owns the plan; this just broadcasts it.
"""

from __future__ import annotations

from typing import Literal

from financeharness.runtime.tool_events import emit_tool_event
from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from pydantic import BaseModel, Field

PlanStatus = Literal["pending", "in_progress", "completed"]


class PlanStep(BaseModel):
  """One item in the live research plan."""

  step: str = Field(
      Ellipsis, description="A short, specific, actionable step (imperative phrase)."
  )
  status: PlanStatus = Field(
      "pending", description="One of: pending, in_progress, completed."
  )


class UpdatePlanRequest(BaseModel):
  """Input for replacing the live plan with its latest full state."""

  steps: list[PlanStep] = Field(
      Ellipsis,
      min_length=1,
      description=(
          "The full ordered plan as a list of {step, status} objects — resend "
          "every step (with its current status) on each update, e.g. "
          '[{"step": "Pull NVDA fundamentals", "status": "completed"}, '
          '{"step": "Run a DCF", "status": "in_progress"}, '
          '{"step": "Write the report", "status": "pending"}].'
      ),
  )


async def _handler(req):
  items = [{"step": s.step, "status": s.status} for s in req.steps]
  emit_tool_event("plan", {"items": items})
  done = sum(1 for s in req.steps if s.status == "completed")
  return ToolResponse(
      markdown=f"Plan updated — {done}/{len(items)} step(s) done.",
      structured={"items": items},
      meta={"steps": len(items), "completed": done},
  )


_DESCRIPTION = (
    "Maintain a short plan for a multi-step task (3+ steps): call this with the"
    " full ordered list of steps, each marked pending / in_progress /"
    " completed. Keep exactly one step in_progress, and update as you finish"
    " steps. It surfaces a live checklist to the user. Skip it for trivial one-"
    " or two-step questions."
)

PLAN_SPEC = ToolSpec(
    name="update_plan",
    display_name="core.update_plan",
    tier="core",
    description=_DESCRIPTION,
    request_schema=UpdatePlanRequest,
    handler=_handler,
    tags=("core", "meta"),
)
