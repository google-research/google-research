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

"""SSE event protocol for POST /research (stream=true).

A versioned `text/event-stream`. Frames are ``event: <type>\\ndata:
<json>\\n\\n``.

Event sequence (research stream):
  run_start    {version, question, equity, mode} once, first
  round_start  {round}                          per agent round
  reasoning    {text}                           thinking delta (streamed)
  tool_call    {name, call_id, args, round}     per tool invocation
  tool_progress {name, call_id, detail}         live sub-status of a running
  tool
  tool_result  {name, call_id, ok, elapsed_s, round, result?}  per tool result
                 (result: truncated data_/compute_ output for the trace; null
                 otherwise)
  source       {index, url, title, headline}    per page read by visit (live
  rail)
  plan         {items:[{step,status}]}          the live research plan
  (update_plan)
  token        {text}                           content delta (streamed report)
  phase        {label}                          a coarse processing phase (e.g.
  grounding)
  answer       {content}                        when the model writes the report
  error        {error}                          on a failed run
  done         {trajectory}                     once, last — the full trajectory
                                                 (same shape as the sync body)

Clarify (scoping) stream reuses tool_call/tool_result, then ends with:
  context        {…}                            the gathered context
  clarify_result {result}                       once, last — the scoping outcome

``EVENT_TYPES`` below is the single source of truth: ``sse_frame`` rejects any
event not in it, so a typo'd or undocumented frame fails fast instead of
shipping.

``reasoning``/``token`` frames stream the model's thinking and the report as it
writes them (interleaved with tool_call/tool_result so a client can reconstruct
the agent's timeline); the later ``answer`` frame carries the authoritative full
content (a client reconciles to it).

The terminal `done` carries the whole trajectory so a streaming client ends with
exactly what a sync client would have received.
"""

from __future__ import annotations

import json
from typing import Any

SSE_VERSION = "0.1"

EVENT_TYPES = frozenset({
    "run_start",
    "round_start",
    "reasoning",
    "tool_call",
    "tool_progress",
    "tool_result",
    "source",  # a page read by visit: {index, url, title, headline} — live sources rail
    "plan",  # the live research plan from update_plan: {items:[{step,status}]}
    "token",
    "phase",  # a coarse processing-phase label (e.g. "grounding") for live status
    "answer",
    "error",
    "done",
    # clarify (scoping) stream
    "context",
    "clarify_result",
})


def sse_frame(event, data):
  """Serialize one SSE frame.

  ``event`` must be a registered ``EVENT_TYPES`` member — an unknown type is a
  programming error (drift), so we raise rather than ship an undocumented frame
  a client can't handle.
  """
  if event not in EVENT_TYPES:
    raise ValueError(
        f"unknown SSE event type: {event!r} (add it to EVENT_TYPES)"
    )
  return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
