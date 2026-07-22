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

"""Tool-side progress events — a context-scoped channel so a running tool can

surface live sub-status (e.g. ``visit`` fetching/reading each URL) without
threading an emitter through every handler signature.

The dispatcher opens a :func:`tool_event_scope` around the handler; inside the
handler, :func:`emit_tool_progress` posts a ``tool_progress`` event for the
current call. A no-op when no scope is active (sync tests, non-streaming runs).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from contextvars import ContextVar
from typing import Any

ToolEmit = Callable[[str, dict[str, Any]], None]

_ctx: ContextVar[tuple[ToolEmit, str, str | None] | None] = ContextVar(
    "fh_tool_emit", default=None
)


@contextlib.contextmanager
def tool_event_scope(
    emit, name, call_id
):
  """Bind the active tool's emitter for the duration of a handler call."""
  if emit is None:
    yield
    return
  token = _ctx.set((emit, name, call_id))
  try:
    yield
  finally:
    _ctx.reset(token)


def emit_tool_progress(detail):
  """From inside a tool handler: post a live sub-status for the current call."""
  ctx = _ctx.get()
  if ctx is None:
    return
  emit, name, call_id = ctx
  emit("tool_progress", {"name": name, "call_id": call_id, "detail": detail})


def emit_tool_event(kind, data):
  """From inside a tool handler: post a custom event (e.g.

  ``visit`` emitting a ``source`` per page read) tagged with the current tool
  name + call id. No-op when no scope is active.
  """
  ctx = _ctx.get()
  if ctx is None:
    return
  emit, name, call_id = ctx
  emit(kind, {"name": name, "call_id": call_id, **data})
