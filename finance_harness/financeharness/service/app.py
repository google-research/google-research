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

"""FastAPI app — research/clarify/compact endpoints, sessions, health.

`POST /research` serves both the sync JSON path and the SSE stream
(stream=true).
`run_research` is module-level so tests can substitute a fake (headless endpoint
tests need no live LLM).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from financeharness.clarify import (  # noqa: F401 — patchable seams for tests
    scope_question,
    scope_question_stream,
)
from financeharness.providers import ModelProfile, available_backbones, client_for, get_profile
from financeharness.providers.profiles import default_profile_name
from financeharness.research import run_research  # noqa: F401 — patchable seam for tests
from financeharness.runtime.config import load_runtime_config
from financeharness.runtime.context_budget import estimate_tokens
from financeharness.runtime.modes import resolve_mode
from financeharness.runtime.summarize import summarize_history
from financeharness.service.events import SSE_VERSION, sse_frame
from financeharness.service.sessions import SessionStore, default_sessions_dir
import httpx
from pydantic import BaseModel, Field

_SENTINEL = object()
_SESSIONS = SessionStore(default_sessions_dir())

# SSE keepalive: tools (visit/search) and non-streaming model calls can run for
# tens of seconds with no events; without a periodic byte the client's stream can
# idle-timeout. A comment frame (ignored by parsers) keeps the connection alive.
_HEARTBEAT_S = 10.0
_PING_FRAME = ": ping\n\n"


async def _drain(queue):
  """Yield SSE frames from the queue, emitting a heartbeat during silent gaps,

  until the sentinel arrives.
  """
  while True:
    try:
      kind, data = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_S)
    except TimeoutError:
      yield _PING_FRAME
      continue
    if kind is _SENTINEL:
      break
    yield sse_frame(kind, data)


API_VERSION = "0.1"

app = FastAPI(title="FinanceHarness", version=API_VERSION)


class ResearchRequest(BaseModel):
  """Request body for one research turn, sync or streamed."""

  question: str = Field(Ellipsis, min_length=1, description="The research question.")
  equity: bool = Field(
      False, description="(Legacy) enable equity tools; prefer `mode`."
  )
  mode: str | None = Field(
      None,
      description=(
          "Execution mode loadout: auto | research | analytical (default auto)."
      ),
  )
  profile: str | None = Field(
      None, description="Backbone profile override (default: vllm)."
  )
  stream: bool = Field(
      False, description="Stream progress as Server-Sent Events."
  )
  session_id: str | None = Field(
      None,
      description=(
          "Thread this turn into a multi-turn session (omit for one-shot)."
      ),
  )
  clarifications: list[dict[str, Any]] | None = Field(
      None,
      description=(
          "Answered clarifying questions [{question, answer}] to weave into"
          " scope."
      ),
  )


class ClarifyRequest(BaseModel):
  """Request body for the pre-research scoping pass."""

  question: str = Field(Ellipsis, min_length=1, description="The question to scope.")
  profile: str | None = Field(None, description="Backbone profile override.")
  stream: bool = Field(
      False, description="Stream scoping progress as Server-Sent Events."
  )


class CompactRequest(BaseModel):
  """Request body for compacting a durable session history."""

  session_id: str = Field(
      Ellipsis, description="The session to compact (summarize older history)."
  )
  instruction: str | None = Field(
      None,
      description=(
          "Optional steer for the summary (e.g. 'keep every ticker figure')."
      ),
  )
  profile: str | None = Field(None, description="Backbone profile override.")


async def _probe(profile):
  """True if the profile's resolved endpoint answers an authenticated ``/models``.

  Sends the API key (a cloud ``/models`` 401s unauthenticated → false offline);
  a keyless local vLLM (key ``EMPTY``) is probed without auth.
  """
  base = str(client_for(profile).base_url).rstrip("/")
  key = profile.resolve_api_key()
  headers = {"Authorization": f"Bearer {key}"} if key and key != "EMPTY" else {}
  try:
    async with httpx.AsyncClient(timeout=3.0) as client:
      r = await client.get(base + "/models", headers=headers)
      return r.status_code == 200
  except Exception:  # noqa: BLE001 — unreachable endpoint → not ready
    return False


@app.get("/health")
async def health(profile = None):
  """Readiness of a backbone + its paired reader.

  ``profile`` selects which backbone to report (default: the configured default)
  — so the TUI's status reflects the *active* backbone after a ``/model``
  switch, not always the default.
  """
  backbone = get_profile(profile) if profile else get_profile()
  reader = get_profile(backbone.reader_profile or "vllm-reader")
  dr_ready = await _probe(backbone)
  reader_ready = await _probe(reader)
  return {
      "status": "ok" if (dr_ready and reader_ready) else "degraded",
      "version": API_VERSION,
      "backbone": {
          "profile": backbone.name,
          "model": backbone.model,
          "ready": dr_ready,
      },
      "reader": {
          "profile": reader.name,
          "model": reader.model,
          "ready": reader_ready,
      },
  }


async def _event_stream(
    req,
    profile,
    history,
):
  """Bridge the agent's on_event callback into an SSE frame stream.

  The run executes as a task; its (sync) on_event pushes events onto a queue
  that this generator drains. The agent's internal `done` (termination only) is
  dropped — the protocol's terminal `done` carries the full trajectory.
  """
  queue: asyncio.Queue = asyncio.Queue()

  def on_event(kind, data):
    if kind == "done":
      return  # superseded by the protocol's terminal done (with trajectory)
    queue.put_nowait((kind, data))

  async def runner():
    try:
      traj = await run_research(
          req.question,
          profile=profile,
          equity=req.equity,
          mode=req.mode,
          history=history,
          clarifications=req.clarifications,
          on_event=on_event,
          stream_tokens=True,
      )
      _SESSIONS.commit(req.session_id, traj)
      queue.put_nowait(
          ("done", {"trajectory": traj, "session_id": req.session_id})
      )
    except Exception as exc:  # noqa: BLE001 — surface as an SSE error + empty done
      queue.put_nowait(("error", {"error": f"{type(exc).__name__}: {exc}"}))
      queue.put_nowait(
          ("done", {"trajectory": None, "session_id": req.session_id})
      )
    finally:
      queue.put_nowait((_SENTINEL, None))

  task = asyncio.create_task(runner())
  yield sse_frame(
      "run_start",
      {
          "version": SSE_VERSION,
          "question": req.question,
          "equity": req.equity,
          "mode": resolve_mode(req.mode, equity=req.equity),
      },
  )
  try:
    async for frame in _drain(queue):
      yield frame
  finally:
    if not task.done():
      task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _clarify_stream(
    req, profile
):
  """Bridge the scoping pass's on_event callback into an SSE frame stream — the

  context search step + hits stream first, then a terminal `clarify_result`.
  """
  queue: asyncio.Queue = asyncio.Queue()

  def on_event(kind, data):
    if kind == "clarify_result":
      return  # superseded by the terminal frame (below)
    queue.put_nowait((kind, data))

  async def runner():
    try:
      result = await scope_question_stream(
          req.question, profile=profile, on_event=on_event
      )
      queue.put_nowait(("clarify_result", {"result": result.model_dump()}))
    except Exception as exc:  # noqa: BLE001 — fail-open: emit a sufficient result
      queue.put_nowait(("error", {"error": f"{type(exc).__name__}: {exc}"}))
      queue.put_nowait(("clarify_result", {"result": {"sufficient": True}}))
    finally:
      queue.put_nowait((_SENTINEL, None))

  task = asyncio.create_task(runner())
  try:
    async for frame in _drain(queue):
      yield frame
  finally:
    if not task.done():
      task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@app.post("/clarify")
async def clarify(req):
  """Scope a question: decide if it's specific enough, else return clarifying

  questions. Sync JSON by default; SSE when stream=true. Fail-open. Scoping uses
  the
  fast local reader (a single lightweight call; fail-open if it isn't served).
  """
  profile = get_profile(req.profile) if req.profile else None
  if req.stream:
    return StreamingResponse(
        _clarify_stream(req, profile), media_type="text/event-stream"
    )
  result = await scope_question(req.question, profile=profile)
  return result.model_dump()


@app.post("/compact")
async def compact(req):
  """Compact a multi-turn session: LLM-summarize the older history (keeping the

  recent turn verbatim) and swap it in, freeing context. Optional `instruction`
  steers what to preserve. Fail-open (returns compacted=false on a no-op/error).
  """
  history = _SESSIONS.history(req.session_id)
  if not history:
    return {"compacted": False, "reason": "unknown or empty session"}
  profile = get_profile(req.profile) if req.profile else get_profile()
  new_history, stats = await summarize_history(
      history, instruction=req.instruction, profile=profile
  )
  if stats.get("compacted"):
    _SESSIONS.replace(req.session_id, new_history)
  return stats


@app.get("/sessions")
async def sessions():
  """List saved sessions (most-recently-updated first) for a pick-from-a-list UX.

  Summaries only — id, title, timestamps, turn count — no message bodies.
  """
  return {"sessions": _SESSIONS.list()}


@app.get("/sessions/{session_id}")
async def session_detail(session_id):
  """A session's stored conversation (system stripped) — lets the TUI rehydrate

  the transcript when resuming. Empty messages for an unknown session.
  """
  return {"id": session_id, "messages": _SESSIONS.history(session_id) or []}


@app.get("/models")
async def models():
  """Switchable backbone profiles + credential availability — powers the `/model`

  command. The default ships as a local OSS backbone; other configured profiles
  appear as available once the user configures their access.
  """
  return {"models": available_backbones(), "default": default_profile_name()}


@app.get("/status")
async def status(
    session_id = None, profile = None
):
  """Context usage for a session (token count vs window) — powers the /status command

  and the sidebar meter. ``used_pct`` is of the configured context window; the
  model
  label reflects the active backbone (``profile``), defaulting to the configured
  one.
  """
  cfg = load_runtime_config()
  backbone = get_profile(profile) if profile else get_profile()
  window = cfg.recovery.context_window
  used = estimate_tokens(_SESSIONS.history(session_id))
  return {
      "model": backbone.model,
      "context_window": window,
      "tokens_used": used,
      "used_pct": round(100 * used / window, 1) if window else 0.0,
      "turns": next(
          (s["turns"] for s in _SESSIONS.list() if s["id"] == session_id), 0
      ),
  }


@app.post("/research")
async def research(req):
  """Run one research trajectory. Sync JSON by default; SSE when stream=true.

  Pass ``session_id`` to thread a session; ``clarifications`` to weave scope
  answers.
  """
  profile = get_profile(req.profile) if req.profile else None
  history = _SESSIONS.history(req.session_id)
  if req.stream:
    return StreamingResponse(
        _event_stream(req, profile, history), media_type="text/event-stream"
    )
  traj = await run_research(
      req.question,
      profile=profile,
      equity=req.equity,
      mode=req.mode,
      history=history,
      clarifications=req.clarifications,
  )
  _SESSIONS.commit(req.session_id, traj)
  if req.session_id:
    traj = {**traj, "session_id": req.session_id}
  return traj
