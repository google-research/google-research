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

"""The agent loop — the orchestrator.

One bounded loop: call the model with the visible tool schemas → if it emits
tool_calls, dispatch each and append the results → repeat until the model stops
(``finish_reason == "stop"``) or a backstop (round / wall-clock cap) trips.
Every exit funnels through :meth:`Agent._build_result` → one trajectory dict.

Native tool calling: the provider's tool-call parser owns the wire format; we
never parse tool-call syntax ourselves. Recovery and reference chaining layer
onto the marked seams without changing this shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import time
from typing import Any

from financeharness.providers import ModelProfile, provider_for
from financeharness.providers.base import AssistantTurn
from financeharness.runtime import recovery
from financeharness.runtime.config import RuntimeConfig, load_runtime_config
from financeharness.runtime.context_budget import over_budget
from financeharness.runtime.dispatch import DispatchResult, dispatch_json_args
from financeharness.runtime.prompts import (
    GROUNDING_REVIEW_PROMPT,
    PROMPT_VARIANTS,
    build_system_prompt,
)
from financeharness.runtime.skill_registry import SkillRegistry, SkillSessionState
from financeharness.runtime.tool_registry import ToolRegistry, ToolSessionState
from openai import AsyncOpenAI

EventCallback = Callable[[str, dict[str, Any]], None]


def _tool_args(raw):
  """Parse a tool call's JSON argument string into a dict for display in the

  transcript. Best-effort: malformed / non-object args surface as ``{}``.
  """
  try:
    data = json.loads(raw) if raw else {}
  except (ValueError, TypeError):
    return {}
  return data if isinstance(data, dict) else {}


_UI_RESULT_BUDGET = (
    600  # chars of a data/compute result surfaced to the UI trace
)
_CALL_ID_FOOTER = (  # the chaining hint appended for the model — not for the UI
    "\n\n_call_id:"
)


def _ui_tool_result(name, result):
  """The grounding data behind a tool call, surfaced to the UI trace so a reader can

  audit the numbers a report rests on. Only the data/compute tools (the fetched
  or
  computed figures that are otherwise invisible) — web evidence already lives in
  the
  Sources rail, and ``calc``/orchestration carry nothing to audit. The chaining
  footer is dropped and the body truncated; ``None`` when there's nothing to
  show.
  """
  if not result.ok or not name.startswith(("data_", "compute_")):
    return None
  md = result.markdown.split(_CALL_ID_FOOTER, 1)[0].strip()
  if not md:
    return None
  if len(md) <= _UI_RESULT_BUDGET:
    return md
  tail = (
      _UI_RESULT_BUDGET // 3
  )  # head + tail so end-of-output figures (implied/range) survive
  return md[: _UI_RESULT_BUDGET - tail].rstrip() + "\n…\n" + md[-tail:].lstrip()


class Agent:
  """A finance deep-research orchestrator over a model profile + tool registry."""

  def __init__(
      self,
      *,
      profile,
      registry,
      config = None,
      client = None,
      system_prompt = None,
      finalize = None,
      skill_registry = None,
      stream_tokens = False,
      grounding_review = False,
      prompt_variant = "auto",
  ):
    self.profile = profile
    self.registry = registry
    self.config = config or load_runtime_config()
    self.client = client
    self.provider = provider_for(profile, client)
    self._system_prompt_override = system_prompt
    self.skill_registry = skill_registry
    # When True, model calls stream and their content deltas are surfaced as
    # ``token`` events (the report fills live). Off by default so the sync /
    # eval paths issue one non-streaming call as before.
    self.stream_tokens = stream_tokens
    # Optional post-processor applied to the final prediction at the single
    # exit (research wiring passes citation post-processing). Keeps the loop
    # itself citation-agnostic.
    self._finalize = finalize
    # When True, after the draft the backbone makes one principle-based pass over
    # its own report — re-reading it against the sources already in context and
    # revising any specific claim it can't ground. The model owns its numbers, so
    # tool-derived figures stay intact (no machinery, no string surgery).
    self._grounding_review = grounding_review
    # Per-mode system-prompt variant (auto/research/analytical).
    self._prompt_base = PROMPT_VARIANTS.get(
        prompt_variant, PROMPT_VARIANTS["auto"]
    )

  async def _issue(
      self,
      work,
      tools,
      max_tokens,
      emit,
  ):
    """Issue a single model call through the provider — streaming (content /

    reasoning deltas → ``token`` / ``reasoning`` events) when enabled, else one
    non-streaming call. Both return the same :class:`AssistantTurn`, so recovery
    treats them identically.
    """
    on_delta: Callable[[str, str], None] | None = None
    if self.stream_tokens:

      def on_delta(kind, text):
        if not text:
          return
        if kind == "content":
          emit("token", {"text": text})
        elif kind == "reasoning":
          emit("reasoning", {"text": text})

    return await self.provider.generate(
        work, tools=tools, max_tokens=max_tokens, on_delta=on_delta
    )

  async def _call_model(
      self,
      messages,
      tools,
      emit,
  ):
    """One model call wrapped in the recovery policy: proactive compaction,

    then backoff / regenerate / compact / fatal on error, and max_tokens
    escalation on truncation. Returns a completion, or ``None`` when
    recovery is exhausted (the loop surfaces a clean termination).
    """
    cfg = self.config
    max_tokens = cfg.generation.max_tokens
    work = messages
    transient = regen = escalations = 0

    while True:
      if cfg.recovery.proactive_compaction and over_budget(
          work,
          context_window=cfg.recovery.context_window,
          max_tokens=max_tokens,
          buffer_tokens=cfg.recovery.compaction_buffer_tokens,
      ):
        compacted = recovery.compact_messages(
            work, keep_recent=cfg.recovery.compaction_keep_recent_tool_results
        )
        if compacted is not None:
          work = compacted[0]

      try:
        response = await asyncio.wait_for(
            self._issue(work, tools or None, max_tokens, emit),
            cfg.limits.per_call_timeout_s,  # a hung model call must not block forever
        )
      except Exception as err:  # noqa: BLE001 — classify then act (TimeoutError → BACKOFF)
        action = recovery.classify_model_error(err)
        if action == recovery.BACKOFF:
          if transient >= cfg.limits.max_transient_retries:
            return None
          await asyncio.sleep(
              recovery.backoff_delay(
                  transient,
                  base_delay_s=cfg.backoff.base_delay_s,
                  max_delay_s=cfg.backoff.max_delay_s,
                  jitter=cfg.backoff.jitter,
              )
          )
          transient += 1
          continue
        if action == recovery.REGENERATE:
          if regen >= cfg.limits.max_regenerate_retries:
            return None
          regen += 1
          continue
        if action == recovery.COMPACT:
          if not cfg.recovery.compaction_enabled:
            return None
          compacted = recovery.compact_messages(
              work, keep_recent=cfg.recovery.compaction_keep_recent_tool_results
          )
          if compacted is None:
            return None
          work = compacted[0]
          continue
        return None  # FATAL

      if response.finish_reason == "length":
        nxt = recovery.escalate_max_tokens(
            max_tokens,
            cfg.generation.max_tokens_ceiling,
            cfg.generation.max_tokens_escalation_factor,
        )
        if (
            nxt is not None
            and escalations < cfg.limits.max_truncation_escalations
        ):
          max_tokens = nxt
          escalations += 1
          continue
      return response

  async def _redecode(
      self, messages, work
  ):
    """Re-decode ``work`` in one no-tools backbone pass and, on a non-empty

    result, commit a clean transcript — the prior turns plus the rewrite, with
    the scaffolding dropped (``messages`` is mutated in place). Best-effort: any
    failure returns None and the caller keeps the draft. The single re-decode
    primitive behind the grounding-review pass.

    Non-streaming on purpose: the authoritative `answer`/`done` payload carries
    the rewrite, so we don't re-stream the body.
    """
    try:
      response = await self.provider.generate(
          work, tools=None, max_tokens=self.config.generation.max_tokens
      )
    except Exception:  # noqa: BLE001 — best-effort; keep the draft
      return None
    new = response.content.strip()
    if not new:
      return None
    messages[:] = messages[:-1] + [{"role": "assistant", "content": new}]
    return new

  async def _maybe_review_grounding(
      self, messages, emit
  ):
    """One backbone pass to ground the draft against the sources already in the

    conversation. The model re-reads its own report and revises any specific
    claim it can't tie to something it actually read — attributing or softening
    it. Because the same model that wrote the figures reviews them, tool-derived
    numbers it understands stay intact. Best-effort: any failure keeps the
    draft.
    """
    emit("phase", {"label": "grounding"})
    work = messages + [{"role": "user", "content": GROUNDING_REVIEW_PROMPT}]
    return await self._redecode(messages, work)

  def _build_run_registry(
      self, session_state, skill_state
  ):
    """Per-run registry: a clone of the base + the meta-tools (load_tool when

    there are deferred tools, load_skill when there are skills) bound to this
    run's state. Never mutates the shared base registry.
    """
    # Local import keeps the meta-tools (which appear as registry ToolSpecs)
    # out of the runtime package's import surface.
    from financeharness.tools.core import build_load_skill_spec, build_load_tool_spec

    registry = self.registry.clone()
    if registry.deferred_tools():
      registry.register(build_load_tool_spec(session_state, registry))
    if self.skill_registry and self.skill_registry.all():
      registry.register(
          build_load_skill_spec(
              skill_state, session_state, self.skill_registry, registry
          )
      )
    return registry

  async def run(
      self,
      question,
      *,
      on_event = None,
      history = None,
  ):
    """Run the loop to completion and return the trajectory dict.

    ``history`` is prior-turn messages (system already stripped) threaded
    before the new question — for multi-turn sessions.
    """
    emit = on_event or (lambda _k, _d: None)
    session_state = ToolSessionState()
    skill_state = SkillSessionState()
    registry = self._build_run_registry(session_state, skill_state)
    system_prompt = self._system_prompt_override or build_system_prompt(
        registry, skill_registry=self.skill_registry, base=self._prompt_base
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *(history or []),
        {"role": "user", "content": question},
    ]
    tool_log: list[dict[str, Any]] = []

    limits = self.config.limits
    t0 = time.monotonic()
    rounds = 0
    termination = "error"
    prediction = ""
    grounding_reviewed = False

    while True:
      if rounds >= limits.max_rounds:
        termination = "max_rounds"
        break
      if time.monotonic() - t0 > limits.run_wall_clock_s:
        termination = "timeout"
        break
      rounds += 1
      emit("round_start", {"round": rounds})

      try:
        response = await self._call_model(
            messages, session_state.visible_schemas(registry), emit
        )
      except Exception as exc:  # noqa: BLE001 — backstop; recovery normally returns None
        termination = "error"
        emit("error", {"error": f"{type(exc).__name__}: {exc}"})
        break
      if response is None:
        termination = "error"
        emit("error", {"error": "model call failed after recovery"})
        break

      messages.append(response.to_message())

      if response.tool_calls:
        for tc in response.tool_calls:
          emit(
              "tool_call",
              {
                  "name": tc.name,
                  "call_id": tc.id,
                  "args": _tool_args(tc.arguments),
                  "round": rounds,
              },
          )
          t_tool = time.monotonic()
          result = await dispatch_json_args(
              tc.name,
              tc.arguments,
              registry=registry,
              session_state=session_state,
              call_id=tc.id,
              emit=emit,
          )
          messages.append({
              "role": "tool",
              "tool_call_id": tc.id,
              "content": result.for_model(),
          })
          tool_log.append(result.to_log())
          emit(
              "tool_result",
              {
                  "name": tc.name,
                  "call_id": tc.id,
                  "ok": result.ok,
                  "elapsed_s": round(time.monotonic() - t_tool, 2),
                  "round": rounds,
                  "result": _ui_tool_result(tc.name, result),
              },
          )
        continue

      # No tool calls → the model is answering.
      prediction = response.content
      termination = (
          "answer"
          if response.finish_reason == "stop"
          else (response.finish_reason or "error")
      )
      # Ground only a report that did research — a no-tool answer (a meta/"what
      # can you do" question, or a pure-knowledge reply) has nothing to ground
      # against, and the review would clobber it.
      if (
          termination == "answer"
          and self._grounding_review
          and prediction.strip()
          and tool_log
      ):
        reviewed = await self._maybe_review_grounding(messages, emit)
        if reviewed is not None:
          grounding_reviewed = reviewed != prediction
          prediction = reviewed
      emit("answer", {"content": prediction})
      break

    return self._build_result(
        question=question,
        prediction=prediction,
        termination=termination,
        rounds=rounds,
        messages=messages,
        tool_log=tool_log,
        elapsed_s=round(time.monotonic() - t0, 3),
        grounding_reviewed=grounding_reviewed,
        emit=emit,
    )

  def _build_result(
      self,
      *,
      question,
      prediction,
      termination,
      rounds,
      messages,
      tool_log,
      elapsed_s,
      grounding_reviewed,
      emit,
  ):
    """The single exit point — assemble the trajectory, applying the

    finalize hook (citation post-processing) when present.
    """
    citation_stats: dict[str, Any] = {}
    if self._finalize is not None and prediction:
      prediction, citation_stats = self._finalize(prediction)
    emit("done", {"termination": termination})
    return {
        "question": question,
        "prediction": prediction,
        "termination": termination,
        "rounds": rounds,
        "messages": messages,
        "tool_log": tool_log,
        "citation_stats": citation_stats,
        "grounding_reviewed": grounding_reviewed,
        "elapsed_s": elapsed_s,
        "model": self.profile.model,
    }
