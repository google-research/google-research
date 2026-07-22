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

"""Clarification scoping — the pre-research alignment pass.

Before a full deep-research run, `scope_question` does a light search and a
single backbone call to judge whether the question is specific enough. If not,
it returns 2–4 concrete clarifying questions (each with suggested options) so
the
client can align scope with the user, then weave the answers back into research.

Model-decided (skips clarification when the question is already specific) and
fail-open (any error → proceed without clarifying, never block research).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import datetime
import time
from typing import Any

from financeharness.providers import ModelProfile, complete, get_profile
from financeharness.runtime.config import RuntimeConfig, load_runtime_config
from financeharness.runtime.jsonutil import extract_json_obj
from financeharness.tools.research.search_backends import DdgsBackend, SearchBackend, SearchResult
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

EventCallback = Callable[[str, dict[str, Any]], None]

_MAX_QUESTIONS = 4
_SCOPE_MAX_TOKENS = 4096

SCOPE_PROMPT = """\
You scope a finance research question before a deep-research run. Given the \
question and a brief web snapshot, decide whether it is specific enough to \
research well as-is, or whether it is genuinely ambiguous in a way that would send \
the research in materially different directions.

Strongly prefer proceeding. Default to sufficient=true and record sensible \
defaults as assumptions — a capable analyst makes reasonable choices (the most \
recent reported period, the primary US-listed entity, USD, a balanced angle) \
rather than asking. A specific company with a clear intent is sufficient — do not \
ask about time horizon, currency, depth, or angle when a reasonable default applies; \
and when a dominant entity is clearly implied ("the chip leader" → NVIDIA), assume \
it and proceed.

The one case you MUST ask: when the question names no identifiable company, ticker, \
or sector to research at all — it refers to "the company", "it", "them", or "the \
stock" with no antecedent and the snapshot doesn't pin one down. There is no \
reasonable default for *which entity*, so ask which one. Otherwise, set \
sufficient=false only when distinct interpretations would genuinely produce \
different research.

Return ONLY a JSON object:
  "sufficient": true unless the question is genuinely ambiguous (per above).
  "assumptions": up to 3 short strings — the defaults you are proceeding with \
(e.g. "most recent reported quarter", "USD", "the US-listed entity").
  "questions": [] when sufficient; otherwise 2–4 objects, each {"id": short-slug, \
"question": the clarifying question, "options": 2–4 concrete answers with the \
most-likely first, "allow_free_text": true} — asking only what genuinely blocks \
the research.

Anchor every time reference to today's date (given below) and the web snapshot; \
current reality sets the dates. Prefer relative period labels ("most recent \
reported quarter", "trailing 12 months", "year to date") over a specific calendar \
quarter, naming a specific period only when the snapshot supports it."""


class ClarifyQuestion(BaseModel):
  """One scoping question the client can ask before research starts."""

  id: str
  question: str
  options: list[str] = Field(default_factory=list)
  allow_free_text: bool = True


class ClarifyResult(BaseModel):
  """Scoping outcome: proceed as-is or ask a small set of clarifying questions."""

  sufficient: bool = True
  questions: list[ClarifyQuestion] = Field(default_factory=list)
  assumptions: list[str] = Field(default_factory=list)


def _to_result(obj):
  """Validate the model's JSON into a ClarifyResult; fail-open on bad shape."""
  try:
    questions = []
    for q in (obj.get("questions") or [])[:_MAX_QUESTIONS]:
      if isinstance(q, dict) and q.get("question"):
        questions.append(
            ClarifyQuestion(
                id=str(q.get("id") or f"q{len(questions) + 1}"),
                question=str(q["question"]),
                options=[str(o) for o in q.get("options") or []][:4],
                allow_free_text=bool(q.get("allow_free_text", True)),
            )
        )
    sufficient = bool(obj.get("sufficient", not questions))
    # If the model said insufficient but gave no usable questions, proceed.
    if not questions:
      sufficient = True
    return ClarifyResult(
        sufficient=sufficient,
        questions=[] if sufficient else questions,
        assumptions=[str(a) for a in obj.get("assumptions") or []][:3],
    )
  except Exception:  # noqa: BLE001 — fail-open
    return ClarifyResult(sufficient=True)


async def _snapshot_hits(
    question, backend
):
  try:
    hits = await backend.search(question, 5)
  except Exception:  # noqa: BLE001 — scoping proceeds without grounding
    return []
  return [h for h in hits[:5] if h.title or h.snippet]


def _format_snapshot(hits):
  lines = [f"- {h.title}: {h.snippet[:160]}" for h in hits]
  return "\n".join(lines) or "(no search results)"


async def _scope_from_snapshot(
    question,
    snapshot,
    *,
    profile,
    client,
    cfg,
    today,
):
  """The model call shared by the sync + streaming paths. Fail-open."""
  user = (
      f"Today's date is {today}.\n\nQuestion: {question}\n\nWeb"
      f" snapshot:\n{snapshot}"
  )
  try:
    resp = await complete(
        profile,
        [
            {"role": "system", "content": SCOPE_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=min(_SCOPE_MAX_TOKENS, cfg.reader.max_tokens),
        client=client,
    )
    raw = resp.choices[0].message.content or ""
  except Exception:  # noqa: BLE001 — never block research on a scoping failure
    return ClarifyResult(sufficient=True)
  obj = extract_json_obj(raw)
  return _to_result(obj) if obj else ClarifyResult(sufficient=True)


async def scope_question(
    question,
    *,
    profile = None,
    backend = None,
    client = None,
    config = None,
    today = None,
):
  """Light-search-grounded scoping pass → ClarifyResult. Never raises."""
  # Scoping is a reasoning gate (judge specificity + draft 2–4 questions), so it runs
  # on the backbone the question uses — the small reader under-asks on subtly ambiguous
  # prompts. The service passes the active backbone; this default covers profile-less calls.
  profile = profile or get_profile()
  backend = backend or DdgsBackend()
  cfg = config or load_runtime_config()
  today = today or datetime.date.today().isoformat()
  hits = await _snapshot_hits(question, backend)
  return await _scope_from_snapshot(
      question,
      _format_snapshot(hits),
      profile=profile,
      client=client,
      cfg=cfg,
      today=today,
  )


async def scope_question_stream(
    question,
    *,
    on_event = None,
    profile = None,
    backend = None,
    client = None,
    config = None,
    today = None,
):
  """Streaming scoping: surface the context search as a transcript step + its

  hits (so the UI can show what's being read), then the scoping result. Same
  fail-open contract as :func:`scope_question`.
  """
  emit = on_event or (lambda _k, _d: None)
  profile = profile or get_profile()
  backend = backend or DdgsBackend()
  cfg = config or load_runtime_config()
  today = today or datetime.date.today().isoformat()

  emit("tool_call", {"name": "search", "args": {"query": question}, "round": 0})
  t0 = time.monotonic()
  hits = await _snapshot_hits(question, backend)
  emit(
      "tool_result",
      {
          "name": "search",
          "ok": True,
          "elapsed_s": round(time.monotonic() - t0, 2),
          "round": 0,
      },
  )
  emit("context", {"sources": [{"title": h.title, "url": h.url} for h in hits]})

  result = await _scope_from_snapshot(
      question,
      _format_snapshot(hits),
      profile=profile,
      client=client,
      cfg=cfg,
      today=today,
  )
  emit("clarify_result", {"result": result.model_dump()})
  return result


def format_clarifications(pairs):
  """Render answered clarifications as a block appended to the research question."""
  items = [
      f"- {p.get('question', '').strip()} → {p.get('answer', '').strip()}"
      for p in pairs or []
      if p.get("answer")
  ]
  if not items:
    return ""
  return "\n\nClarifications from the user:\n" + "\n".join(items)
