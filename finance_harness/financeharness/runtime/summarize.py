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

"""Conversation summarization — the `/compact` capability.

Frees context on a long multi-turn session by replacing the *older* history with
an
LLM-written structured snapshot, keeping the most recent turn(s) verbatim. An
optional caller instruction steers what to preserve (e.g. "keep every ticker
figure
and table"). Fail-open: on any error the history is returned unchanged.

Distinct from ``runtime.recovery.compact_messages`` — that elides old
tool-result
*bodies* automatically *within* a single run; this summarizes across the whole
*session*, on demand, preserving meaning rather than dropping it.
"""

from __future__ import annotations

from typing import Any

from financeharness.providers import ModelProfile, complete
from financeharness.runtime.config import RuntimeConfig, load_runtime_config
from openai import AsyncOpenAI

_KEEP_RECENT_TURNS = 1  # keep the latest user→answer turn verbatim
_RENDER_BUDGET = (
    1200  # chars per message shown to the summarizer (tool bodies are long)
)

SUMMARY_PROMPT = """\
You compact a finance research conversation so it fits the model's context window \
while losing nothing that matters. Write a faithful structured snapshot another \
analyst could continue from — not a vague paraphrase.

Cover, as applicable:
- The question(s) and the goal/intent.
- Entities in scope (tickers, companies, sector).
- Key figures established — preserve exact numbers, tickers, dates, and units \
verbatim; never round or invent.
- Tools used and their notable results.
- Conclusions reached, the current state, and any open next steps.

Return the snapshot as concise markdown. Report only what the conversation actually \
established; do not add new facts."""


def _user_indices(messages):
  return [i for i, m in enumerate(messages) if m.get("role") == "user"]


def _split(
    messages, keep_recent_turns
):
  """(older, recent_tail).

  The tail starts at the keep_recent_turns-th-from-last user message; older is
  everything before it. Empty older → nothing to compact.
  """
  users = _user_indices(messages)
  if len(users) <= keep_recent_turns:
    return [], messages
  cut = users[-keep_recent_turns]
  return messages[:cut], messages[cut:]


def _render(messages):
  lines: list[str] = []
  for m in messages:
    role = m.get("role", "?")
    content = (m.get("content") or "").strip()
    if not content and m.get("tool_calls"):
      calls = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
      content = f"(calls: {calls})"
    if content:
      lines.append(f"{role}: {content[:_RENDER_BUDGET]}")
  return "\n".join(lines)


async def summarize_history(
    messages,
    *,
    instruction = None,
    profile,
    client = None,
    config = None,
    keep_recent_turns = _KEEP_RECENT_TURNS,
):
  """Summarize the older portion of a session, keeping the recent tail verbatim.

  Returns ``(new_messages, stats)``. Fail-open: returns the input unchanged on a
  no-op (≤ keep_recent_turns turns) or any summarization error.
  """
  cfg = config or load_runtime_config()
  older, tail = _split(messages, keep_recent_turns)
  if not older:
    return messages, {
        "compacted": False,
        "reason": "nothing older than the recent turn",
    }

  try:
    # render inside the try too: a malformed stored tool_call must fail-open
    # (compacted=false), never raise out of /compact.
    extra = (
        f"\n\nCaller instruction: {instruction.strip()}" if instruction else ""
    )
    user = f"Conversation so far:\n\n{_render(older)}{extra}"
    resp = await complete(
        profile,
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=cfg.generation.max_tokens,
        client=client,
    )
    summary = (resp.choices[0].message.content or "").strip()
  except Exception:  # noqa: BLE001 — fail-open: never lose the session on a compaction error
    return messages, {"compacted": False, "reason": "summarization failed"}
  if not summary:
    return messages, {"compacted": False, "reason": "empty summary"}

  snapshot = {
      "role": "assistant",
      "content": f"[Earlier conversation — compacted summary]\n\n{summary}",
  }
  new = [snapshot, *tail]
  return new, {
      "compacted": True,
      "messages_before": len(messages),
      "messages_after": len(new),
      "kept_recent_turns": keep_recent_turns,
      "summary_chars": len(summary),
  }
