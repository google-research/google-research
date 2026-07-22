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

"""Model-call recovery policy — pure decisions, no I/O.

Decides what to do when a model call doesn't cleanly succeed:
  - truncation (``finish_reason == "length"``) → escalate ``max_tokens``
  - transient (429/5xx/timeout/connection) → exponential backoff + retry
  - context overflow (400 + overflow signature) → compact the prompt + retry
  - malformed generation (other 400) → regenerate (re-roll, temperature > 0)
  - fatal (401/403/404/bug) → stop, fail fast

Pure functions of their inputs (exception / budget / messages), so the policy
is table-testable without a live server. Numeric knobs come from config.
"""

from __future__ import annotations

import random
from typing import Any

from financeharness.runtime.retry import is_default_retryable, status_of

BACKOFF = "backoff"
COMPACT = "compact"
REGENERATE = "regenerate"
FATAL = "fatal"

_CONTEXT_OVERFLOW_SIGNATURES = (
    "maximum context length",
    "reduce the length",
    "longer than the maximum",
    "context length",
)

_ELISION_MARKER = "[elided to fit context window"

# Transient SDK exception class names that carry no HTTP status (the request
# never got a response) and aren't builtin Timeout/Connection subclasses.
_TRANSIENT_EXC_NAMES = frozenset({
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ConnectTimeout",
    "ReadTimeout",
    "ConnectError",
    "RemoteProtocolError",
})


def _is_transient_by_name(err):
  return any(t.__name__ in _TRANSIENT_EXC_NAMES for t in type(err).__mro__)


def _message_of(err):
  msg = getattr(err, "message", None)
  if not isinstance(msg, str) or not msg:
    msg = str(err)
  return msg


def is_context_overflow(err):
  """True when ``err`` is a prompt-too-long error (HTTP 400 + overflow text)."""
  if status_of(err) != 400:
    return False
  text = _message_of(err).lower()
  return any(sig in text for sig in _CONTEXT_OVERFLOW_SIGNATURES)


def classify_model_error(err):
  """Map an exception to COMPACT / REGENERATE / BACKOFF / FATAL.

  The 400 split is by kind, not by matching tool-parser strings: a fixed,
  schema-valid request only yields 400s for (a) context overflow → COMPACT or
  (b) a malformed generation the parser rejected → REGENERATE (clears on a
  re-roll). Anything else transient → BACKOFF; otherwise FATAL.
  """
  if is_context_overflow(err):
    return COMPACT
  if status_of(err) == 400:
    return REGENERATE
  if is_default_retryable(err) or _is_transient_by_name(err):
    return BACKOFF
  return FATAL


def escalate_max_tokens(
    current, ceiling, factor = 2
):
  """Next ``max_tokens`` after a truncation, or ``None`` if already at the

  ceiling (accept the best-effort partial).
  """
  if current >= ceiling:
    return None
  return min(current * max(factor, 2), ceiling)


def backoff_delay(
    attempt,
    *,
    base_delay_s,
    max_delay_s,
    jitter = True,
    rng = None,
):
  """Exponential backoff with proportional jitter, clamped to ``max_delay_s``.

  Deterministic when ``jitter=False``.
  """
  delay = min(base_delay_s * (2**attempt), max_delay_s)
  if jitter:
    r = (rng or random).uniform(0, delay * 0.25)
    delay = min(delay + r, max_delay_s)
  return delay


def compact_messages(
    messages,
    *,
    keep_recent = 4,
):
  """Shrink the prompt by eliding old tool-result bodies (keep the most recent

  ``keep_recent`` intact). Preserves role + tool_call_id linkage; the full
  output stays in the trajectory's tool_log. Returns ``(new_messages,
  n_elided)`` or ``None`` when nothing is left to compact (idempotent). Never
  mutates the input.
  """
  tool_idxs = [
      i
      for i, m in enumerate(messages)
      if m.get("role") == "tool" and isinstance(m.get("content"), str)
  ]
  protected = set(tool_idxs[-keep_recent:]) if keep_recent > 0 else set()
  candidates = [i for i in tool_idxs if i not in protected]

  new_messages = [dict(m) for m in messages]
  n_elided = 0
  for i in candidates:
    content = new_messages[i].get("content") or ""
    if _ELISION_MARKER in content or len(content) <= len(_ELISION_MARKER) + 64:
      continue
    new_messages[i]["content"] = (
        f"{_ELISION_MARKER} — {len(content)} chars omitted; the full tool "
        "output is preserved in the run trajectory's tool_log.]"
    )
    n_elided += 1

  if n_elided == 0:
    return None
  return new_messages, n_elided
