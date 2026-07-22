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

"""Retry helper for tool handlers + the shared transient-error taxonomy.

Wraps an async-callable factory; absorbs transient failures (timeouts, transport
blips, rate limits, transient 5xx) before they reach the LLM. Actionable errors
(bad ticker, schema mismatch, auth) bubble through immediately — the model is
good at recovering from those and bad at understanding "the harness retried".
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import random

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class RetryableError(Exception):
  """Explicit caller signal that an error is transient."""


def status_of(err: BaseException) -> int | None:
  """HTTP status from an SDK exception, across provider shapes: OpenAI/httpx on

  ``.status_code``, requests on ``.status``, google-genai/gRPC on ``.code``
  (where
  ``.status`` is the string name). First integer wins — a new backbone slots in
  by
  error shape here, never a per-provider branch elsewhere.
  """
  for attr in ("status_code", "status", "code"):
    value = getattr(err, attr, None)
    if isinstance(value, int):
      return value
  return None


def is_default_retryable(err: BaseException) -> bool:
  """Conservative classifier: retry only well-known transient signals."""
  if isinstance(err, RetryableError):
    return True
  if isinstance(err, asyncio.TimeoutError | TimeoutError | ConnectionError):
    return True
  status = status_of(err)
  return status is not None and status in _RETRYABLE_STATUS


async def with_retry[T](
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 10.0,
    jitter: bool = True,
    retryable: Callable[[BaseException], bool] = is_default_retryable,
) -> T:
  """Run an async coroutine with exponential backoff on retryable errors.

  ``coro_factory`` returns a fresh coroutine each call (coroutines are
  single-await). Re-raises on non-retryable errors or once attempts are spent.
  """
  last_error: BaseException | None = None
  for attempt in range(max_attempts):
    try:
      return await coro_factory()
    except BaseException as err:
      last_error = err
      if not retryable(err) or attempt == max_attempts - 1:
        raise
      delay = min(base_delay_s * (2**attempt), max_delay_s)
      if jitter:
        delay += random.uniform(0, delay * 0.25)
      await asyncio.sleep(delay)
  raise last_error  # type: ignore[misc]  # unreachable
