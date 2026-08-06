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

"""Fetch/parsing helpers for the `visit` tool.

This module owns network retrieval and native HTML/PDF text extraction. The
tool-facing module keeps orchestration and rendering concerns separate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import concurrent.futures
import dataclasses
import io
import multiprocessing
import os

import httpx
import pdfplumber
import trafilatura


@dataclasses.dataclass
class FetchedPage:
  """Raw fetched page text or a fetch failure reason."""

  url: str
  text: str
  ok: bool
  error: str | None = None


Fetcher = Callable[[str], Awaitable[FetchedPage]]

_FETCH_TIMEOUT_S = 30.0

# Native HTML/PDF parsers (lxml via trafilatura, pdfplumber) are CPU-bound and
# corrupt the heap when run in parallel *threads*, so this was one thread for a
# long time. That is not survivable: on 2026-08-04 a single page sent
# trafilatura's `sanitize_tree` into an unbounded computation, the 25s
# `wait_for` abandoned the future but could not stop the thread, and because
# `sanitize_tree` is Python it held the GIL. One document wedged the only parse
# worker, starved the event loop, and killed a 400-task run an hour in — the
# process then burned a core for 6.5 hours with zero output.
#
# Parsing therefore runs in *processes*: separate interpreters share no heap
# (so the thread hazard cannot occur), a wedged child cannot hold this process's
# GIL, and — unlike a thread — it can actually be killed. See _recycle_pool.
#
# spawn, not fork: forking a process that already has asyncio + threads running
# is a deadlock hazard. spawn re-imports in the child, which is why every
# function submitted here must be module-level (all three call sites are) and
# why the entry points need their `if __name__ == "__main__"` guards (they have
# them) — otherwise each worker would re-execute the benchmark.
#
# FH_PARSE_WORKERS=0 restores the legacy single-thread pool. It is an escape
# hatch, not a supported mode: it reinstates the failure above.
_PARSE_WORKERS = int(os.environ.get("FH_PARSE_WORKERS", "4"))
_PARSE_TIMEOUT_S = 25.0
# A parse this large is either a pathological document or one whose tail we do
# not need; both are better truncated than allowed to wedge a worker. Bounding
# the input is a mitigation, not a guarantee — `sanitize_tree` blows up on
# document *structure*, which a size cap does not bound. The kill path below is
# what actually contains it.
_MAX_PARSE_CHARS = 4_000_000
_MAX_PARSE_BYTES = 32_000_000


def _new_pool():
  if _PARSE_WORKERS <= 0:
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="parse"
    )
  return concurrent.futures.ProcessPoolExecutor(
      max_workers=max(1, _PARSE_WORKERS),
      mp_context=multiprocessing.get_context("spawn"),
  )


_PARSE_POOL = _new_pool()
_POOL_LOCK = asyncio.Lock()


async def _recycle_pool(dead):
  """Replace the pool after a parse timed out, killing the wedged worker.

  A timed-out `run_in_executor` future is abandoned, not cancelled — the work
  keeps running. With processes we can end it for real. Rebuilding wholesale
  (rather than killing one child) keeps this simple and is cheap: it happens
  only on the rare pathological document, and a spawn pool starts its workers
  lazily on next submit.

  Args:
    dead: The executor pool to shut down and recycle.
  """
  global _PARSE_POOL
  async with _POOL_LOCK:
    if _PARSE_POOL is not dead:
      return  # another coroutine already recycled it
    _PARSE_POOL = _new_pool()
  # `_processes` is private, but there is no public way to reap a running
  # child; shutdown(wait=False) leaves it spinning, which is the whole bug.
  for proc in list(getattr(dead, "_processes", {}).values()):
    try:
      proc.kill()
    except (OSError, RuntimeError, ProcessLookupError, AttributeError):
      pass
  dead.shutdown(wait=False, cancel_futures=True)


def _bounded(arg):
  """Cap an oversized parser input; pass everything else through untouched."""
  if isinstance(arg, str) and len(arg) > _MAX_PARSE_CHARS:
    return arg[:_MAX_PARSE_CHARS]
  if isinstance(arg, (bytes, bytearray)) and len(arg) > _MAX_PARSE_BYTES:
    return bytes(arg[:_MAX_PARSE_BYTES])
  return arg


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_FETCH_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_RETRY_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})
_FETCH_ATTEMPTS = 3
_FETCH_BACKOFF_S = 1.5


async def _parse(fn, *args):
  """Run a native parser on the parse pool, bounded and recoverable.

  On timeout the wedged worker is killed and the pool replaced, so one
  pathological document costs a single page rather than the whole run. Raises
  TimeoutError, which callers already treat as a failed fetch.

  Args:
    fn: The parser function to execute.
    *args: Arguments to pass to the parser function.

  Returns:
    The result of the parser function.
  """
  loop = asyncio.get_running_loop()
  pool = _PARSE_POOL
  try:
    return await asyncio.wait_for(
        loop.run_in_executor(pool, fn, *(_bounded(a) for a in args)),
        timeout=_PARSE_TIMEOUT_S,
    )
  except asyncio.TimeoutError:
    await _recycle_pool(pool)
    raise


def _extract_pdf(content):
  parts: list[str] = []
  with pdfplumber.open(io.BytesIO(content)) as pdf:
    for page in pdf.pages:
      parts.append(page.extract_text() or "")
  return "\n".join(parts)


async def _fetch_once(url, timeout):
  """One GET -> extracted text; retryable failures carry a ``retryable:`` prefix."""
  try:
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout, headers=_FETCH_HEADERS
    ) as client:
      resp = await client.get(url)
    if resp.status_code in _RETRY_STATUS:
      return FetchedPage(
          url=url, text="", ok=False, error=f"retryable:HTTP {resp.status_code}"
      )
    ctype = resp.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
      text = await _parse(_extract_pdf, resp.content)
    else:
      text = await _parse(trafilatura.extract, resp.text)
    text = (text or "").strip()
    if not text:
      return FetchedPage(
          url=url, text="", ok=False, error="retryable:no extractable content"
      )
    return FetchedPage(url=url, text=text, ok=True)
  except (
      httpx.HTTPError,
      RuntimeError,
      ValueError,
      TypeError,
      OSError,
  ) as exc:
    return FetchedPage(
        url=url,
        text="",
        ok=False,
        error=f"retryable:{type(exc).__name__}: {exc}",
    )


async def quick_fetch(url, *, timeout=8.0):
  """Single-attempt fetch for quick search-result readability validation."""
  page = await _fetch_once(url, timeout)
  if not page.ok and (page.error or "").startswith("retryable:"):
    page = FetchedPage(
        url=url, text="", ok=False, error=page.error[len("retryable:") :]
    )
  return page


async def http_fetch(url, *, timeout=_FETCH_TIMEOUT_S):
  """Default fetcher: HTTP GET -> extracted text, retrying transient failures."""
  page = FetchedPage(url=url, text="", ok=False, error="no attempt")
  for attempt in range(_FETCH_ATTEMPTS):
    page = await _fetch_once(url, timeout)
    if page.ok or not (page.error or "").startswith("retryable:"):
      break
    if attempt < _FETCH_ATTEMPTS - 1:
      await asyncio.sleep(_FETCH_BACKOFF_S * (attempt + 1))
  if not page.ok and (page.error or "").startswith("retryable:"):
    page = FetchedPage(
        url=url, text="", ok=False, error=page.error[len("retryable:") :]
    )
  return page
