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
from dataclasses import dataclass
import io


@dataclass
class FetchedPage:
  """Raw fetched page text or a fetch failure reason."""

  url: str
  text: str
  ok: bool
  error: str | None = None


Fetcher = Callable[[str], Awaitable[FetchedPage]]

_FETCH_TIMEOUT_S = 30.0

# Native HTML/PDF parsers (lxml via trafilatura, pdfplumber) are CPU-bound C
# extensions that corrupt the heap when run in parallel threads. Serialize every
# parse through one dedicated worker while network fetches stay concurrent.
_PARSE_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="parse"
)
_PARSE_TIMEOUT_S = 25.0

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
  """Run a native parser on the single parse worker."""
  loop = asyncio.get_running_loop()
  return await asyncio.wait_for(
      loop.run_in_executor(_PARSE_POOL, fn, *args), timeout=_PARSE_TIMEOUT_S
  )


def _extract_pdf(content):
  import pdfplumber

  parts: list[str] = []
  with pdfplumber.open(io.BytesIO(content)) as pdf:
    for page in pdf.pages:
      parts.append(page.extract_text() or "")
  return "\n".join(parts)


async def _fetch_once(url, timeout):
  """One GET -> extracted text; retryable failures carry a ``retryable:`` prefix."""
  import httpx
  import trafilatura

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
  except Exception as exc:  # noqa: BLE001 — surfaced as a failed page, never raised
    return FetchedPage(
        url=url,
        text="",
        ok=False,
        error=f"retryable:{type(exc).__name__}: {exc}",
    )


async def quick_fetch(url, *, timeout = 8.0):
  """Single-attempt fetch for quick search-result readability validation."""
  page = await _fetch_once(url, timeout)
  if not page.ok and (page.error or "").startswith("retryable:"):
    page = FetchedPage(
        url=url, text="", ok=False, error=page.error[len("retryable:") :]
    )
  return page


async def http_fetch(
    url, *, timeout = _FETCH_TIMEOUT_S
):
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
