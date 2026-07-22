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

"""The `search` core tool — batched web discovery with pre-flight validation.

Runs one or more queries through a :class:`SearchBackend`, dedups by URL,
records
titles, and (when a validator is provided) **over-fetches candidates and keeps
only
the fetchable ones** — pre-fetching each with a quick GET so paywalls /
bot-walls /
dead links are dropped before the model ever sees them, and warming the cache so
the
later `visit` is an instant, already-validated hit. The model spends its
attention on
research, not on fighting false-positive results. Falls back to unvalidated
results
if validation yields too few (never starve the model).
"""

from __future__ import annotations

import asyncio

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.research.cache import FetchCache
from financeharness.tools.research.search_backends import DdgsBackend, SearchBackend
from financeharness.tools.research.visit_fetch import Fetcher
from pydantic import BaseModel, Field, field_validator

_DEFAULT_MAX_RESULTS = 8
_OVERFETCH_FACTOR = (
    2  # pull ~2× candidates so validation can drop the unfetchable
)
_MAX_VALIDATE = 16  # hard cap on pre-fetch validations per call (latency bound)
_VALIDATE_CONCURRENCY = 8
_MIN_CHARS = (
    200  # extracted-text floor that distinguishes real content from a wall
)

_DESCRIPTION = (
    "Search the web for one or more queries and return ranked, fetchable"
    " results (title, URL, snippet) to investigate. Results are pre-checked for"
    " readability, so `visit` the ones you pick to read and cite them."
)


class SearchRequest(BaseModel):
  """Input for a batched web search."""

  queries: list[str] = Field(
      Ellipsis, description="One or more search queries to run in this batch."
  )
  max_results: int | None = Field(
      None, description="Optional cap on results per query (default 8)."
  )

  @field_validator("queries", mode="before")
  @classmethod
  def _wrap_single(cls, v):
    # A lone string is a common model shape for a list field — accept it.
    return [v] if isinstance(v, str) else v


_VISIT_NUDGE = (
    "`visit` the entries most relevant to your question to read and cite them —"
    " grounding each figure you use in the page you actually read."
)


def _render(results):
  lines = [f"Found {len(results)} result(s):", ""]
  for i, r in enumerate(results, 1):
    lines.append(f"{i}. **{r['title'] or r['url']}** — {r['url']}")
    if r["snippet"]:
      lines.append(f"   {r['snippet']}")
  lines += ["", _VISIT_NUDGE]
  return "\n".join(lines)


async def _validate(
    candidates, validator, cache
):
  """Pre-fetch up to ``_MAX_VALIDATE`` candidates concurrently; keep the ones that

  return real content (warming the cache for an instant later visit).
  """
  to_check = candidates[:_MAX_VALIDATE]
  sem = asyncio.Semaphore(_VALIDATE_CONCURRENCY)

  async def check(c):
    async with sem:
      page = await validator(c["url"])
    if page.ok and len(page.text) >= _MIN_CHARS:
      cache.set_content(c["url"], page.text)  # warm → visit becomes a cache hit
      return c
    return None

  checked = await asyncio.gather(*(check(c) for c in to_check))
  return [c for c in checked if c is not None]


def build_search_spec(
    cache,
    backend = None,
    *,
    default_max_results = _DEFAULT_MAX_RESULTS,
    validator = None,
):
  """Build the per-run `search` tool bound to a cache + backend.

  When ``validator`` is given, results are pre-flight-validated (over-fetch →
  drop unfetchable → warm cache); without it, raw backend results are returned.
  """
  backend = backend or DdgsBackend()

  async def handler(req):
    per = req.max_results or default_max_results
    cap = per * len(req.queries)
    fetch_n = per * _OVERFETCH_FACTOR if validator else per
    seen: set[str] = set()
    candidates: list[dict] = []
    for q in req.queries:
      try:
        hits = await backend.search(q, fetch_n)
      except Exception:  # noqa: BLE001 — one query failing shouldn't sink the batch
        hits = []
      for h in hits:
        if h.url in seen:
          continue
        seen.add(h.url)
        cache.set_title(h.url, h.title)
        candidates.append(
            {"title": h.title, "url": h.url, "snippet": h.snippet, "query": q}
        )

    if not candidates:
      raise ToolError(
          "No search results — the upstream search may be unavailable or the "
          "queries too narrow. Try rephrasing or fewer, broader queries."
      )

    validated_meta: dict = {}
    if validator is not None:
      valid = await _validate(candidates, validator, cache)
      validated_meta = {
          "candidates": len(candidates),
          "validated": len(valid),
          "dropped": len(candidates[:_MAX_VALIDATE]) - len(valid),
      }
      # Fall back to unvalidated candidates rather than starve the model.
      results = (valid or candidates)[:cap]
    else:
      results = candidates[:cap]

    return ToolResponse(
        markdown=_render(results),
        structured={"results": results, "count": len(results)},
        meta={"backend": backend.name, **validated_meta},
    )

  return ToolSpec(
      name="search",
      display_name="search",
      tier="core",
      description=_DESCRIPTION,
      request_schema=SearchRequest,
      handler=handler,
      tags=("web", "research"),
  )
