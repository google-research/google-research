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

"""Search backends — a pluggable interface + the ddgs default.

The deep-research `search` tool talks to a backend through this small protocol,
so SearXNG / Jina / a corpus index can be added later without touching the tool.
The core ships `DdgsBackend` (the `ddgs` library — zero infra) and `StubBackend`
(canned results for offline tests).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
  """One search hit returned by a backend."""

  title: str
  url: str
  snippet: str


class SearchBackend(Protocol):
  """Minimal async protocol implemented by search providers."""

  name: str

  async def search(self, query, max_results):
    Ellipsis


class DdgsBackend:
  """DuckDuckGo via the `ddgs` library.

  Synchronous under the hood, so the blocking call runs in a worker thread.
  """

  name = "ddgs"

  async def search(self, query, max_results):
    return await asyncio.to_thread(self._search_sync, query, max_results)

  @classmethod
  def _search_sync(cls, query, max_results):
    """Runs synchronous search using the DDGS library."""
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=max_results)
    out: list[SearchResult] = []
    for r in rows:
      url = r.get("href") or r.get("url") or ""
      if not url:
        continue
      out.append(
          SearchResult(
              title=(r.get("title") or "").strip(),
              url=url,
              snippet=(r.get("body") or "").strip(),
          )
      )
    return out


@dataclass
class StubBackend:
  """Canned results for offline tests.

  ``by_query`` maps a query to results; a query with no entry returns
  ``default``.
  """

  name: str = "stub"
  by_query: dict[str, list[SearchResult]] | None = None
  default: list[SearchResult] | None = None

  async def search(self, query, max_results):
    results = (self.by_query or {}).get(query, self.default or [])
    return results[:max_results]
