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

"""FinanceGym point-in-time search backend + fetcher.

Swaps the live-web pair (`DdgsBackend` + `http_fetch`) for the FinanceGym PIT
environment, so a benchmark run never touches the open internet. Both halves are
per-task: the backend is built with that task's `cutoff` and passes it as
`max_date` on **every** `/search` call, which is what the benchmark enforces as
point-in-time compliance.

The two services (see `FinanceGym/docs/api-contract.md`):

* ``POST {embed_url}``   — OpenAI-compatible embeddings; embeds the query
string.
* ``POST {search_url}/search`` — ``{query_embedding, k, max_date}`` → ranked
hits.
* ``POST {search_url}/fetch``  — ``{doc_id}`` → the full article text.

The harness addresses pages by **URL** while FinanceGym addresses them by
**doc_id**, so the backend keeps a per-task url→doc_id map that the fetcher
reads back. A URL the backend never returned is refused rather than fetched —
that single rule is what stops `visit` from silently falling through to the live
web and voiding PIT compliance.
"""

from __future__ import annotations

import dataclasses

from financeharness.tools.research.search_backends import SearchResult
from financeharness.tools.research.visit_fetch import FetchedPage
import httpx

_DEFAULT_TIMEOUT_S = 60.0
_SYNTHETIC_SCHEME = "financegym://"


@dataclasses.dataclass
class FinanceGymBackend:
  """PIT search over the FinanceGym corpus, pinned to one task's cutoff.

  Attributes:
    cutoff: ISO date string cutoff for point-in-time search compliance.
    search_url: Base URL for the FinanceGym search service endpoint.
    embed_url: Base URL for the OpenAI-compatible embedding service endpoint.
    embed_model: Optional embedding model name.
    timeout_s: Network request timeout in seconds.
    name: Backend identifier string.
    queries: Accumulated query strings issued by the agent.
    doc_ids_fetched: Accumulated set of document IDs fetched by the agent.
  """

  cutoff: str
  search_url: str = "http://127.0.0.1:8889"
  embed_url: str = "http://127.0.0.1:8888/v1/embeddings"
  embed_model: str | None = None
  timeout_s: float = _DEFAULT_TIMEOUT_S
  name: str = "financegym-pit"

  _doc_ids: dict[str, str] = dataclasses.field(default_factory=dict, init=False)
  queries: list[str] = dataclasses.field(default_factory=list, init=False)
  doc_ids_fetched: set[str] = dataclasses.field(default_factory=set, init=False)

  # --- retrieval ---------------------------------------------------------

  async def _embed(self, client, query):
    body: dict[str, str] = {"input": query}
    if self.embed_model:
      body["model"] = self.embed_model
    r = await client.post(self.embed_url, json=body)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

  async def search(self, query, max_results):
    """Embed the query, run a PIT-filtered search, and map hits to URLs."""
    self.queries.append(query)
    async with httpx.AsyncClient(timeout=self.timeout_s) as client:
      embedding = await self._embed(client, query)
      r = await client.post(
          f"{self.search_url}/search",
          json={
              "query_embedding": embedding,
              "k": max_results,
              # Mandatory: the server never returns documents published after
              # this date. Dropping it would void PIT compliance silently.
              "max_date": self.cutoff,
          },
      )
      r.raise_for_status()
      hits = r.json()["results"]

    out: list[SearchResult] = []
    for h in hits:
      doc_id = h.get("doc_id") or ""
      if not doc_id:
        continue
      url = h.get("url") or ""
      # Fall back to a synthetic handle when the corpus row has no URL, and when
      # two documents share one URL — the map must stay 1:1 or `visit` would
      # read the wrong article.
      if not url or self._doc_ids.get(url, doc_id) != doc_id:
        url = f"{_SYNTHETIC_SCHEME}{doc_id}"
      self._doc_ids[url] = doc_id
      domain = h.get("domain") or "source"
      pub_date = h.get("pub_date") or ""
      out.append(
          SearchResult(
              title=f"{domain} — {pub_date}".strip(" —"),
              url=url,
              snippet=(h.get("text_preview") or "").strip(),
          )
      )
    return out

  # --- fetch -------------------------------------------------------------

  async def fetch(self, url):
    """Fetch a document the backend previously returned. Never hits the web."""
    doc_id = self._doc_ids.get(url)
    if doc_id is None:
      return FetchedPage(
          url=url,
          text="",
          ok=False,
          error=(
              "outside the point-in-time corpus — only URLs returned by"
              " `search` can be read in this environment"
          ),
      )
    try:
      async with httpx.AsyncClient(timeout=self.timeout_s) as client:
        r = await client.post(
            f"{self.search_url}/fetch", json={"doc_id": doc_id}
        )
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
    except (
        httpx.HTTPError,
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
      return FetchedPage(
          url=url, text="", ok=False, error=f"{type(exc).__name__}: {exc}"
      )
    if not text:
      return FetchedPage(url=url, text="", ok=False, error="empty document")
    self.doc_ids_fetched.add(doc_id)
    return FetchedPage(url=url, text=text, ok=True)

  def fetcher(self):
    """The :data:`Fetcher` callable to inject into `search` + `visit`."""
    return self.fetch

  async def health(self):
    """Raise unless both services answer — fail fast before spending a run."""
    async with httpx.AsyncClient(timeout=self.timeout_s) as client:
      r = await client.get(f"{self.search_url}/health")
      r.raise_for_status()
      await self._embed(client, "health check")
    self.queries.clear()
