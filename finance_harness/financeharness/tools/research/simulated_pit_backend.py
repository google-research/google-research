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

"""Live-web stand-in for the FinanceGym PIT environment — **development only**.

When the PIT corpus bundle isn't available, this lets a benchmark run execute
end-to-end against the open web so the rest of the stack (provider wiring,
prompts, tool loop, report shape, judging pipeline) can be exercised. It pairs
the default `DdgsBackend` with a fetcher that reads each page's publication date
and refuses anything newer than the task's cutoff.
"""

from __future__ import annotations

import dataclasses
import json as _json

from financeharness.tools.research import search_backends

# _parse serializes every native (lxml) parse through one worker — running them
# in parallel threads corrupts the heap. Reuse it rather than re-deriving that.
from financeharness.tools.research import visit_fetch
import httpx
import trafilatura

_DEFAULT_TIMEOUT_S = 30.0
_MIN_CHARS = 200  # below this, the "page" is a bot-wall or a stub, not a source


def _extract_text_and_date(html):
  """(text, iso_date|None) from raw HTML.

  trafilatura's metadata pass resolves the publication date (via htmldate) in
  the same parse that extracts the body, so this costs one parse, not two.
  Falls back to a plain text extraction if the metadata path is unavailable, in
  which case the caller sees an undated page rather than an exception.

  Args:
    html: Raw HTML string to parse.

  Returns:
    A tuple of (text, iso_date|None).
  """
  try:
    raw = trafilatura.extract(html, output_format="json", with_metadata=True)
    if raw:
      obj = _json.loads(raw)
      date = (obj.get("date") or "").strip() or None
      return (obj.get("text") or "").strip(), date
  except (ValueError, KeyError, TypeError, AttributeError, RuntimeError):
    pass
  return (trafilatura.extract(html) or "").strip(), None


@dataclasses.dataclass
class SimulatedPITBackend:
  """DDGS search + a publication-date gate at fetch time.

  The date check lives in the *fetcher* on purpose: `search` pre-validates its
  candidates through the same callable, so post-cutoff hits are pruned from the
  result list before the model ever sees them — no separate filtering pass.

  Attributes:
    cutoff: ISO date string cutoff for point-in-time search compliance.
    inner: Underlying SearchBackend instance used for execution.
    allow_undated: Whether to allow pages whose publication date cannot be
      established (default False).
    timeout_s: Network request timeout in seconds.
    name: Backend identifier string.
    queries: Accumulated query strings issued by the agent.
    urls_fetched: Accumulated set of URLs fetched by the agent.
    dropped_post_cutoff: Count of candidates dropped due to post-cutoff date.
    dropped_undated: Count of candidates dropped due to unestablished date.
  """

  cutoff: str
  inner: search_backends.SearchBackend = dataclasses.field(
      default_factory=search_backends.DdgsBackend
  )
  allow_undated: bool = False
  timeout_s: float = _DEFAULT_TIMEOUT_S
  name: str = "web-simulated-pit"

  queries: list[str] = dataclasses.field(default_factory=list, init=False)
  urls_fetched: set[str] = dataclasses.field(default_factory=set, init=False)
  dropped_post_cutoff: int = dataclasses.field(default=0, init=False)
  dropped_undated: int = dataclasses.field(default=0, init=False)

  async def search(self, query, max_results):
    """Executes a search against the underlying SearchBackend.

    Args:
      query: The search query string.
      max_results: Maximum number of results to return.

    Returns:
      A list of search results.
    """
    self.queries.append(query)
    return await self.inner.search(query, max_results)

  async def fetch(self, url):
    """Fetches a URL and validates its publication date against the cutoff.

    Args:
      url: The URL to fetch.

    Returns:
      A FetchedPage instance.
    """
    try:
      async with httpx.AsyncClient(
          follow_redirects=True,
          timeout=self.timeout_s,
          headers=visit_fetch._FETCH_HEADERS,  # pylint: disable=protected-access
      ) as client:
        resp = await client.get(url)
      if resp.status_code >= 400:
        return visit_fetch.FetchedPage(
            url=url, text="", ok=False, error=f"HTTP {resp.status_code}"
        )
      text, date = await visit_fetch._parse(  # pylint: disable=protected-access
          _extract_text_and_date, resp.text
      )
    except (
        httpx.HTTPError,
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
      return visit_fetch.FetchedPage(
          url=url, text="", ok=False, error=f"{type(exc).__name__}: {exc}"
      )

    if len(text) < _MIN_CHARS:
      return visit_fetch.FetchedPage(
          url=url, text="", ok=False, error="no extractable content"
      )
    if date is None:
      if not self.allow_undated:
        self.dropped_undated += 1
        return visit_fetch.FetchedPage(
            url=url,
            text="",
            ok=False,
            error=(
                "publication date could not be established (point-in-time run)"
            ),
        )
    # ISO-8601 dates compare correctly as strings; both sides are YYYY-MM-DD.
    elif date[:10] > self.cutoff:
      self.dropped_post_cutoff += 1
      return visit_fetch.FetchedPage(
          url=url,
          text="",
          ok=False,
          error=f"published {date[:10]}, after the {self.cutoff} cutoff",
      )

    self.urls_fetched.add(url)
    return visit_fetch.FetchedPage(url=url, text=text, ok=True)

  def fetcher(self):
    """The :data:`Fetcher` callable to inject into `search` + `visit`.

    Returns:
      The fetch bound method.
    """
    return self.fetch
