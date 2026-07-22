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

"""The `visit` core tool — retrieve pages and extract grounded content.

For each URL: fetch (HTML via trafilatura, PDF via pdfplumber), then run a
reader-model extraction over the page text returning ``{accessible, summary,
evidence}``. Only accessible, non-empty pages are stored and added to the
citation index — paywalls / bot-walls / empty pages are reported but never
cited.

Fetch and the reader client are injectable so the tool is unit-testable offline.
The reader runs through the same provider seam using the paired reader profile
selected for the active backbone.
"""

from __future__ import annotations

import json
import re

from financeharness.providers import ModelProfile
from financeharness.runtime.config import RuntimeConfig, load_runtime_config
from financeharness.runtime.tool_events import emit_tool_event, emit_tool_progress
from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.research.cache import FetchCache
from financeharness.tools.research.visit_fetch import Fetcher, http_fetch
from financeharness.tools.research.visit_reader import read_page
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

__all__ = ["VisitRequest", "_coerce_urls", "build_visit_spec"]

_HEADLINE_CHARS = 140  # the rail wraps, but cap so one source can't dominate

_DESCRIPTION = (
    "Fetch one or more web pages and return faithful, grounded summaries with"
    " citation markers. Accessible pages are added to the bibliography; cite"
    " them inline as [N]. Use after `search` to read the most relevant results."
)


def _split_one(s):
  """Split one string into URLs: a JSON array, or a comma/whitespace-joined run."""
  s = s.strip()
  if s.startswith("["):
    try:
      parsed = json.loads(s)
      if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, TypeError):
      pass
  if "," in s or "\n" in s:
    urls = [
        p.strip()
        for p in re.split(r"[,\s]+", s)
        if p.strip().startswith("http")
    ]
    if urls:
      return urls
  return [s]


def _coerce_urls(v):
  """Repair the URL-list shapes models actually emit: a lone string, a

  JSON-stringified array, or a comma / whitespace-joined string (also inside a
  list element). In practice, ~89% of visit failures were malformed
  multi-URL args.
  """
  if isinstance(v, str):
    return _split_one(v)
  if isinstance(v, list):
    out: list[str] = []
    for item in v:
      out.extend(_split_one(item) if isinstance(item, str) else [item])
    return out
  return v


class VisitRequest(BaseModel):
  """Input for fetching and reading one or more URLs."""

  urls: list[str] = Field(
      Ellipsis, description="One or more page URLs to fetch and read."
  )
  goal: str | None = Field(
      None, description="Optional focus for the extraction."
  )

  @field_validator("urls", mode="before")
  @classmethod
  def _normalize(cls, v):
    return _coerce_urls(v)


def _headline(summary):
  """A one-line takeaway from the reader summary for the live sources rail —

  the first sentence if it's a sensible length, else a truncation.
  """
  s = " ".join(summary.split())  # collapse newlines/runs
  m = re.search(r"^(.+?[.!?])(\s|$)", s)
  head = m.group(1) if m and len(m.group(1)) <= _HEADLINE_CHARS else s
  return (
      head
      if len(head) <= _HEADLINE_CHARS
      else head[: _HEADLINE_CHARS - 1].rstrip() + "…"
  )


def _render(pages):
  lines: list[str] = []
  for p in pages:
    if p["accessible"]:
      lines.append(f"Source [{p['citation_index']}]: {p['url']}")
      lines.append(p["summary"])
      if p.get("evidence"):
        lines.append(f"> {p['evidence']}")
    elif p.get("reader_error"):
      # The reader itself failed (config/transport), not the page — surface it
      # distinctly so it isn't mistaken for a paywall (and so a misconfigured
      # reader is visible rather than looking like the whole web is blocked).
      lines.append(
          f"⚠ reader error on {p['url']}: {p['reader_error']} — the page was "
          "fetched but the reader could not process it (a tool/config issue, "
          "not a paywall)."
      )
    else:
      reason = p.get("error") or "not accessible (paywall / bot-wall / empty)"
      # Anti-fabrication: a failed read is not evidence. Steer the model to
      # an alternative source rather than asserting from the snippet.
      lines.append(
          f"✗ could not read {p['url']} — {reason}. It's unread, so rely on "
          "another source or note the gap."
      )
    lines.append("")
  return "\n".join(lines).strip()


def build_visit_spec(
    cache,
    profile,
    *,
    fetcher = None,
    client = None,
    config = None,
):
  """Build the per-run `visit` tool bound to the cache + reader backbone."""
  fetcher = fetcher or http_fetch
  cfg = config or load_runtime_config()

  async def handler(req):
    pages: list[dict] = []
    total = len(req.urls)
    for i, url in enumerate(req.urls):
      tag = f"({i + 1}/{total}) " if total > 1 else ""
      short = url.split("://")[-1][:42]
      if cache.has_content(url):
        text = cache.get_content(url) or ""
      else:
        emit_tool_progress(f"{tag}fetching {short}")
        page = await fetcher(url)
        if not page.ok:
          pages.append({
              "url": url,
              "accessible": False,
              "citation_index": None,
              "summary": "",
              "error": page.error,
          })
          continue
        text = page.text
        cache.set_content(url, text)

      emit_tool_progress(f"{tag}reading {short}")
      reader = await read_page(
          profile, text, goal=req.goal, client=client, config=cfg
      )
      if reader["accessible"] and reader["summary"]:
        idx = cache.add_citation(url, cache.get_title(url))
        # Surface the source live (numbered, with a headline) so the rail
        # can render it the moment a page is read, not only at the end.
        emit_tool_event(
            "source",
            {
                "index": idx,
                "url": url,
                "title": cache.get_title(url) or url,
                "headline": _headline(reader["summary"]),
            },
        )
        pages.append({
            "url": url,
            "accessible": True,
            "citation_index": idx,
            "summary": reader["summary"],
            "evidence": reader.get("evidence", ""),
        })
      else:
        pages.append({
            "url": url,
            "accessible": False,
            "citation_index": None,
            "summary": reader.get("summary", ""),
            "error": reader.get("error"),
            "reader_error": reader.get("reader_error"),
        })

    cited = sum(1 for p in pages if p["accessible"])
    return ToolResponse(
        markdown=_render(pages) or "No pages could be read.",
        structured={"pages": pages, "cited": cited},
        meta={"visited": len(pages), "cited": cited},
    )

  return ToolSpec(
      name="visit",
      display_name="visit",
      tier="core",
      description=_DESCRIPTION,
      request_schema=VisitRequest,
      handler=handler,
      tags=("web", "research"),
  )
