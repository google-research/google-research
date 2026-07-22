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

"""FetchCache — per-run shared state for the research tools.

`search` records discovered titles; `visit` records fetched page content and
appends citations; `compose_citations` reads the citation index. One cache per
agent loop, threaded into the tool factories — so the trio coordinate without
global state.

The citation index is the single source of the bibliography: a URL is cited
only when `visit` successfully retrieved an accessible page, in visit order,
deduped by URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
  """One visited source in the run-local bibliography."""

  index: int  # 1-based, the [N] marker
  url: str
  title: str


@dataclass
class FetchCache:
  """Run-local page-content, title, and citation cache shared by research tools."""

  _content: dict[str, str] = field(default_factory=dict)
  _titles: dict[str, str] = field(default_factory=dict)
  _citations: list[Citation] = field(default_factory=list)
  _cited_index: dict[str, int] = field(default_factory=dict)

  # titles (set by search + visit) -------------------------------------- #
  def set_title(self, url, title):
    if title:
      self._titles[url] = title

  def get_title(self, url):
    return self._titles.get(url)

  # content (set by visit; read on cache hit) --------------------------- #
  def set_content(self, url, text):
    self._content[url] = text

  def get_content(self, url):
    return self._content.get(url)

  def has_content(self, url):
    return url in self._content

  # citations (appended by visit; read by compose_citations) ------------ #
  def add_citation(self, url, title = None):
    """Add `url` to the bibliography (dedup by URL, visit order).

    Returns its 1-based marker index.
    """
    if url in self._cited_index:
      return self._cited_index[url]
    index = len(self._citations) + 1
    self._citations.append(
        Citation(
            index=index, url=url, title=title or self._titles.get(url) or url
        )
    )
    self._cited_index[url] = index
    return index

  @property
  def citations(self):
    return list(self._citations)
