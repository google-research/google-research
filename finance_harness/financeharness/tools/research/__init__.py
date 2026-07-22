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

"""Deep-research tools: search (discovery), visit (retrieval), compose_citations.

Built per run via factories that share a :class:`FetchCache`.
"""

from financeharness.tools.research.assembly import (
    build_equity_research_registry,
    build_research_registry,
    citation_finalizer,
    default_skill_registry,
)
from financeharness.tools.research.cache import Citation, FetchCache
from financeharness.tools.research.citations import build_citations_spec
from financeharness.tools.research.search import build_search_spec
from financeharness.tools.research.search_backends import (
    DdgsBackend,
    SearchBackend,
    SearchResult,
    StubBackend,
)
from financeharness.tools.research.visit import build_visit_spec
from financeharness.tools.research.visit_fetch import FetchedPage

__all__ = [
    "Citation",
    "DdgsBackend",
    "FetchCache",
    "FetchedPage",
    "SearchBackend",
    "SearchResult",
    "StubBackend",
    "build_citations_spec",
    "build_equity_research_registry",
    "build_research_registry",
    "build_search_spec",
    "build_visit_spec",
    "citation_finalizer",
    "default_skill_registry",
]
