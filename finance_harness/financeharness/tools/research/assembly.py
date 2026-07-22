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

"""Assembly — wire the research trio + cache into a registry and finalizer.

One place builds a fresh `FetchCache`, the three core tools bound to it, and the
citation finalizer the Agent applies at its single exit. Per request → no shared
state across runs.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from typing import Any

from financeharness.providers import ModelProfile
from financeharness.runtime.citations import validate_and_append_references
from financeharness.runtime.config import RuntimeConfig
from financeharness.runtime.skill_registry import (
    SkillRegistry,
    load_skill_registry,
    merge_skills,
)
from financeharness.runtime.tool_registry import ToolRegistry
from financeharness.tools.compute import COMPUTE_SPECS
from financeharness.tools.compute.arithmetic import SPEC as CALC_SPEC
from financeharness.tools.core.plan import PLAN_SPEC
from financeharness.tools.data.equity import EQUITY_DATA_SPECS
from financeharness.tools.data.market import MARKET_DATA_SPECS
from financeharness.tools.research.cache import FetchCache
from financeharness.tools.research.citations import build_citations_spec
from financeharness.tools.research.search import build_search_spec
from financeharness.tools.research.search_backends import SearchBackend
from financeharness.tools.research.visit import build_visit_spec
from financeharness.tools.research.visit_fetch import Fetcher, quick_fetch
from openai import AsyncOpenAI

_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"


def build_research_registry(
    cache,
    reader_profile,
    *,
    backend = None,
    fetcher = None,
    client = None,
    config = None,
):
  """A registry with search + visit + compose_citations bound to ``cache``.

  ``reader_profile`` is the model `visit` uses for page extraction — a
  *separate, cheaper/faster* model than the orchestrator (extraction is simple;
  a flagship reasoning model per page is slow + costly).
  """
  # search pre-flight validation uses the same fetcher as visit when injected
  # (tests), else a quick single-attempt fetch in production.
  registry = ToolRegistry()
  registry.register(
      build_search_spec(cache, backend, validator=fetcher or quick_fetch)
  )
  registry.register(
      build_visit_spec(
          cache, reader_profile, fetcher=fetcher, client=client, config=config
      )
  )
  registry.register(build_citations_spec(cache))
  return registry


def build_equity_research_registry(
    cache,
    reader_profile,
    *,
    backend = None,
    fetcher = None,
    client = None,
    config = None,
):
  """The research trio + the deferred equity data + valuation compute tools."""
  registry = build_research_registry(
      cache,
      reader_profile,
      backend=backend,
      fetcher=fetcher,
      client=client,
      config=config,
  )
  registry.register(
      CALC_SPEC
  )  # core: exact arithmetic so the model never computes in its head
  registry.register(
      PLAN_SPEC
  )  # core: a live research plan/checklist for multi-step work
  for spec in (*EQUITY_DATA_SPECS, *MARKET_DATA_SPECS, *COMPUTE_SPECS):
    registry.register(spec)
  return registry


def _project_skill_roots():
  """Where user/project skills are discovered, in increasing precedence: the

  project's ``./skills/`` (CWD), then any ``FH_SKILLS_DIR``
  (os.pathsep-separated).
  Later roots override built-ins by name — skills are extensible without code.
  """
  roots = [Path.cwd() / "skills"]
  env = os.environ.get("FH_SKILLS_DIR")
  if env:
    roots += [Path(p) for p in env.split(os.pathsep) if p.strip()]
  return roots


def default_skill_registry():
  """Bundled first-party skills + any drop-in project/user skills (./skills,

  FH_SKILLS_DIR), the latter overriding by name. A skill is a SKILL.md recipe
  that
  orchestrates existing tools — so user skills are safe by construction.
  """
  reg = load_skill_registry(_SKILLS_ROOT)  # built-in (strict)
  for root in _project_skill_roots():
    if (
        root.resolve() != _SKILLS_ROOT.resolve()
    ):  # don't re-scan the built-in root
      merge_skills(reg, root, strict=False)  # best-effort, override
  return reg


def citation_finalizer(
    cache,
):
  """An Agent ``finalize`` hook that appends the bibliography from ``cache``."""
  return lambda prediction: validate_and_append_references(
      prediction, cache.citations
  )
