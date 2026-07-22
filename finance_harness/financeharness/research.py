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

"""One-shot research entry point — assemble the harness and run a trajectory.

The canonical "ask a question, get a cited report" helper, used by the CLI and
the QA harness alike. Web-only by default; ``equity=True`` adds the equity data
+
valuation tools and the bundled skills (two-tier disclosure).
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from financeharness.providers import ModelProfile, get_profile
from financeharness.runtime.agent import Agent
from financeharness.runtime.config import RuntimeConfig
from financeharness.runtime.skill_registry import SkillRegistry
from financeharness.tools.research import (
    FetchCache,
    SearchBackend,
    build_equity_research_registry,
    citation_finalizer,
    default_skill_registry,
)
from financeharness.tools.research.visit_fetch import Fetcher
from openai import AsyncOpenAI


async def run_research(
    question,
    *,
    profile = None,
    reader_profile = None,
    equity = False,
    skill_registry = None,
    history = None,
    clarifications = None,
    mode = None,
    backend = None,
    fetcher = None,
    client = None,
    config = None,
    on_event = None,
    stream_tokens = False,
    grounding_review = None,
):
  """Run a deep-research trajectory end-to-end and return it (with citations).

  ``profile`` is the orchestrator (default flagship); ``reader_profile`` is the
  cheaper/faster model `visit` uses. ``equity=True`` adds the equity data +
  valuation tools and the bundled skills. ``grounding_review`` overrides the
  self-grounding pass (default: on in research mode) — a testing/comparison
  seam.
  """
  from financeharness.runtime.modes import get_mode, resolve_mode

  # Mode selects the prompt variant only — the tool registry is the same full
  # surface for every mode (web visible + equity/valuation deferred-but-callable
  # + skills + load_tool). A consistent toolset means switching mode mid-session
  # never strands the model with history that references a now-missing tool.
  # resolve_mode is the single source shared with the service's run_start frame,
  # so the advertised mode always matches the variant that actually runs.
  variant = get_mode(resolve_mode(mode, equity=equity)).prompt_variant

  profile = profile or get_profile()
  # Pair the reader with the backbone (cloud backbones read with a cloud reader, so
  # a run never depends on the local reader being served). Explicit override wins.
  reader_profile = reader_profile or get_profile(
      profile.reader_profile or "vllm-reader"
  )
  cache = FetchCache()
  registry = build_equity_research_registry(
      cache,
      reader_profile,
      backend=backend,
      fetcher=fetcher,
      client=client,
      config=config,
  )
  skill_registry = skill_registry or default_skill_registry()

  agent = Agent(
      profile=profile,
      registry=registry,
      config=config,
      client=client,
      finalize=citation_finalizer(cache),
      skill_registry=skill_registry,
      stream_tokens=stream_tokens,
      # One backbone self-grounding pass over the draft — the model rereads its
      # report against the sources it read and grounds any claim it can't support
      # (tool figures it owns stay intact). Research mode only by default: it's a
      # web-research anti-fabrication pass, so auto/analytical skip the extra call.
      # An explicit override wins over the mode default.
      grounding_review=(variant == "research")
      if grounding_review is None
      else grounding_review,
      prompt_variant=variant,
  )
  from financeharness.clarify import format_clarifications

  composed = question + format_clarifications(clarifications)
  traj = await agent.run(composed, on_event=on_event, history=history)
  traj["citations"] = [
      {"index": c.index, "url": c.url, "title": c.title}
      for c in cache.citations
  ]
  return traj


def save_trajectory(traj, path):
  """Write a trajectory to JSON (parents created)."""
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(traj, indent=2, ensure_ascii=False))
  return p
