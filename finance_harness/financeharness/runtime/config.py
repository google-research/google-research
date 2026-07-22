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

"""Runtime configuration — one source of truth for the generation budget, run

limits, error-recovery policy, and reader params.

Precedence (low → high): built-in defaults < file (``configs/runtime.json``) <
explicit ``overrides`` (test seam) < environment variables. The loader never
raises on a missing/malformed file — the harness always has a usable config.
Resource limits are anti-runaway *backstops* sized generous; the bounded round
count is what guarantees termination.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import contextlib
import copy
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "runtime.json"
)


class _Section(BaseModel):
  # Tolerate unknown keys in the file rather than crashing the loader.
  model_config = ConfigDict(extra="ignore")


class GenerationConfig(_Section):
  """Main model generation token budget and truncation escalation settings."""

  max_tokens: int = 16384
  max_tokens_ceiling: int = 65536
  max_tokens_escalation_factor: int = 2


class LimitsConfig(_Section):
  """Run, call, round, and retry caps that prevent runaway execution."""

  per_call_timeout_s: int = 1800
  run_wall_clock_s: int = 3600
  max_rounds: int = 100
  max_transient_retries: int = 10
  max_truncation_escalations: int = 3
  max_regenerate_retries: int = 3


class BackoffConfig(_Section):
  """Transient-error backoff parameters."""

  base_delay_s: float = 1.0
  max_delay_s: float = 30.0
  jitter: bool = True


class RecoveryConfig(_Section):
  """Context-window and compaction controls used by model-call recovery."""

  compaction_enabled: bool = True
  compaction_keep_recent_tool_results: int = 4
  context_window: int = 131072
  compaction_buffer_tokens: int = 4096
  proactive_compaction: bool = True


class ReaderConfig(_Section):
  """Generation budget for the page-reader extraction call, separate from the

  agent's main budget so it's tunable independently.
  """

  max_tokens: int = (
      8192  # generous so reasoning-model readers leave room for output
  )


class RuntimeConfig(_Section):
  """All runtime configuration sections after defaults/file/env merging."""

  generation: GenerationConfig = GenerationConfig()
  limits: LimitsConfig = LimitsConfig()
  backoff: BackoffConfig = BackoffConfig()
  recovery: RecoveryConfig = RecoveryConfig()
  reader: ReaderConfig = ReaderConfig()


# env var -> (section, key, caster). Highest precedence so a live deployment is
# tunable without editing the file.
_ENV_OVERRIDES: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "MAX_TOKENS": ("generation", "max_tokens", int),
    "PER_CALL_TIMEOUT_S": ("limits", "per_call_timeout_s", int),
    "MAX_RUN_DURATION_S": ("limits", "run_wall_clock_s", int),
    "MAX_LLM_CALL_PER_RUN": ("limits", "max_rounds", int),
    "CONTEXT_WINDOW": ("recovery", "context_window", int),
}


def _strip_comments(value):
  """Drop ``_``-prefixed keys recursively so the file can self-document."""
  if isinstance(value, dict):
    return {
        k: _strip_comments(v)
        for k, v in value.items()
        if not (isinstance(k, str) and k.startswith("_"))
    }
  return value


def _deep_merge(
    base, over
):
  """Recursively merge ``over`` onto a copy of ``base`` (dicts merge; scalars

  replace).
  """
  out = copy.deepcopy(base)
  for k, v in over.items():
    if isinstance(v, Mapping) and isinstance(out.get(k), dict):
      out[k] = _deep_merge(out[k], v)
    else:
      out[k] = v
  return out


def load_runtime_config(
    path = None,
    *,
    env = None,
    overrides = None,
):
  """Load the runtime config applying defaults < file < overrides < env."""
  env = os.environ if env is None else env
  merged = RuntimeConfig().model_dump()

  p = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
  if p.exists():
    try:
      file_cfg = _strip_comments(json.loads(p.read_text()))
      if isinstance(file_cfg, dict):
        merged = _deep_merge(merged, file_cfg)
    except (json.JSONDecodeError, OSError, ValueError):
      pass  # malformed file → defaults; the harness must always run

  if overrides:
    merged = _deep_merge(merged, _strip_comments(overrides))

  for env_name, (section, key, cast) in _ENV_OVERRIDES.items():
    raw = env.get(env_name)
    if not raw:
      continue
    # ignore an unparseable env value; keep the lower layer
    with contextlib.suppress(ValueError, TypeError):
      merged[section][key] = cast(raw)

  return RuntimeConfig(**merged)
