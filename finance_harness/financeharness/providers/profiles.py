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

"""Model profiles — the config that selects *which* backbone a call uses.

A profile names a model, an OpenAI-compatible endpoint, the credential to use,
generation defaults, and an optional quirk adapter. Profiles come from built-in
defaults merged with an optional JSON file (``configs/providers.json``); the
active default is chosen by ``FH_PROFILE`` env > file ``"default"`` key >
``"vllm"`` (the OSS backbone).

Design mirrors ``runtime/config.py`` precedence (defaults < file < env) and
never raises on a missing/malformed file — the seam always has a usable default.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Repo-relative default location for the profiles file (optional).
_DEFAULT_PROFILES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "providers.json"
)


class Generation(BaseModel):
  """Sampling/budget defaults for a profile.

  ``max_tokens`` here is the starting budget; the recovery layer may escalate
  it.
  """

  model_config = ConfigDict(extra="forbid")

  temperature: float = 0.7
  top_p: float = 1.0
  max_tokens: int = 4096
  presence_penalty: float = 0.0


class ModelProfile(BaseModel):
  """A single backbone target.

  ``base_url=None`` uses the OpenAI default endpoint; set it to e.g.
  ``http://127.0.0.1:8000/v1`` for a local vLLM. The API key is resolved at
  client-construction time (never stored).
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  model: str
  base_url: str | None = None
  api_key_env: str = "OPENAI_API_KEY"
  api_key_literal: str | None = None  # e.g. "EMPTY" for an open vLLM
  timeout_s: float = 600.0
  generation: Generation = Field(default_factory=Generation)
  extra_body: dict[str, Any] = Field(default_factory=dict)
  adapter: str | None = None  # quirk-adapter name (see providers/adapters)
  # The wire protocol the loop uses (providers/base.Provider): "chat" =
  # OpenAI-compatible Chat Completions (works with vLLM / any compatible endpoint);
  # "gemini" = the native google-genai SDK. Selects the provider via provider_for().
  provider: Literal["chat", "gemini"] = "chat"
  # Role in the harness: a "backbone" drives the agent loop (and is switchable via
  # /model); a "reader" is the page-reading model and is never a loop backbone.
  role: Literal["backbone", "reader"] = "backbone"
  # The reader (page-extraction) profile to pair with this backbone. None → the
  # default vllm-reader. Cloud backbones name a cloud reader so a run never depends
  # on the local reader being served.
  reader_profile: str | None = None

  def resolve_api_key(self, env = None):
    """Resolve the credential: explicit literal wins, else the env var,

    else "" (the client falls back to a placeholder for keyless vLLM).
    """
    if self.api_key_literal is not None:
      return self.api_key_literal
    return (env if env is not None else os.environ).get(self.api_key_env, "")


# Built-in profiles. The default backbone is a local **open-weight** stack (the
# loop + the page reader) served via vLLM over an OpenAI-compatible endpoint;
# `gemini` is a selectable cloud backbone (role=backbone, switchable via /model).
# Per-profile base_url is overridable via FH_<NAME>_BASE_URL; the vLLM defaults
# point at a local OpenAI-compatible serving, loopback-published. The `model`
# ids should match your vLLM ``--served-model-name``.
_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"  # local vLLM (open-weight backbone)
_VLLM_READER_BASE_URL = (  # local vLLM (open-weight reader)
    "http://127.0.0.1:8001/v1"
)

_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "vllm": {
        "name": "vllm",
        "model": "vllm-backbone",
        "base_url": _VLLM_BASE_URL,
        "api_key_literal": "EMPTY",
        "adapter": "vllm",  # thinking mode + top_k
        "reader_profile": "vllm-reader",  # the local reader
        "generation": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    },
    "vllm-reader": {
        "name": "vllm-reader",
        "model": "vllm-reader",
        "role": "reader",
        "base_url": _VLLM_READER_BASE_URL,
        "api_key_literal": "EMPTY",
        # Reader does fast JSON extraction — thinking OFF (top_k via extra_body).
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 8192},
    },
    "gemini": {
        "name": "gemini",
        "model": "gemini-3.5-flash",
        # Native google-genai SDK in the loop — structured thoughts + thought_signature
        # threading (clean multi-turn tool calling). Needs GEMINI_API_KEY. base_url is
        # the OpenAI-compat endpoint, used only by the page-reader's complete() path.
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "reader_profile": (
            "gemini"
        ),  # flash-3.5 reads its own pages (compat path)
        "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 16384},
    },
}

_DEFAULT_PROFILE_NAME = "vllm"


def _read_file(path):
  """Best-effort JSON read; returns {} on any problem (never raises)."""
  try:
    return json.loads(path.read_text())
  except Exception:  # noqa: BLE001 — config is best-effort; fall back to defaults
    return {}


def load_profiles(
    path = None,
    env = None,
):
  """Load all profiles: built-ins merged with the JSON file's ``profiles``

  (file wins per-name). Validation errors on a single file profile are
  skipped, not fatal.
  """
  raw: dict[str, dict[str, Any]] = {
      k: dict(v) for k, v in _BUILTIN_PROFILES.items()
  }

  p = Path(path) if path is not None else _DEFAULT_PROFILES_PATH
  file_data = _read_file(p) if p.exists() else {}
  for name, spec in (file_data.get("profiles") or {}).items():
    merged = {**raw.get(name, {}), **spec, "name": name}
    raw[name] = merged

  e = env if env is not None else os.environ
  profiles: dict[str, ModelProfile] = {}
  for name, spec in raw.items():
    # Per-profile base_url override: FH_<NAME>_BASE_URL (NAME upper, - → _).
    override = e.get(f"FH_{name.upper().replace('-', '_')}_BASE_URL")
    if override:
      spec = {**spec, "base_url": override}
    try:
      profiles[name] = ModelProfile(**spec)
    except Exception:  # noqa: BLE001 — skip a malformed profile, keep the rest
      continue
  return profiles


def default_profile_name(
    path = None,
    env = None,
):
  """Active default: ``FH_PROFILE`` env > file ``"default"`` > built-in (vllm)."""
  e = env if env is not None else os.environ
  if e.get("FH_PROFILE"):
    return e["FH_PROFILE"]
  p = Path(path) if path is not None else _DEFAULT_PROFILES_PATH
  file_default = _read_file(p).get("default") if p.exists() else None
  return file_default or _DEFAULT_PROFILE_NAME


def get_profile(
    name = None,
    *,
    path = None,
    env = None,
):
  """Resolve a profile by name (or the active default).

  Falls back to the built-in default (``vllm``) if the requested name is
  unknown.
  """
  profiles = load_profiles(path=path, env=env)
  resolved = name or default_profile_name(path=path, env=env)
  return profiles.get(resolved) or profiles[_DEFAULT_PROFILE_NAME]


def available_backbones(
    *,
    path = None,
    env = None,
):
  """Switchable backbone profiles (``role == "backbone"``) with credential

  availability — the page reader is excluded. ``available`` is True for a
  keyless
  vLLM (``api_key_literal`` set) or when the profile's API key resolves. Powers
  the
  ``/model`` switch: available first (the default floated up), then
  alphabetically.
  Which profile is active in a session is client state (the stateless endpoint
  only
  knows the default — it's returned separately by ``/models``).
  """
  e = env if env is not None else os.environ
  active = default_profile_name(path=path, env=env)
  out: list[dict[str, Any]] = []
  for p in load_profiles(path=path, env=env).values():
    if (
        p.role != "backbone"
    ):  # readers are paired internally, not user-selectable
      continue
    available = p.api_key_literal is not None or bool(p.resolve_api_key(e))
    out.append({"name": p.name, "model": p.model, "available": available})
  # available first, the default floated to the top, then alphabetical.
  out.sort(key=lambda m: (not m["available"], m["name"] != active, m["name"]))
  return out
