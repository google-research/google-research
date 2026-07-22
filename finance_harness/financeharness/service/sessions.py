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

"""Durable session store for the service (light multi-turn, resume across restarts).

Each session is one JSON file ``<dir>/<id>.json`` holding the accumulated
conversation (system stripped) plus metadata — the directory *is* the index, so
there is no second source of truth to drift. ``list()`` reads the metadata for a
pick-from-a-list UX; the long UUIDs stay internal. A turn is committed only when
it
cleanly answered (no-poison-on-error). Writes are atomic (temp + rename); reads
fail-open (a corrupt or missing file reads as absent, never raises).

Location: ``$FH_SESSIONS_DIR`` or ``~/.financeharness/sessions``.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any
import uuid

from financeharness.runtime.sessions import is_committable, strip_system

_TITLE_MAX = 80


def default_sessions_dir():
  """Default durable-session directory, overridable with ``FH_SESSIONS_DIR``."""

  env = os.environ.get("FH_SESSIONS_DIR")
  if env:
    return Path(env).expanduser()
  return Path.home() / ".financeharness" / "sessions"


def _now():
  return datetime.now(UTC).isoformat(timespec="seconds")


def _title_from(messages):
  """First user question, trimmed — the human-readable label in the session list."""
  for m in messages:
    if m.get("role") == "user" and (c := (m.get("content") or "").strip()):
      return c[:_TITLE_MAX] + ("…" if len(c) > _TITLE_MAX else "")
  return "(untitled)"


def _turns(messages):
  return sum(1 for m in messages if m.get("role") == "user")


class SessionStore:
  """File-backed session store.

  ``directory`` defaults to the user sessions dir.
  """

  def __init__(self, directory = None):
    self._dir = (
        Path(directory) if directory is not None else default_sessions_dir()
    )

  # ----- ids / paths ----------------------------------------------------- #

  def new_id(self):
    return uuid.uuid4().hex

  def _path(self, session_id):
    return self._dir / f"{session_id}.json"

  # ----- low-level record I/O (fail-open) -------------------------------- #

  def _read(self, session_id):
    try:
      with self._path(session_id).open(encoding="utf-8") as fh:
        rec = json.load(fh)
      return rec if isinstance(rec, dict) else None
    except (OSError, json.JSONDecodeError):
      return None

  def _write(self, record):
    self._dir.mkdir(parents=True, exist_ok=True)
    path = self._path(record["id"])
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
      json.dump(record, fh, ensure_ascii=False)
    os.replace(tmp, path)  # atomic — no half-written session

  def _save(self, session_id, messages):
    msgs = strip_system(messages)
    existing = self._read(session_id)
    record = {
        "id": session_id,
        "created_at": (existing or {}).get("created_at") or _now(),
        "updated_at": _now(),
        "title": (existing or {}).get("title") or _title_from(msgs),
        "turns": _turns(msgs),
        "messages": msgs,
    }
    self._write(record)

  # ----- public API ------------------------------------------------------ #

  def history(self, session_id):
    if not session_id:
      return None
    rec = self._read(session_id)
    return rec.get("messages") if rec else None

  def commit(self, session_id, trajectory):
    """Thread this turn into the session, only if it cleanly answered."""
    if not session_id or not is_committable(trajectory.get("termination", "")):
      return
    self._save(session_id, trajectory.get("messages") or [])

  def replace(self, session_id, messages):
    """Replace a session's stored history (used by /compact to swap in a summary)."""
    self._save(session_id, messages)

  def list(self):
    """Session summaries (no message bodies), most-recently-updated first."""
    out: list[dict[str, Any]] = []
    if not self._dir.is_dir():
      return out
    for path in self._dir.glob("*.json"):
      rec = self._read(path.stem)
      if not rec:
        continue
      out.append({
          "id": rec.get("id", path.stem),
          "title": rec.get("title", "(untitled)"),
          "created_at": rec.get("created_at", ""),
          "updated_at": rec.get("updated_at", ""),
          "turns": rec.get("turns", 0),
      })
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out

  def __contains__(self, session_id):
    return self._path(session_id).exists()
