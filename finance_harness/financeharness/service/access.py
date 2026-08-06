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

"""Invite-token access control: one revocable grant per person, with quotas.

A run spends real model quota and `visit` fetches arbitrary URLs from inside
this host's network, so a shared secret is not enough once the service is
reachable by more than one person: it can't be revoked for one user, can't be
rate-limited per user, and can't own a session. Each invitee gets their own
token instead.

Only the SHA-256 of a token is stored, so a leaked grants file does not grant
access. Tokens are shown exactly once, at mint time.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import uuid

_TOKEN_PREFIX = "fh_"
# Per-grant defaults. A single research run averages ~45 rounds / ~20 minutes of
# model time, so these are deliberately small — raise per grant when minting.
_DEFAULT_DAILY_RUNS = 20
_DEFAULT_CONCURRENT_RUNS = 2


def default_access_file():
  """Grants file, overridable with ``FH_ACCESS_FILE``."""
  env = os.environ.get("FH_ACCESS_FILE")
  if env:
    return pathlib.Path(env).expanduser()
  return pathlib.Path.home() / ".financeharness" / "access.json"


def _now():
  return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _today():
  return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")


def _hash(token):
  return hashlib.sha256(token.encode("utf-8")).hexdigest()


class QuotaExceededError(Exception):
  """A grant hit its daily or concurrent run limit."""


class Grant:
  """One invitee's access: identity, limits, and live usage."""

  def __init__(self, record):
    self.id = record["id"]
    self.label = record.get("label") or record["id"]
    self.admin = bool(record.get("admin"))
    self.daily_runs = int(record.get("daily_runs", _DEFAULT_DAILY_RUNS))
    self.concurrent_runs = int(
        record.get("concurrent_runs", _DEFAULT_CONCURRENT_RUNS)
    )
    self.revoked_at = record.get("revoked_at")
    self.created_at = record.get("created_at", "")

  @property
  def revoked(self):
    return bool(self.revoked_at)

  def public(self):
    """Identity + limits, safe to hand back to the client (never the token)."""
    return {
        "id": self.id,
        "label": self.label,
        "admin": self.admin,
        "daily_runs": self.daily_runs,
        "concurrent_runs": self.concurrent_runs,
        "created_at": self.created_at,
    }


class AccessStore:
  """File-backed grants, with in-process concurrency and per-UTC-day counters.

  Daily counts live in the same file so they survive a restart; concurrent
  counts are in-process, which assumes a single uvicorn worker (the deployment
  this service is built for). Run it with multiple workers and the concurrency
  cap becomes per-worker.
  """

  def __init__(self, path=None):
    self._path = (
        pathlib.Path(path) if path is not None else default_access_file()
    )
    self._active: dict[str, int] = {}

  # ----- record I/O ------------------------------------------------------- #

  def _read(self):
    try:
      with self._path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
      return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
      return {}

  def _write(self, doc):
    self._path.parent.mkdir(parents=True, exist_ok=True)
    tmp = self._path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
      json.dump(doc, fh, ensure_ascii=False, indent=2)
    # Token hashes and usage — not world-readable, and never group-readable.
    os.chmod(tmp, 0o600)
    os.replace(tmp, self._path)

  # ----- lifecycle -------------------------------------------------------- #

  def enabled(self):
    """True once at least one grant exists — the service gates itself then."""
    return bool(self._read().get("grants"))

  def mint(
      self,
      label,
      *,
      daily_runs=_DEFAULT_DAILY_RUNS,
      concurrent_runs=_DEFAULT_CONCURRENT_RUNS,
      admin=False,
  ):
    """Create a grant and return ``(grant, raw_token)``.

    The raw token is returned exactly once — only its hash is persisted.

    Args:
      label: A human-readable label for the grantee.
      daily_runs: Max daily runs allowed for the grant.
      concurrent_runs: Max concurrent runs allowed for the grant.
      admin: Whether the grant has admin privileges.

    Returns:
      A tuple of (Grant, str) containing the Grant object and raw token.
    """
    doc = self._read()
    grants = doc.setdefault("grants", [])
    raw = _TOKEN_PREFIX + secrets.token_urlsafe(32)
    record = {
        "id": uuid.uuid4().hex[:8],
        "label": label,
        "token_sha256": _hash(raw),
        "created_at": _now(),
        "revoked_at": None,
        "daily_runs": daily_runs,
        "concurrent_runs": concurrent_runs,
        "admin": admin,
    }
    grants.append(record)
    self._write(doc)
    return Grant(record), raw

  def revoke(self, id_or_label):
    """Revoke by grant id or label. Returns the number of grants revoked."""
    doc = self._read()
    hit = 0
    for record in doc.get("grants", []):
      if record.get("revoked_at"):
        continue
      if id_or_label in (record.get("id"), record.get("label")):
        record["revoked_at"] = _now()
        hit += 1
    if hit:
      self._write(doc)
    return hit

  def grants(self):
    """Every grant, active and revoked, with today's usage attached."""
    doc = self._read()
    usage = doc.get("usage", {}).get(_today(), {})
    out = []
    for record in doc.get("grants", []):
      grant = Grant(record)
      info = grant.public()
      info["revoked"] = grant.revoked
      info["used_today"] = int(usage.get(grant.id, 0))
      info["active_now"] = self._active.get(grant.id, 0)
      out.append(info)
    return out

  # ----- authentication --------------------------------------------------- #

  def resolve(self, token):
    """The live grant for ``token``, or None if unknown or revoked."""
    if not token:
      return None
    supplied = _hash(token)
    for record in self._read().get("grants", []):
      if record.get("revoked_at"):
        continue
      # Constant-time: a naive `==` leaks the stored hash one character at a
      # time. Comparing hashes (not raw tokens) keeps the operands fixed-width.
      if hmac.compare_digest(supplied, record.get("token_sha256", "")):
        return Grant(record)
    return None

  # ----- quota ------------------------------------------------------------ #

  def usage_today(self, grant):
    doc = self._read()
    return int(doc.get("usage", {}).get(_today(), {}).get(grant.id, 0))

  def _bump_daily(self, grant):
    doc = self._read()
    usage = doc.setdefault("usage", {})
    # Keep only today's bucket — the file is a live counter, not an audit log.
    today = _today()
    for day in [d for d in usage if d != today]:
      del usage[day]
    day_counts = usage.setdefault(today, {})
    day_counts[grant.id] = int(day_counts.get(grant.id, 0)) + 1
    self._write(doc)

  def begin_run(self, grant):
    """Claim one run against ``grant``'s quota, or raise ``QuotaExceededError``.

    The caller must pair this with ``end_run`` in a finally block, or the
    concurrency slot leaks for the process's lifetime.

    Args:
      grant: The Grant instance to claim a run for.

    Raises:
      QuotaExceededError: If concurrent runs or daily runs limit is reached.
    """
    active = self._active.get(grant.id, 0)
    if active >= grant.concurrent_runs:
      raise QuotaExceededError(
          f"{grant.concurrent_runs} concurrent run(s) already in flight for"
          f" '{grant.label}' — wait for one to finish"
      )
    used = self.usage_today(grant)
    if used >= grant.daily_runs:
      raise QuotaExceededError(
          f"daily limit of {grant.daily_runs} run(s) reached for"
          f" '{grant.label}' — resets at 00:00 UTC"
      )
    self._active[grant.id] = active + 1
    self._bump_daily(grant)

  def end_run(self, grant):
    """Release the concurrency slot claimed by ``begin_run``."""
    active = self._active.get(grant.id, 0)
    if active <= 1:
      self._active.pop(grant.id, None)
    else:
      self._active[grant.id] = active - 1
