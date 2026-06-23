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

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

TERMINAL_STATES = frozenset({
    "accepted",
    "no_events_found",
    "pending_retry",
    "unverified_only",
    "waiting_next_day_quota",
    "failed",
})

SCHEDULER_STATES = frozenset({
    "pending",
    "running",
    *TERMINAL_STATES,
})

RUNTIME_TERMINAL_STATE_KEY = "_cw_terminal_state"


def utc_now():
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_terminal_status(status):
  if status not in TERMINAL_STATES:
    raise ValueError(f"Unsupported terminal status: {status}")
  return status


def validate_scheduler_status(status):
  if status not in SCHEDULER_STATES:
    raise ValueError(f"Unsupported scheduler status: {status}")
  return status


def final_dataset_json_path(config):
  final_dataset_path = Path(config["paths"]["final_dataset"])
  if final_dataset_path.suffix.lower() == ".csv":
    return final_dataset_path.with_suffix(".json")
  return final_dataset_path


def task_output_dir(config):
  return final_dataset_json_path(config).parent


def heartbeat_path(config):
  return task_output_dir(config) / "heartbeat.json"


def task_outcome_path(config):
  return task_output_dir(config) / "task_outcome.json"


def _ensure_parent(path):
  path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path, payload):
  _ensure_parent(path)
  fd, temp_path = tempfile.mkstemp(
      prefix=f".{path.name}.", dir=str(path.parent)
  )
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(temp_path, path)
  finally:
    if os.path.exists(temp_path):
      os.unlink(temp_path)


def read_json_file(path):
  if not path.exists():
    return None
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return None


def read_heartbeat(config):
  return read_json_file(heartbeat_path(config))


def read_task_outcome(config):
  return read_json_file(task_outcome_path(config))


def write_heartbeat(config, payload):
  _atomic_write_json(heartbeat_path(config), payload)


def write_task_outcome(config, payload):
  _atomic_write_json(task_outcome_path(config), payload)


def set_terminal_state(
    config,
    terminal_status,
    reason,
    last_completed_stage,
    total_events = 0,
    verified_events = 0,
    safe_to_accept = False,
    workflow_outcome = None,
    extra_fields = None,
):
  validate_terminal_status(terminal_status)
  payload: dict[str, Any] = {
      "terminal_status": terminal_status,
      "reason": reason,
      "safe_to_accept": bool(safe_to_accept),
      "verified_events": int(verified_events),
      "total_events": int(total_events),
      "last_completed_stage": last_completed_stage,
      "workflow_outcome": workflow_outcome or terminal_status,
      "recorded_at_utc": utc_now(),
  }
  if extra_fields:
    payload.update(extra_fields)
  config[RUNTIME_TERMINAL_STATE_KEY] = payload
  return payload


def get_terminal_state(config):
  payload = config.get(RUNTIME_TERMINAL_STATE_KEY)
  if not isinstance(payload, dict):
    return None
  status = payload.get("terminal_status")
  if status not in TERMINAL_STATES:
    return None
  return payload


def build_heartbeat_payload(
    *,
    task_id,
    keyword,
    project_name,
    stage,
    status,
    time_range,
    config_path,
    pid = None,
):
  validate_scheduler_status(status)
  return {
      "task_id": task_id,
      "keyword": keyword,
      "project_name": project_name,
      "stage": stage,
      "status": status,
      "pid": int(pid or os.getpid()),
      "time_range": time_range,
      "config_path": config_path,
      "updated_at_utc": utc_now(),
  }


def build_task_outcome_payload(
    config, terminal_state
):
  status = validate_terminal_status(str(terminal_state["terminal_status"]))
  return {
      "task_id": config.get("_cw_task_id", ""),
      "keyword": config.get("target_variable", ""),
      "project_name": config.get("project_name", ""),
      "terminal_status": status,
      "reason": str(terminal_state.get("reason", "")).strip(),
      "safe_to_accept": bool(terminal_state.get("safe_to_accept", False)),
      "verified_events": int(terminal_state.get("verified_events", 0) or 0),
      "total_events": int(terminal_state.get("total_events", 0) or 0),
      "last_completed_stage": (
          str(terminal_state.get("last_completed_stage", "")).strip()
      ),
      "workflow_outcome": (
          str(terminal_state.get("workflow_outcome", status)).strip()
      ),
      "final_dataset_path": str(final_dataset_json_path(config)),
      "recorded_at_utc": utc_now(),
  }
