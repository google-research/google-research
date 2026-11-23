#!/bin/bash
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


# ==============================================================================
# safe_lemma_runner.sh: Zero-Zombie, Signal-Safe Autonomous Rollout Runner
# Re-launches a headless agent CLI until the orchestrator log records a verified
# [PHASE 5: COMPLETE], surviving crashes, rate limits and endpoint outages.
#
#   LEAP_AGENT_CLI   the binary to drive (required): claude, codex, ...
#   LEAP_TARGET_DIR  the problem root to watch for completion
#   LEAP_RESUME_FLAG how the resume prompt is passed on restarts. Default "-p"
#                    (Claude Code: `claude -p ...`). Set it to "" to pass the
#                    prompt as the last positional argument (`codex exec ...`).
#   LEAP_RESUME_PREFIX text prepended to the resume prompt, e.g. '$leap-prove '
#                    so Codex re-loads the orchestrator skill on restart.
#
# Usually started through `leap loop`, which sets these for you:
#   leap loop claude --agent orchestrator-lemma -p "Prove problems/foo.lean"
#   leap loop codex exec "$leap-prove Prove problems/foo.lean"
# ==============================================================================
set -u
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

export PATH="${HOME}/.local/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
AGENT_CLI="${LEAP_AGENT_CLI:-}"
if [[ -z "${AGENT_CLI}" ]]; then
  echo "error: LEAP_AGENT_CLI is unset. Set it to the binary that runs your" \
       "agent -- there is no portable default to guess." >&2
  exit 1
fi
if ! command -v "${AGENT_CLI}" >/dev/null 2>&1 && [[ ! -x "${AGENT_CLI}" ]]; then
  echo "error: LEAP_AGENT_CLI='${AGENT_CLI}' is neither on PATH nor an" \
       "executable file." >&2
  exit 1
fi
child_pid=""

cleanup() {
  echo "[$(date)] safe_lemma_runner received termination signal. Cleaning up..."
  if [[ -n "${child_pid:-}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  exit 0
}

trap cleanup SIGINT SIGTERM SIGHUP EXIT

FIRST_RUN=true
TARGET_DIR="${LEAP_TARGET_DIR:-}"
if [[ -z "${TARGET_DIR}" ]]; then
  for arg in "$@"; do
    if [[ "${arg}" =~ problem\ root:\ ([^[:space:]\"]+) ]]; then
      TARGET_DIR="${BASH_REMATCH[1]}"
    elif [[ -z "${TARGET_DIR}" && "${arg}" =~ ([^[:space:]\"]+)\.lean ]]; then
      # Default problem root, matching the orchestrator: runs/<file stem>.
      TARGET_DIR="runs/$(basename "${BASH_REMATCH[1]}")"
    fi
  done
fi

while true; do
  CLI_ARGS=()
  if [[ "${FIRST_RUN}" == "true" ]] || [[ -n "${TARGET_DIR}" && ! -f "${TARGET_DIR}/proof_graph.json" ]]; then
    CLI_ARGS=("$@")
  else
    if [[ -n "${TARGET_DIR}" ]]; then
      echo "[$(date)] Releasing stranded claims in ${TARGET_DIR} via release_all..."
      state_manager_bin --root_dir="${TARGET_DIR}" --action=release_all \
        || "${LEAP_PYTHON:-python3}" "${HERE}/state_manager.py" \
             --root_dir="${TARGET_DIR}" --action=release_all || true
    fi

    # Strip existing -p / --prompt arguments to replace with continuation prompt
    skip_next=false
    for arg in "$@"; do
      if [[ "${skip_next}" == "true" ]]; then
        skip_next=false
        continue
      fi
      if [[ "${arg}" == "-p" || "${arg}" == "--prompt" ]]; then
        skip_next=true
        continue
      elif [[ "${arg}" =~ ^(-p|--prompt)= ]]; then
        continue
      fi
      CLI_ARGS+=("${arg}")
    done

    RESUME_PROMPT="${LEAP_RESUME_PREFIX:-}Resume orchestration on the existing graph at ${TARGET_DIR}. All proved nodes and subgoals are preserved in proof_graph.json. Inspect open nodes, release any stranded claims, and continue solving to completion."
    if [[ -n "${LEAP_RESUME_FLAG-"-p"}" ]]; then
      CLI_ARGS+=("${LEAP_RESUME_FLAG-"-p"}" "${RESUME_PROMPT}")
    elif [[ ${#CLI_ARGS[@]} -gt 0 ]]; then
      # Positional prompt: replace the original last argument.
      CLI_ARGS=("${CLI_ARGS[@]:0:${#CLI_ARGS[@]}-1}" "${RESUME_PROMPT}")
    else
      CLI_ARGS=("${RESUME_PROMPT}")
    fi
  fi

  "${AGENT_CLI}" \
    "${CLI_ARGS[@]}" &

  child_pid=$!
  wait "${child_pid}"
  exit_code=$?
  child_pid=""

  if [[ ${exit_code} -eq 0 ]]; then
    # Check if Phase 5 Complete is recorded in THIS specific problem orchestrator log (must be checked [x])
    if [[ -n "${TARGET_DIR}" && -f "${TARGET_DIR}/orchestrator_log.md" ]] && grep -q -- "- \[x\] \[PHASE 5: COMPLETE\]" "${TARGET_DIR}/orchestrator_log.md"; then
      if [[ -f "${TARGET_DIR}/assembled/final_proof.lean" ]] && grep -w -q "sorry" "${TARGET_DIR}/assembled/final_proof.lean"; then
        echo "[$(date)] Assembled final_proof.lean still contains sorry placeholders. Continuing orchestration..."
      else
        echo "[$(date)] Target ${TARGET_DIR} reached verified Phase 5 Complete! Exiting loop."
        break
      fi
    fi
    echo "[$(date)] CLI turn finished. Continuing orchestration in 10s..."
    FIRST_RUN=false
    sleep 10 &
    sleep_pid=$!
    wait "${sleep_pid}" 2>/dev/null || break
  elif [[ ${exit_code} -eq 130 ]]; then
    echo "[$(date)] Execution terminated by user signal (SIGINT / exit code ${exit_code}). Exiting loop."
    break
  else
    echo "[$(date)] Execution exited with code ${exit_code} (transient outage / signal 143). Retrying with clean resume prompt in 60s..."
    FIRST_RUN=false
    sleep 60 &
    sleep_pid=$!
    wait "${sleep_pid}" 2>/dev/null || break
  fi
done
