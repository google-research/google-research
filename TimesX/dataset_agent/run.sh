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


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${SCRIPT_DIR}/orchestrator" ]]; then
  # Running under Bazel
  ORCHESTRATOR_BIN="${SCRIPT_DIR}/orchestrator"
else
  # Running in open source environment
  ORCHESTRATOR_BIN="python3 -m dataset_agent.orchestrator"
fi

${ORCHESTRATOR_BIN} \
  --config dataset_agent/configs/smoke_1.yaml \
  --validate-config-only

if [[ "${RUN_DATASET_AGENT_LIVE_SMOKE:-0}" == "1" ]]; then
  ${ORCHESTRATOR_BIN} \
    --config dataset_agent/configs/smoke_1.yaml \
    --output-root /tmp/dataset_agent_smoke
fi
