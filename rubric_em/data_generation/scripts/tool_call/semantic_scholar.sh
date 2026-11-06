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


# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#
set -e

export S2_API_KEY="${S2_API_KEY}"

echo "======================================================"
echo "🚀 Starting Semantic Scholar Async Tool Execution"
echo "======================================================"

# ==============================================================================
#
# ==============================================================================
cd ${PIPELINE_ROOT:-.}
SCRIPT_PATH="tool_call_stage/semantic_scholar_asyn_processing.py"

# ==============================================================================
#
# ==============================================================================
RUN_ID="first_run"
ROUND=""              # Required
PART=1                #
BATCH_SIZE=20         #
MAX_CONCURRENCY=2     #

# ==============================================================================
#
# ==============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --run_id)
      RUN_ID="$2"
      shift 2
      ;;
    --round)
      ROUND="$2"
      shift 2
      ;;
    --part)
      PART="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max_concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 <mode> --round <num> [options]"
      exit 1
      ;;
  esac
done

#
if [ -z "$ROUND" ]; then
  echo "Error: --round is required."
  echo "Usage: $0 <mode> --round <num> [options]"
  exit 1
fi

echo "Configuration:"
echo "  -> RUN_ID:          $RUN_ID"
echo "  -> ROUND:           $ROUND"
echo "  -> PART:            $PART"
echo "  -> BATCH_SIZE:      $BATCH_SIZE"
echo "  -> MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "======================================================"

# ==============================================================================
#
# ==============================================================================

# 🌟 [Added]
INPUT_FILE="files/rounds_processing/round_${ROUND}/semantic_scholar/${RUN_ID}_${PART}.jsonl"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "\n⏭️ [Skip] No Semantic Scholar input file found for Round $ROUND, Part $PART. Skipping."
    exit 0
fi
# ==============================================================================

python $SCRIPT_PATH \
    --run_id "$RUN_ID" \
    --round "$ROUND" \
    --part "$PART" \
    --batch_size "$BATCH_SIZE" \
    --max_concurrency "$MAX_CONCURRENCY"

echo "======================================================"
echo "✅ Semantic Scholar processing for Round $ROUND | Part $PART completed."
echo "======================================================"