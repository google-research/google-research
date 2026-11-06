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

# ==========================================================================
# Evaluate CREDIT ASSIGNMENT RL checkpoints
#
# Usage:
#   bash script/rl/eval_credit_assignment.sh --step 50 --task deep_research_bench --gpu 0
#   bash script/rl/eval_credit_assignment.sh --step 150 --task researchqa --gpu 1
# ==========================================================================

STEP=""
GPU=0
TASK="deep_research_bench"
NUM_EXAMPLES="final_run"
MAX_CONCURRENT=4
EVAL_ONLY=false

CKPT_BASE=""
OUTPUT_BASE="${OUTPUT_DIR:-./benchmark_results}"
PROMPT_VERSION="rubrics_sft_v2"
TOOL="gemini_grounded_google_search"
TOOL_MODEL="gemini-3-flash-preview"

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --tool-model) TOOL_MODEL="$2"; shift 2 ;;
        --ckpt-base) CKPT_BASE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$STEP" ]; then
    echo "Error: --step is required"
    echo "Available steps:"
    ls "$CKPT_BASE" 2>/dev/null
    exit 1
fi

MODEL="$CKPT_BASE/step_$STEP"
NAME="credit-assignment-rl-step-$STEP"
TOOL_MODEL_SLUG=$(echo "$TOOL_MODEL" | sed -e 's/^gemini-/g/' -e 's/-preview$//' -e 's/\.//g' | tr '/' '-')
NAME="${NAME}_${TOOL_MODEL_SLUG}"

if [ ! -d "$MODEL" ]; then
    echo "Error: checkpoint not found at $MODEL"
    echo "Available steps:"
    ls "$CKPT_BASE" 2>/dev/null
    exit 1
fi

EXTRA_ARGS=""
if [ "$EVAL_ONLY" = true ]; then
    EXTRA_ARGS="--eval-only"
fi

bash ${REPO_ROOT:-/path/to/repo}/script/eval_launcher.sh \
    --model "$MODEL" \
    --task "$TASK" \
    --gpu "$GPU" \
    --name "$NAME" \
    --num-examples "$NUM_EXAMPLES" \
    --max-concurrent "$MAX_CONCURRENT" \
    --prompt "$PROMPT_VERSION" \
    --tool "$TOOL" \
    --tool-model "$TOOL_MODEL" \
    $EXTRA_ARGS
