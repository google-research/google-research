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
# Evaluate SFT checkpoints
#
# Usage:
#   bash script/sft/eval_sft.sh --task deep_research_bench --gpu 0
#   bash script/sft/eval_sft.sh --task researchqa --gpu 1
#   bash script/sft/eval_sft.sh --model /path/to/custom/sft --task deep_research_bench --gpu 2
# ==========================================================================

GPU=0
TASK="deep_research_bench"
NUM_EXAMPLES="final_run"
MAX_CONCURRENT=4
NO_CITATION=false

# Default SFT model (the one used as base for all RL runs)
MODEL=""
NAME=""
PROMPT_VERSION="rubrics_sft_v2"
TOOL="gemini_grounded_google_search"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --no-citation) NO_CITATION=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-generate name from model path if not provided
if [ -z "$NAME" ]; then
    NAME="sft-$(basename $MODEL)"
fi

EXTRA_ARGS=""
if [ "$NO_CITATION" = true ]; then
    EXTRA_ARGS="--no-citation"
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
    $EXTRA_ARGS
