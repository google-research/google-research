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


cd ${PIPELINE_ROOT:-.}

# 1. Initialize parameters and defaults
part="1"
thinking_level=""
include_error_flag=""
model_name="gemini-3.1-pro-preview"
prompt="scratchpad"

# 2. Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start_round) start_round="$2"; shift ;;
        --end_round) end_round="$2"; shift ;;
        --run_id) run_id="$2"; shift ;;
        --part) part="$2"; shift ;;
        --thinking_level) thinking_level="$2"; shift ;;
        --model_name) model_name="$2"; shift ;;
        --prompt) prompt="$2"; shift ;;
        --include_error_tool_call) include_error_flag="--include_error_tool_call" ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# 3. Validate required params
if [ -z "$start_round" ] || [ -z "$end_round" ] || [ -z "$run_id" ]; then
    echo "Error: missing required arguments!"
    echo "Usage: bash auto_pipeline.sh --start_round N --end_round M --run_id ID [--part P] [--thinking_level LEVEL] [--model_name MODEL] [--prompt PROMPT]"
    exit 1
fi

thinking_args=()
if [ -n "$thinking_level" ]; then
    thinking_args=(--thinking_level "$thinking_level")
fi

# 4. Run the Python orchestrator
python auto_pipeline.py \
    --start_round "$start_round" \
    --end_round "$end_round" \
    --run_id "$run_id" \
    --part "$part" \
    "${thinking_args[@]}" \
    --model_name "$model_name" \
    --prompt "$prompt" \
    $include_error_flag
