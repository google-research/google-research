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

# Submit and download Google Search batch job via Gemini Batch API (File API).

set -e

RUN_ID="first_run"
ROUND=1
PART=1
MODEL_NAME="gemini-3-flash-preview"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_id) RUN_ID="$2"; shift 2 ;;
        --round) ROUND="$2"; shift 2 ;;
        --part) PART="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Google Search Batch Job"
echo "  Run ID:    $RUN_ID"
echo "  Round:     $ROUND"
echo "  Part:      $PART"
echo "=========================================="

echo -e "\n[Step 1/2] Submitting batch job..."
python tool_call_stage/tool_batch_submit.py \
    --run_id "$RUN_ID" --round "$ROUND" --part "$PART" \
    --model_name "$MODEL_NAME"

echo -e "\n[Step 2/2] Downloading batch results..."
python tool_call_stage/tool_batch_download.py \
    --run_id "$RUN_ID" --round "$ROUND" --part "$PART"

echo -e "\nGoogle Search batch complete."
