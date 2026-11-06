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


# 1. Set your defaults
RUN_ID="first_run"
ROUND=""

# 2. Parse the arguments
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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# 3. Validation (The "Required" check)
if [[ -z "$ROUND" ]]; then
    echo "Error: The --round argument is required."
    echo "Usage: $0 --round <value> [--run_id <value>]"
    exit 1
fi

# 4. Use your variables
echo "Starting process..."
echo "Run ID: $RUN_ID"
echo "Round:  $ROUND"


cd ${PIPELINE_ROOT:-.}
python prompt_processing_stage/processing_model_output_to_tool_input.py --run_id $RUN_ID --round $ROUND