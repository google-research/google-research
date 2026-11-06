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


# ==============================================================================
# 1. Extract mode (first positional argument)
# ==============================================================================
# Check first argument exists and is not a flag
if [[ -z "$1" || "$1" == --* ]]; then
  echo "❌ Error: Must specify mode as first argument"
  echo "Usage: $0 <mode> --round <num> [options]"
  echo "Supported modes: model_response, parsing_model_to_tool, parsing_tool_to_input, google_search, semantic_scholar"
  exit 1
fi

MODE=$1
shift 1  # Shift past mode argument

# ==============================================================================
# 2. Configuration defaults
# ==============================================================================
ROUND=""                              # Required
RUN_ID="first_run"                    # Default
PART=1                                # Default
BATCH_SIZE=20                         # Default
MAX_CONCURRENCY=2                     # Default
THINKING_LEVEL="NONE"                 # Default
INCLUDE_ERROR_FLAG=""                 # Default
MODEL_NAME="gemini-3.1-pro-preview"   # Added
PROMPT="scratchpad"                   # Added

cd ${PIPELINE_ROOT:-.}


# ==============================================================================
# 3. Parse arguments
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
    --thinking_level)
      THINKING_LEVEL="$2"
      shift 2
      ;;
    --model_name)                     # Added
      MODEL_NAME="$2"
      shift 2
      ;;
    --prompt)                         # Added
      PROMPT="$2"
      shift 2
      ;;
    --include_error_tool_call)
      INCLUDE_ERROR_FLAG="--include_error_tool_call" # capture flag
      shift 1 # flag has no value
      ;;
    *)
      echo "Error: unknown argument $1"
      exit 1
      ;;
  esac
done

# ==============================================================================
# 4. Required parameter validation
# ==============================================================================
if [ -z "$ROUND" ]; then
  echo "Error: --round is required!"
  echo "Usage: $0 <mode> --round <num> [options]"
  exit 1
fi

# Warn if model_response-specific args used in other modes
if [[ "$MODE" != "model_response" && ("$THINKING_LEVEL" != "NONE" || "$MODEL_NAME" != "gemini-3.1-pro-preview" || "$PROMPT" != "scratchpad") ]]; then
  echo "Warning: some args only apply to model_response mode"
fi

# ==============================================================================
#
# ==============================================================================
echo "==================================================================="
echo " ✨ Pipeline Meta Script Started ✨"
echo "==================================================================="
echo " 🎯 Mode:            $MODE"
echo " 🔄 Round:           $ROUND"
echo " 🆔 Run ID:          $RUN_ID"
echo " 🧩 Part:            $PART"
echo " 📦 Batch Size:      $BATCH_SIZE"
echo " ⚡ Max Concurrency: $MAX_CONCURRENCY"
if [[ "$MODE" == "model_response" ]]; then
  echo " 🧠 Thinking Level:  $THINKING_LEVEL"
  echo " 🤖 Model Name:      $MODEL_NAME"
  echo " 📝 Prompt Version:  $PROMPT"
fi
if [[ "$MODE" == "parsing_tool_to_input" ]]; then
  if [[ -n "$INCLUDE_ERROR_FLAG" ]]; then
    echo " ⚠️ Include Errors:  True"
  else
    echo " ⚠️ Include Errors:  False"
  fi
fi
echo "==================================================================="

# ==============================================================================
#
# ==============================================================================
echo "🚀 Starting stage: $MODE (run_id=$RUN_ID, round=$ROUND, part=$PART)"

case $MODE in
  model_response)
    bash scripts/model_response/batch.sh \
      --run_id "$RUN_ID" --round "$ROUND" --part "$PART" \
      --model_name "$MODEL_NAME" --prompt "$PROMPT" \
      --thinking_level "$THINKING_LEVEL"
    ;;
    
  parsing_model_to_tool)
    bash scripts/parsing/parsing_model_output_to_tool_input.sh \
      --run_id "$RUN_ID" --round "$ROUND"
    ;;
    
  parsing_tool_to_input)
    bash scripts/parsing/parsing_tool_output_to_input.sh \
      --run_id "$RUN_ID" --round "$ROUND" $INCLUDE_ERROR_FLAG
    
    #
    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
        exit $RET_CODE #
    fi

    #
    NEXT_ROUND=$((ROUND + 1))
    NEXT_INPUT_FILE="files/rounds_processing/round_${NEXT_ROUND}/input/${RUN_ID}.jsonl"
    
    if [ -f "$NEXT_INPUT_FILE" ]; then
        #
        LINE_COUNT=$(grep -c "[^[:space:]]" "$NEXT_INPUT_FILE" || true)
        
        if [ "$LINE_COUNT" -eq 0 ]; then
            echo "==================================================================="
            echo "🏁 No remaining inputs for this round."
            echo "🏁 The pipeline is finished."
            echo "==================================================================="
            exit 222
        fi
    else
        echo "==================================================================="
        echo "🏁 No input file found for this round."
        echo "🏁 The pipeline is finished."
        echo "==================================================================="
        exit 222
    fi
    # ============================================================================
    ;;
    
  google_search)
    bash scripts/tool_call/google_search.sh \
      --run_id "$RUN_ID" --round "$ROUND" --part "$PART" \
      --model_name "$MODEL_NAME"
    ;;
    
  semantic_scholar)
    bash scripts/tool_call/semantic_scholar.sh \
      --run_id "$RUN_ID" --round "$ROUND" --part "$PART" --batch_size "$BATCH_SIZE" --max_concurrency "$MAX_CONCURRENCY"
    ;;
    
  *)
    #
    echo "❌ Unknown mode: $MODE"
    echo "Valid modes:"
    echo " - model_response"
    echo " - parsing_model_to_tool"
    echo " - parsing_tool_to_input"
    echo " - google_search"
    echo " - semantic_scholar"
    exit 1
    ;;
esac

#
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "==================================================================="
  echo "✅ Stage $MODE completed successfully."
  echo "==================================================================="
elif [ $EXIT_CODE -eq 222 ]; then
  echo "==================================================================="
  echo "🏁 Pipeline finished: nothing left to process."
  echo "==================================================================="
  exit 222  #
else
  echo "==================================================================="
  echo "❌ Stage $MODE failed with exit code $EXIT_CODE."
  echo "==================================================================="
  exit 1
fi