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
# Evaluate RUBRIC BANK V2 RL checkpoints
#
# Features:
#   - Uses rubric bank system prompt (with <reference_examples> support)
#   - Loads rubric bank from checkpoint for cross-episode retrieval
#   - After generation, generates reflections and stores in per-dataset bank
#
# Usage:
#   bash script/rl/eval_rubricem.sh --step 100 --task deep_research_bench --gpu 0
#   bash script/rl/eval_rubricem.sh --step 100 --task dsqa --gpu 2
# ==========================================================================

STEP=""
GPU=0
TASK="deep_research_bench"
NUM_EXAMPLES="final_run"
MAX_CONCURRENT=4
NUM_PASSES=1
EVAL_ONLY=false
NO_RETRIEVAL=false
NO_TRAINING_BANK=false
NO_CITATION=false
CROSS_TAKEAWAYS_ONLY=false
DEBUG_PROMPTS=false
ACADEMIC_CITATION=false
TOOL_MODEL="gemini-3-flash-preview"
USE_STANDARD_INSTRUCTION=false

CKPT_BASE=""
OUTPUT_BASE="${OUTPUT_DIR:-./benchmark_results}"
PROMPT_VERSION="rubrics_sft_v2"
TOOL="gemini_grounded_google_search"

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --num-passes) NUM_PASSES="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --no-retrieval) NO_RETRIEVAL=true; shift ;;
        --no-training-bank) NO_TRAINING_BANK=true; shift ;;
        --no-citation) NO_CITATION=true; shift ;;
        --cross-takeaways-only) CROSS_TAKEAWAYS_ONLY=true; shift ;;
        --debug-prompts) DEBUG_PROMPTS=true; shift ;;
        --academic-citation) ACADEMIC_CITATION=true; shift ;;
        --tool-model) TOOL_MODEL="$2"; shift 2 ;;
        --use-standard-instruction) USE_STANDARD_INSTRUCTION=true; shift ;;
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
if [ "$NO_RETRIEVAL" = true ]; then
    NAME="rubric-bank-v2-step-${STEP}_no_retrieval"
elif [ "$NO_TRAINING_BANK" = true ] && [ "$CROSS_TAKEAWAYS_ONLY" = true ]; then
    NAME="rubric-bank-v2-step-${STEP}_no_training_bank_cross_takeaway_only"
elif [ "$NO_TRAINING_BANK" = true ]; then
    NAME="rubric-bank-v2-step-${STEP}_no_training_bank"
else
    NAME="rubric-bank-v2-step-$STEP"
fi
# Append tool-model slug so different Gemini search models go to different dirs.
TOOL_MODEL_SLUG=$(echo "$TOOL_MODEL" | sed -e 's/^gemini-/g/' -e 's/-preview$//' -e 's/\.//g' | tr '/' '-')
NAME="${NAME}_${TOOL_MODEL_SLUG}"
BANK_PATH="$MODEL/rubric_bank/rubric_bank.pkl"

if [ ! -d "$MODEL" ]; then
    echo "Error: checkpoint not found at $MODEL"
    echo "Available steps:"
    ls "$CKPT_BASE" 2>/dev/null
    exit 1
fi

if [ "$NO_RETRIEVAL" = true ]; then
    echo "No-retrieval mode: skipping bank loading"
    BANK_PATH=""
elif [ "$NO_TRAINING_BANK" = true ]; then
    echo "No-training-bank mode: eval bank only (starts empty, builds during eval)"
    BANK_PATH=""
elif [ ! -f "$BANK_PATH" ]; then
    echo "Warning: rubric bank not found at $BANK_PATH"
    echo "Running without bank retrieval"
    BANK_PATH=""
fi

echo "=========================================="
echo "Rubric Bank V2 Evaluation"
echo "  Model:     $MODEL"
echo "  Bank:      ${BANK_PATH:-none}"
echo "  Task:      $TASK"
echo "  Name:      $NAME"
echo "  Passes:    $NUM_PASSES"
echo "  Retrieval: $([ "$NO_RETRIEVAL" = true ] && echo 'disabled' || ([ "$NO_TRAINING_BANK" = true ] && echo 'eval-bank only' || echo 'full'))"
echo "=========================================="

# Use the rubric bank prompt version
PROMPT_VERSION="rubrics_sft_v2_rubric_bank"
mkdir -p "$OUTPUT_BASE"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"
RESULT_DIR="$OUTPUT_BASE/$NAME"

# ==========================================
# Phase 1: Generation for all passes
# ==========================================
if [ "$EVAL_ONLY" = true ]; then
    echo "Skipping generation (--eval-only)"
else
echo ""
echo "Phase 1: Generating responses for $NUM_PASSES passes..."

# Determine starting pass: skip passes that already have both jsonl and bank
START_PASS=1
if [ "$NUM_EXAMPLES" != "final_run" ] && [ -n "$NUM_EXAMPLES" ]; then
    _TASK_PREFIX="${TASK}_n${NUM_EXAMPLES}"
else
    _TASK_PREFIX="${TASK}"
fi
for PASS in $(seq 1 $NUM_PASSES); do
    PASS_JSONL="$RESULT_DIR/${_TASK_PREFIX}_pass_${PASS}.jsonl"
    PASS_BANK="$RESULT_DIR/${_TASK_PREFIX}_rubric_bank_pass_${PASS}/rubric_bank.pkl"
    if [ -f "$PASS_JSONL" ] && [ -f "$PASS_BANK" ]; then
        echo "Pass $PASS already complete: $PASS_JSONL + $PASS_BANK"
        START_PASS=$((PASS + 1))
    else
        break
    fi
done

# Also check for a pre-existing bank without _pass_N suffix (from single-pass run)
# Treat it as pass 1's bank
LEGACY_BANK="$RESULT_DIR/${TASK}_rubric_bank/rubric_bank.pkl"
if [ "$START_PASS" -eq 1 ] && [ -f "$LEGACY_BANK" ]; then
    # A previous single-pass run left a bank — treat as pass 1
    LEGACY_JSONL="$RESULT_DIR/${TASK}.jsonl"
    if [ -f "$LEGACY_JSONL" ]; then
        echo "Found legacy single-pass outputs, treating as pass 1"
        mv "$LEGACY_JSONL" "$RESULT_DIR/${TASK}_pass_1.jsonl" 2>/dev/null || true
        mkdir -p "$RESULT_DIR/${TASK}_rubric_bank_pass_1"
        cp "$LEGACY_BANK" "$RESULT_DIR/${TASK}_rubric_bank_pass_1/rubric_bank.pkl"
        START_PASS=2
    fi
fi

if [ "$START_PASS" -gt "$NUM_PASSES" ]; then
    echo "All $NUM_PASSES generation passes already complete, skipping to evaluation"
else
    for PASS in $(seq $START_PASS $NUM_PASSES); do
        echo ""
        echo "=========================================="
        echo "Generation Pass $PASS / $NUM_PASSES"
        echo "=========================================="

        EXTRA_ARGS=""
        if [ -n "$BANK_PATH" ]; then
            EXTRA_ARGS="--rubric-bank $BANK_PATH"
        fi

        # Initialize eval bank for reflection storage — but NOT for --no-retrieval mode
        if [ "$NO_RETRIEVAL" != true ]; then
            if [ "$PASS" -gt 1 ]; then
                PREV_PASS=$((PASS - 1))
                PREV_EVAL_BANK="$RESULT_DIR/${_TASK_PREFIX}_rubric_bank_pass_${PREV_PASS}/rubric_bank.pkl"
                if [ -f "$PREV_EVAL_BANK" ]; then
                    echo "Loading eval bank from pass $PREV_PASS: $PREV_EVAL_BANK"
                    EXTRA_ARGS="$EXTRA_ARGS --rubric-bank-eval $PREV_EVAL_BANK"
                else
                    echo "Warning: No eval bank from pass $PREV_PASS, using empty bank"
                    EXTRA_ARGS="$EXTRA_ARGS --rubric-bank-eval __empty__"
                fi
            else
                # Pass 1: initialize empty eval bank so reflections are generated and stored
                EXTRA_ARGS="$EXTRA_ARGS --rubric-bank-eval __empty__"
            fi
        fi

        if [ "$CROSS_TAKEAWAYS_ONLY" = true ]; then
            EXTRA_ARGS="$EXTRA_ARGS --cross-takeaways-only"
        fi
        if [ "$DEBUG_PROMPTS" = true ]; then
            EXTRA_ARGS="$EXTRA_ARGS --debug-prompts"
        fi
        if [ "$USE_STANDARD_INSTRUCTION" = true ]; then
            EXTRA_ARGS="$EXTRA_ARGS --use-standard-instruction"
        fi

        export RUBRIC_BANK_EVAL_SAVE_SUFFIX="_pass_${PASS}"

        # Generate only (skip eval)
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
            --skip-eval \
            $EXTRA_ARGS

        # Rename generation output with pass suffix
        if [ "$NUM_EXAMPLES" != "final_run" ] && [ -n "$NUM_EXAMPLES" ]; then
            TASK_PREFIX="${TASK}_n${NUM_EXAMPLES}"
        else
            TASK_PREFIX="${TASK}"
        fi
        JSONL="$RESULT_DIR/${TASK_PREFIX}.jsonl"
        if [ -f "$JSONL" ]; then
            mv "$JSONL" "$RESULT_DIR/${TASK_PREFIX}_pass_${PASS}.jsonl"
            echo "Saved: ${TASK_PREFIX}_pass_${PASS}.jsonl"
        fi
    done
fi
fi  # end of EVAL_ONLY check

# ==========================================
# Phase 2: Evaluation for all passes (no vLLM needed)
# ==========================================
echo ""
echo "Phase 2: Evaluating all $NUM_PASSES passes..."

for PASS in $(seq 1 $NUM_PASSES); do
    echo ""
    echo "=========================================="
    echo "Evaluation Pass $PASS / $NUM_PASSES"
    echo "=========================================="

    if [ "$NUM_EXAMPLES" != "final_run" ] && [ -n "$NUM_EXAMPLES" ]; then
        TASK_PREFIX="${TASK}_n${NUM_EXAMPLES}"
    else
        TASK_PREFIX="${TASK}"
    fi

    PASS_JSONL="$RESULT_DIR/${TASK_PREFIX}_pass_${PASS}.jsonl"
    if [ ! -f "$PASS_JSONL" ]; then
        echo "Warning: $PASS_JSONL not found, skipping eval for pass $PASS"
        continue
    fi

    # Use pass-specific name for eval so DRB results go to separate dirs
    PASS_EVAL_NAME="${NAME}_pass_${PASS}"
    PASS_EVAL_DIR="$OUTPUT_BASE/$PASS_EVAL_NAME"
    mkdir -p "$PASS_EVAL_DIR"

    # Symlink the pass jsonl as the expected filename for eval_launcher
    ln -sf "$PASS_JSONL" "$PASS_EVAL_DIR/$TASK.jsonl"

    EVAL_EXTRA_ARGS=""
    if [ "$NO_CITATION" = true ]; then
        EVAL_EXTRA_ARGS="--no-citation"
    elif [ "$ACADEMIC_CITATION" = true ]; then
        EVAL_EXTRA_ARGS="--academic-citation"
    fi

    bash ${REPO_ROOT:-/path/to/repo}/script/eval_launcher.sh \
        --eval-only \
        --task "$TASK" \
        --name "$PASS_EVAL_NAME" \
        --tool-model "$TOOL_MODEL" \
        $EVAL_EXTRA_ARGS

    # Copy eval results back to main dir with pass suffix
    for EVAL_FILE in "$PASS_EVAL_DIR/${TASK}_eval_results.json" "$PASS_EVAL_DIR/dsqa_eval_results.json"; do
        if [ -f "$EVAL_FILE" ]; then
            BASENAME=$(basename "$EVAL_FILE" .json)
            cp "$EVAL_FILE" "$RESULT_DIR/${BASENAME}_pass_${PASS}.json"
            echo "Saved: $RESULT_DIR/${BASENAME}_pass_${PASS}.json"
        fi
    done
done

echo ""
echo "=========================================="
echo "All $NUM_PASSES passes complete for $TASK"
echo "=========================================="
echo "Output structure:"
echo "  $RESULT_DIR/"
for PASS in $(seq 1 $NUM_PASSES); do
    echo "    ${TASK}_pass_${PASS}.jsonl"
    echo "    ${TASK}_*_pass_${PASS}.json"
    echo "    ${TASK}_rubric_bank_pass_${PASS}/"
done
