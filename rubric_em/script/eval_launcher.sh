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
# Single-command: generate + evaluate for ONE benchmark
# Launches vLLM server, runs generation, kills vLLM, then runs evaluation.
# Auto-assigns ports to allow multiple jobs on the same node.
#
# Usage:
#   bash script/eval_launcher.sh --model /path/to/model --task deep_research_bench --gpu 0
#   bash script/eval_launcher.sh --model /path/to/model --task researchqa --gpu 1
#   bash script/eval_launcher.sh --model /path/to/model --task deep_research_bench --name my-run
#   bash script/eval_launcher.sh --eval-only --name my-run --task healthbench  # skip generation
# ==========================================================================

set -e

# ── Defaults ──
GPU=0
TASK="deep_research_bench"
MODEL=""
NAME=""
NUM_EXAMPLES="final_run"
MAX_CONCURRENT=4
TOOL="gemini_grounded_google_search"
PROMPT_VERSION="rubrics_sft_v2"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
MAX_MODEL_LEN=40960
EVAL_ONLY=false
SKIP_EVAL=false
RESAMPLE=true
RUBRIC_BANK_PATH=""
RUBRIC_BANK_EVAL_PATH=""
CROSS_TAKEAWAYS_ONLY=false
DEBUG_PROMPTS=false
NO_CITATION=false
ACADEMIC_CITATION=false
TOOL_MODEL="gemini-3-flash-preview"   # Gemini grounded search model. Override via --tool-model.
USE_STANDARD_INSTRUCTION=false

# DRB evaluation paths
DRB_REPO="${DRB_REPO:-/path/to/deep_research_bench}"
DRB_FORMATTER="${REPO_ROOT:-/path/to/repo}/agent/evaluation/deep_research_bench_eval/drb_formatter.py"
SQA_EVAL="${REPO_ROOT:-/path/to/repo}/agent/evaluation/sqa_eval/run_eval.py"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --tool) TOOL="$2"; shift 2 ;;
        --prompt) PROMPT_VERSION="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --no-resample) RESAMPLE=false; shift ;;
        --rubric-bank) RUBRIC_BANK_PATH="$2"; shift 2 ;;
        --rubric-bank-eval) RUBRIC_BANK_EVAL_PATH="$2"; shift 2 ;;
        --cross-takeaways-only) CROSS_TAKEAWAYS_ONLY=true; shift ;;
        --debug-prompts) DEBUG_PROMPTS=true; shift ;;
        --no-citation) NO_CITATION=true; shift ;;
        --academic-citation) ACADEMIC_CITATION=true; shift ;;
        --tool-model) TOOL_MODEL="$2"; shift 2 ;;
        --use-standard-instruction) USE_STANDARD_INSTRUCTION=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$EVAL_ONLY" = false ] && [ -z "$MODEL" ]; then
    echo "Error: --model is required (unless using --eval-only)"
    exit 1
fi

if [ -z "$NAME" ]; then
    if [ -n "$MODEL" ]; then
        NAME=$(basename "$MODEL")
    else
        echo "Error: --name is required with --eval-only"
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
RESULT_DIR="$OUTPUT_DIR/$NAME"
# Add num_examples suffix to filename when not using full dataset
if [ "$NUM_EXAMPLES" != "final_run" ] && [ -n "$NUM_EXAMPLES" ]; then
    JSONL="$RESULT_DIR/${TASK}_n${NUM_EXAMPLES}.jsonl"
else
    JSONL="$RESULT_DIR/$TASK.jsonl"
fi

echo "=========================================="
echo "Evaluation Pipeline"
echo "  Model:     $MODEL"
echo "  Task:      $TASK"
echo "  GPU:       $GPU"
echo "  Name:      $NAME"
echo "  Output:    $JSONL"
echo "  Tool model:$TOOL_MODEL"
echo "  Eval-only: $EVAL_ONLY"
echo "  Skip-eval: $SKIP_EVAL"
echo "=========================================="

# ══════════════════════════════════════════════
# PHASE 1: GENERATION
# ══════════════════════════════════════════════
if [ "$EVAL_ONLY" = false ]; then
    # ── Auto-assign ports (based on GPU to avoid collisions) ──
    VLLM_PORT=$((30001 + GPU))
    MCP_PORT=$((8000 + GPU))

    # ── Environment ──
    export S2_API_KEY="${S2_API_KEY:-}"
    cd ${REPO_ROOT:-/path/to/repo}/agent

    export LLM_MAX_CONCURRENT_CALLS=$MAX_CONCURRENT

    # ── Cleanup on exit ──
    VLLM_PID=""
    MCP_PID=""
    cleanup() {
        echo ""
        echo "Cleaning up..."
        [ -n "$MCP_PID" ] && kill $MCP_PID 2>/dev/null && echo "Stopped MCP server (PID $MCP_PID)"
        [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null && echo "Stopped vLLM server (PID $VLLM_PID)"
        wait 2>/dev/null
    }
    trap cleanup EXIT INT TERM

    # ── Launch MCP server (GPU-specific port, avoids conflicts between parallel runs) ──
    echo "Starting MCP server on port $MCP_PORT..."
    source ${CONDA_INIT:-/path/to/conda_init.sh}
    conda activate dr_agent
    python -m dr_agent.mcp_backend.main \
        --transport http \
        --port $MCP_PORT \
        --host 0.0.0.0 \
        --path /mcp \
        > /tmp/mcp_eval_gpu${GPU}.log 2>&1 &
    MCP_PID=$!

    # Wait for MCP to be ready
    for i in $(seq 1 120); do
        if curl -s http://localhost:$MCP_PORT/mcp > /dev/null 2>&1; then
            echo "MCP server ready (took ${i}s)"
            break
        fi
        if ! kill -0 $MCP_PID 2>/dev/null; then
            echo "Warning: MCP server failed to start. Continuing without it."
            MCP_PID=""
            break
        fi
        sleep 1
    done

    # ── Launch vLLM server ──
    echo "Starting vLLM server on GPU $GPU, port $VLLM_PORT..."
    CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" \
        --dtype auto \
        --port $VLLM_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --enable-prefix-caching \
        > /tmp/vllm_eval_gpu${GPU}.log 2>&1 &
    VLLM_PID=$!

    # Wait for vLLM to be ready
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 600); do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            echo "vLLM server ready (took ${i}s)"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "Error: vLLM server failed to start. Check /tmp/vllm_eval_gpu${GPU}.log"
            exit 1
        fi
        sleep 1
    done

    if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "Error: vLLM server did not start within 600s"
        exit 1
    fi

    # ── Run generation ──
    echo "Starting generation: $TASK"

    python workflows/auto_search_sft.py \
        generate-dataset $TASK \
        --num-examples $NUM_EXAMPLES \
        --max-concurrent $MAX_CONCURRENT \
        --use-cache \
        --config workflows/auto_search_sft.yaml \
        --config-overrides "use_browse_agent=false,search_agent_max_tool_calls=10,search_tool_name=$TOOL,prompt_version=$PROMPT_VERSION,tool_api=,tool_model_name=$TOOL_MODEL,search_agent_model_name=$MODEL,search_agent_base_url=http://localhost:${VLLM_PORT}/v1,mcp_port=$MCP_PORT,search_agent_resample=$RESAMPLE${RUBRIC_BANK_PATH:+,rubric_bank_path=$RUBRIC_BANK_PATH}${RUBRIC_BANK_EVAL_PATH:+,rubric_bank_eval_path=$RUBRIC_BANK_EVAL_PATH}$([ "$CROSS_TAKEAWAYS_ONLY" = true ] && echo ',rubric_bank_cross_takeaways_only=true')$([ "$DEBUG_PROMPTS" = true ] && echo ',rubric_bank_debug_prompts=true')$([ "$USE_STANDARD_INSTRUCTION" = true ] && echo ',use_standard_instruction=true')" \
        --output "$JSONL"

    echo "Generation complete: $JSONL"

    # Kill servers — no longer needed for evaluation
    kill $VLLM_PID 2>/dev/null || true; VLLM_PID=""
    kill $MCP_PID 2>/dev/null || true; MCP_PID=""
    echo "Servers stopped (freeing resources for evaluation)"
fi

# ══════════════════════════════════════════════
# PHASE 2: EVALUATION
# ══════════════════════════════════════════════
if [ "$SKIP_EVAL" = true ]; then
    echo "Skipping evaluation (--skip-eval)"
    echo "Output: $JSONL"
    exit 0
fi

if [ ! -f "$JSONL" ]; then
    echo "Error: $JSONL not found. Run generation first."
    exit 1
fi

# ── Citation conversion ──
if [ "$NO_CITATION" = true ]; then
    JSONL_NOCITE="${JSONL%.jsonl}_no_citation.jsonl"
    echo "Stripping citations: $JSONL -> $JSONL_NOCITE"
    python ${REPO_ROOT:-/path/to/repo}/script/utils/strip_citations.py "$JSONL" "$JSONL_NOCITE"
    JSONL="$JSONL_NOCITE"
elif [ "$ACADEMIC_CITATION" = true ]; then
    JSONL_ACAD="${JSONL%.jsonl}_academic_cite.jsonl"
    echo "Converting to academic citations: $JSONL -> $JSONL_ACAD"
    python ${REPO_ROOT:-/path/to/repo}/script/utils/convert_citations_academic.py "$JSONL" "$JSONL_ACAD"
    JSONL="$JSONL_ACAD"
fi

echo ""
echo "=========================================="
echo "Running evaluation: $TASK"
echo "=========================================="

# API keys for DRB evaluation (RACE + FACT scoring via Gemini)
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export JINA_API_KEY="${JINA_API_KEY:-}"

cd ${REPO_ROOT:-/path/to/repo}/agent

case $TASK in
    deep_research_bench)
        # DRB: format → RACE + Citation evaluation
        echo "Step 1: Formatting for DRB..."
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        python "$DRB_FORMATTER" \
            --input_file_path "$JSONL" \
            --task_name "$NAME" \
            --drb_repo_path "$DRB_REPO"

        echo "Step 2: Running DRB benchmark (RACE + Citation)..."
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate drb
        export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
        export JINA_API_KEY="${JINA_API_KEY:-}"
        cd "$DRB_REPO"
        bash run_benchmark_scraped.sh "$NAME"
        cd ${REPO_ROOT:-/path/to/repo}/agent
        echo "DRB results: $DRB_REPO/results/"
        ;;

    healthbench)
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        python scripts/evaluate.py healthbench "$JSONL"
        echo "Results: ${JSONL%.jsonl}_eval_results.json"
        ;;

    sqav2)
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
        export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
        export HF_TOKENS=""
        export TMPDIR="/tmp"
        mkdir -p "$TMPDIR"
        SQA_OUTPUT_DIR="$RESULT_DIR/sqa_eval_logs"
        python "$SQA_EVAL" run --input_file "$JSONL" --output_dir "$SQA_OUTPUT_DIR" 2>&1 | tee /tmp/sqa_eval_output.txt
        # Save scores to JSON for consistency with other benchmarks
        python3 -c "
import re, json
text = open('/tmp/sqa_eval_output.txt').read()
metrics = {}
for m in re.finditer(r'(\w+/\w+):\s+([\d.]+)', text):
    metrics[m.group(1)] = float(m.group(2))
if metrics:
    out = '${JSONL%.jsonl}_eval_results.json'
    json.dump({'task': 'sqav2', 'metrics': metrics, 'score': metrics.get('global_avg/mean', 0)}, open(out, 'w'), indent=2)
    print(f'Saved SQA results to {out}')
"
        echo "Results: ${JSONL%.jsonl}_eval_results.json"
        ;;

    researchqa)
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        python scripts/evaluate.py researchqa "$JSONL"
        echo "Results: ${JSONL%.jsonl}_eval_results.json"
        ;;

    dsqa)
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        DSQA_EVAL="${DSQA_REPO:-/path/to/deepsearchqa}/eval_dsqa.py"
        python "$DSQA_EVAL" \
            --mode evaluate \
            --input "$JSONL" \
            --output-dir "$RESULT_DIR" \
            --name "$NAME" \
            --max-concurrent 15
        echo "Results: $RESULT_DIR/${NAME}_results.json"
        ;;

    researchrubrics)
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        python ${REPO_ROOT:-/path/to/repo}/script/utils/eval_researchrubrics.py \
            --input "$JSONL" \
            --output-dir "$RESULT_DIR" \
            --max-concurrent 5
        echo "Results: $RESULT_DIR/researchrubrics_eval_results.json"
        ;;

    simpleqa|2wiki|webwalker)
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        python scripts/evaluate.py "$TASK" "$JSONL"
        echo "Results: ${JSONL%.jsonl}_eval_results.json"
        ;;

    *)
        echo "Unknown task: $TASK, trying generic evaluation..."
        source ${CONDA_INIT:-/path/to/conda_init.sh}
        conda activate dr_agent
        python scripts/evaluate.py "$TASK" "$JSONL"
        ;;
esac

echo ""
echo "=========================================="
echo "Pipeline complete: $TASK"
echo "Output: $JSONL"
echo "=========================================="
