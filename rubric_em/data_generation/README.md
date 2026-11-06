# Step-Wise SFT Data Generation Pipeline

This pipeline generates structured SFT training data through iterative,
step-wise prompting of a teacher model (Gemini). Unlike single-pass generation,
this approach mirrors the agent's actual inference-time behavior: the model
generates one action per turn, receives real tool outputs, and continues
reasoning based on those results.

## Why Step-Wise Generation?

**Single-pass generation** (prompting a teacher model to produce an entire
multi-turn trajectory at once) has fundamental limitations for agentic SFT data:

1.  **Tool output fidelity:** The teacher model must hallucinate tool outputs
    (search results, API responses) since it cannot actually execute tools
    during generation. This creates a distribution mismatch between training
    data and real inference.

2.  **Trajectory coherence:** Real agent trajectories are *reactive* — each
    action depends on the actual tool output from the previous step. Single-pass
    trajectories cannot capture this dependency because the model generates the
    entire sequence without feedback.

3.  **Search result freshness:** Hallucinated search results are often outdated
    or fabricated. Step-wise generation uses live tool calls, producing training
    data grounded in real, current information.

**Step-wise generation** solves these by decomposing trajectory generation into
rounds:

```
[Round N]
  1. Teacher model generates next action (search query, answer, etc.)
  2. Parse model output to extract tool calls
  3. Execute real tool calls (Google Search, Semantic Scholar)
  4. Parse tool outputs and format as context for next round
  → [Round N+1]
```

Each round produces one turn of the conversation. The teacher sees real tool
outputs before deciding its next action, producing trajectories that faithfully
represent actual agent behavior.

## Pipeline Architecture

```
auto_pipeline.py              # Orchestrator: runs rounds sequentially
├── model_call_stage/          # Step 1: Teacher model generation
│   ├── prompt.py              # System prompts (first-round and later-round)
│   ├── model_response_*.py    # Batch model calling (submit/download/process)
├── prompt_processing_stage/   # Steps 2 & 4: Parse between stages
│   ├── processing_model_output_to_tool_input.py
│   └── processing_tool_output_to_next_round_input.py
├── tool_call_stage/           # Step 3: Execute real tool calls
│   ├── semantic_scholar_asyn_processing.py  # Async S2 API calls
│   └── tool_batch_submit.py, tool_batch_download.py  # Google Search via batch API
└── process_output_files/      # Final: Convert completed trajectories to SFT format
    └── convert_benchmark_data.py
```

## Usage

### Full automated pipeline

```bash
bash auto_pipeline.sh \
    --start_round 1 \
    --end_round 10 \
    --run_id "my_sft_run" \
    --part 1 \
    --model_name "gemini-3.1-pro-preview" \
    --prompt "scratchpad"
```

The pipeline automatically:

-   Runs rounds 1 through 10 (or until all trajectories complete with
    `<answer>`)
-   Executes Google Search and Semantic Scholar tool calls in parallel
-   Early-stops when no more data needs processing

### Step-by-step execution

For finer control, run individual stages:

```bash
# Round 1: Generate initial model responses
bash scripts/run_pipeline.sh model_response --round 1 --run_id "my_run" --part 1

# Parse model output to extract tool calls
bash scripts/run_pipeline.sh parsing_model_to_tool --round 1 --run_id "my_run"

# Execute tool calls (run in parallel)
bash scripts/run_pipeline.sh semantic_scholar --round 1 --run_id "my_run" --part 1
bash scripts/run_pipeline.sh google_search --round 1 --run_id "my_run" --part 1

# Parse tool outputs for next round
bash scripts/run_pipeline.sh parsing_tool_to_input --round 1 --run_id "my_run"

# Continue to round 2...
bash scripts/run_pipeline.sh model_response --round 2 --run_id "my_run" --part 1
```

### Convert to SFT format

After generation completes:

```bash
python process_output_files/convert_benchmark_data.py --run_id "my_run"
```

## Required Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="your-project"   # For Gemini batch API
export GEMINI_API_KEY="..."                   # For Gemini model calls
export S2_API_KEY="..."                       # Optional: Semantic Scholar (higher rate limits)
```

## Prompt Versions

The `--prompt` flag selects the teacher prompt version:

-   `scratchpad` — Uses `<scratchpad>` for internal reasoning (converted to
    `<think>` in post-processing)
-   `action_only_scratchpad` — Minimal reasoning, action-focused
-   `plain_iterative` — Unstructured iterative prompt (ablation baseline)

See `model_call_stage/prompt.py` for full prompt definitions.
