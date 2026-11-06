# Evaluation Scripts

This directory contains scripts for evaluating models on various benchmarks.

## Prerequisites

### Required Environment Variables

Set the following environment variables before running evaluation scripts:

```bash
export GEMINI_API_KEY="your-gemini-api-key"     # Required: used for the Gemini grounded search tool and LLM-as-judge evaluation
```

### Optional Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"     # Optional: needed for some baselines that use OpenAI models
export S2_API_KEY="your-s2-api-key"             # Optional: for Semantic Scholar search (snippet_search tool)
export JINA_API_KEY="your-jina-api-key"         # Optional: for Jina webpage browsing tool
export SERPER_API_KEY="your-serper-api-key"      # Optional: for Serper web search tool
```

### Repository Setup

Set `REPO_ROOT` to point to this repository's root directory:

```bash
export REPO_ROOT="/path/to/this/repo"
```

## Scripts Overview

### Main Evaluation Launcher

-   `eval_launcher.sh` -- Unified evaluation pipeline that handles model
    serving, generation, and evaluation for all benchmarks.

### RL Model Evaluations

-   `rl/eval_baseline.sh` -- Evaluate baseline RL checkpoints
-   `rl/eval_credit_assignment.sh` -- Evaluate credit assignment RL checkpoints
-   `rl/eval_rubricem.sh` -- Evaluate RL checkpoints

### SFT Model Evaluation

-   `sft/eval_sft.sh` -- Evaluate SFT checkpoints

### Utilities

-   `utils/eval_researchrubrics.py` -- ResearchRubrics benchmark evaluation
    pipeline
-   `utils/strip_citations.py` -- Strip citation tags from generated responses
-   `utils/convert_citations_academic.py` -- Convert inline citations to
    academic format

## Supported Benchmarks

-   `deep_research_bench` -- Deep Research Bench (RACE + Citation evaluation)
-   `healthbench` -- Medical question answering
-   `sqav2` -- Scientific Question Answering v2
-   `researchqa` -- Research question answering
-   `dsqa` -- Deep Search QA
-   `researchrubrics` -- Rubric-based research report evaluation
-   `simpleqa` -- Simple factual QA
-   `2wiki` -- Two-hop Wikipedia QA
-   `webwalker` -- Web navigation QA

## Example Usage

```bash
# Evaluate an RL checkpoint on deep_research_bench
bash script/eval_launcher.sh \
    --model /path/to/checkpoint \
    --task deep_research_bench \
    --gpu 0 \
    --name my-experiment

# Evaluate with rubric bank
bash script/rl/eval_rubricem.sh \
    --step 100 \
    --task deep_research_bench \
    --gpu 0
```
