# DatasetAgent of TimesX

This DatasetAgent builds web-grounded textual context for multimodal time-series forecasting evaluations.

Paper: Rethinking Multimodal Time-Series Forecasting Evaluation, ICML 2026

## Install

All commands below assume you are executing from the `TimesX` repository root directory.

```bash
cd TimesX
```

Recommended with uv:

```bash
uv sync
```

Fallback with pip:

```bash
python3 -m venv /tmp/dataset_agent_venv
source /tmp/dataset_agent_venv/bin/activate
pip install -r dataset_agent/requirements.txt
python -m playwright install chromium
```

## Configure API keys

Copy `dataset_agent/env.example` and export the required credentials:

```bash
export CUSTOMSEARCH_API_KEY="your_google_custom_search_api_key"
export CUSTOMSEARCH_CX_KEY="your_google_custom_search_engine_id"
export GEMINI_API_KEY="your_gemini_api_key"
```

`CUSTOMSEARCH_API_KEY` is for Google Custom Search. `GEMINI_API_KEY` is for Gemini LLM calls. 
## Configure a run

Start from `dataset_agent/configs/smoke_1.yaml` and edit:

- `keywords_to_process`: variables or search topics to process.
- `target_variable`: usually the first keyword for a single-run config.
- `time_range.start` and `time_range.end`: target date window.
- `geography_str` and `geography`: geographic scope.
- `llm_models.*`: Gemini model names for each LLM role.

Useful runtime controls:

- `event_sourcing.initial_event_iterations`: initial event discovery rounds.
- `event_sourcing.max_initial_events`: maximum initial events before validation.
- `event_sourcing.iterative_crawl_count`: evidence collection depth.
- `task_execution.validation_tasks.query_budget_per_event`: follow-up search budget per event.
- `task_execution.validation_tasks.crawl_count_per_query`: crawled sources per search query.
- `paths.output_root`: optional output root; can also be overridden by `--output-root`.

## Run

Validate config without network access:

```bash
uv run python -m dataset_agent.orchestrator \
  --config dataset_agent/configs/smoke_1.yaml \
  --validate-config-only
```

Or without uv, after installing dependencies:

```bash
python3 -m dataset_agent.orchestrator \
  --config dataset_agent/configs/smoke_1.yaml \
  --validate-config-only
```

Run the default lightweight check:

```bash
uv run bash dataset_agent/run.sh
```

Run a live smoke with outputs outside the source tree:

```bash
uv run python -m dataset_agent.orchestrator \
  --config dataset_agent/configs/smoke_1.yaml \
  --output-root /tmp/dataset_agent_smoke
```

Live runs depend on external provider quota and search results. Verified event counts are not deterministic.

## Outputs

With `--output-root /tmp/dataset_agent_smoke`, outputs are written under:

```text
/tmp/dataset_agent_smoke/<project_name>/<domain>/data/output/<keyword>/<start>_to_<end>/
```

Typical files include `final_dataset.json`, `final_dataset.csv`, `run.log`.

## Layout

- `orchestrator.py`: command-line entrypoint.
- `event_pipeline.py`: active event-centric pipeline.
- `configs/`: runnable YAML examples.
- `modules/`: implementation modules for trends, event discovery, crawling, evidence cleaning, LLM calls, validation, and runtime state.
- `run.sh`: minimal config-validation runner.
