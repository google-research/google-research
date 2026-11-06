# Example Eval Trajectories & Viewer

Reference trajectories produced by **RubricEM-8B** (final, step-1400) on the
evaluation benchmarks, plus a lightweight browser viewer to inspect them.

## Contents

-   `trajectory_viewer.html` — a **self-contained** (no dependencies) trajectory
    viewer. Open it in any browser and load a `*.jsonl` file; it renders each
    example's query, the full scaffold trajectory (plan → research → review →
    answer, color-coded by tag), the tool calls (search queries, runtimes, raw
    outputs), and the final answer. Navigate with the buttons or the ← / → keys.
-   Example trajectories are not included in this release (a data release may
    follow separately); small enough to browse directly:
    -   `deep_research_bench_sample.jsonl`
    -   `healthbench_sample.jsonl`
    -   `researchqa_sample.jsonl`
    -   `2wiki_sample.jsonl`

## Usage

```bash
# open the viewer, then click "Choose file" and pick an answers.jsonl from your own eval run
open eval_trajectories/trajectory_viewer.html          # macOS
xdg-open eval_trajectories/trajectory_viewer.html      # Linux
```

## Trajectory schema (`answers.jsonl`)

Each line is one example:

| field                        | meaning                                      |
| ---------------------------- | -------------------------------------------- |
| `example_id`                 | index                                        |
| `problem`                    | the query                                    |
| `final_response`             | the model's final answer (clean)             |
| `full_traces.generated_text` | the full raw trajectory with scaffold tags   |
:                              : (`<think>`, `<structured_plan>`, `<review>`, :
:                              : `<answer>`, tool I/O)                        :
| `full_traces.tool_calls[]`   | per-call `tool_name`, `query`, `runtime`,    |
:                              : `raw_output`, `documents`, …                 :
| `original_data`              | source item (`problem`,                      |
:                              : `additional_instructions`, `dataset_name`)   :

Full eval logs for all 8 benchmarks (answers + judge results) are released
alongside the model — see the main README's *Model Checkpoints* section.
