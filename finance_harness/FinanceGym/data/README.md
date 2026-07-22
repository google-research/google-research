# FinanceGym Benchmark Data

## `benchmark_400_public.jsonl`

The public **400-question** point-in-time (PIT) finance deep-research benchmark.
One JSON object per line:

| Field      | Type                  | Description                             |
| ---------- | --------------------- | --------------------------------------- |
| `task_id`  | string                | Stable unique id for the question.      |
| `question` | string                | The research question. Used verbatim to |
:            :                       : match submissions during grading.       :
| `cutoff`   | string (`YYYY-MM-DD`) | The point-in-time date. An agent may    |
:            :                       : only use information available **on or  :
:            :                       : before** this date.                     :

Example line:

```json
{"task_id": "69f904b728538874c086db16", "question": "How does Air France-KLM's accelerated transition toward a 60.5% majority stake in SAS impact its strategic bandwidth ...", "cutoff": "2025-08-05"}
```

### Rubrics are withheld

The scoring rubrics are **not** part of this public set — this prevents score
hacking. Grading is run by maintainers against a private full-rubric set. See
[`../docs/grading.md`](../docs/grading.md).

### How to use it

Download the questions and the PIT search environment, run your agent, and
submit a report: see [`../docs/participate.md`](../docs/participate.md). Current
standings are in [`../docs/leaderboard.md`](../docs/leaderboard.md).
