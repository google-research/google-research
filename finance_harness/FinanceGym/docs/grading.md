# Grading & Leaderboard Update (Maintainers)

> **Audience:** maintainers. Grading uses the **withheld full-rubric set** (`benchmark_400.jsonl`),
> which is **not** in this public repo. Participants never see rubrics.

Grading is fully reproducible: the same submission always yields the same score. It runs in three
steps — **judge → aggregate → update the leaderboard**.

> **Verified.** These commands are exercised by `tests/test_rubric_judge.py`,
> `tests/test_aggregate.py`, and `tests/test_quality_judge.py`, and were run live end-to-end
> (`scripts/run_judge.sh` → `scores.jsonl` → aggregate). Expect ~30 s per question for the judge call.

## Prerequisites

```bash
pip install -e ".[dev]"        # judge + aggregate deps
export GOOGLE_API_KEY=...       # Gemini API key for the judge (GEMINI_API_KEY is also honored)
```

## Inputs

| Input | Source | Public? |
|-------|--------|---------|
| `answers.jsonl` | The submission issue or email — see [participate.md](participate.md). | yes |
| `benchmark_400.jsonl` | Maintainer-only full-rubric benchmark. | **no** |

## Step 1 — Judge

`scripts/run_judge.sh` iterates the submission, matches each answer to its question by `question`
text, calls `judge_pair_to_record`, and writes one score record per question.

```bash
QUESTIONS=benchmark_400.jsonl \
ANSWERS=answers/<agent>.jsonl \
AGENT=<agent> \
SCORES_OUT=scores/<agent>.jsonl \
bash scripts/run_judge.sh
```

The judge (`financegym/judge/rubric_judge.py`) scores every rubric item on a **0–4** scale:

| Score | Meaning |
|------:|---------|
| 0 | NOT ADDRESSED |
| 1 | MENTIONED (no substance) |
| 2 | PARTIAL (directionally correct, missing specifics) |
| 3 | SUBSTANTIVE (correct with specifics, minor gaps) |
| 4 | FULLY GROUNDED (correct, specific, plausibly sourced) |

Each `scores/<agent>.jsonl` record carries per-item scores plus the aggregate sums used next:
`total_sum`/`total_max`, `antecedent_sum`/`antecedent_max` (Hindsight),
`consequent_sum`/`consequent_max` (Foresight), and axis labels (`topic`, `sector`,
`reasoning_type`, `situation_type`).

## Step 2 — Aggregate

Use `financegym.judge.aggregate` to turn the per-question records into a leaderboard row.
The headline metric is **macro-averaged**: each question's normalized rate is `s / (4n)`
(score over max, `n` rubric items), and the score is the **mean of those per-question rates**,
so every question is weighted equally. Rates are fractions in `[0, 1]`; multiply by 100 for the
leaderboard's %.

```python
import json
from financegym.judge.aggregate import macro, bootstrap_ci, axis_breakdown

rows = [json.loads(l) for l in open("scores/<agent>.jsonl")]

overall   = macro(rows)                                        # mean of per-question total_norm
hindsight = macro(rows, "antecedent_sum", "antecedent_max")
foresight = macro(rows, "consequent_sum", "consequent_max")
lo, hi    = bootstrap_ci(rows)                                 # 95% CI, seed 42, 1000 iters
by_sector = axis_breakdown(rows, "sector")                     # optional per-axis breakdown

print(f"Overall   {overall*100:.1f}%  (95% CI {lo*100:.1f}–{hi*100:.1f})")
print(f"Hindsight {hindsight*100:.1f}%   Foresight {foresight*100:.1f}%")
```

The leaderboard row is: **Overall** (`macro`), **Hindsight** (antecedent macro), **Foresight**
(consequent macro), and the 95% bootstrap CI over the same macro mean.

## Step 3 — Update the leaderboard

The leaderboard lives in this repo at [`leaderboard.md`](leaderboard.md). Add or replace the agent's
row in both the Overall table and its category panel (values ×100 for %), re-rank by Overall, then
commit and push:

```bash
# edit docs/leaderboard.md — insert the new row, re-sort by Overall
git add docs/leaderboard.md
git commit -m "leaderboard: add <agent> (Overall XX.X)"
```

Close the submission issue with a link to the updated leaderboard row.

## Reproducibility pins

| Item | Value |
|------|-------|
| Judge model | configurable via `FINANCEGYM_MODEL` (a current Gemini model) |
| Judge system prompt | `financegym/judge/rubric_judge.py` → `JUDGE_SYSTEM` |
| Report char cap | 12,000 |
| Rubric scale | 0–4 per item |
| Bootstrap | seed `42`, `1000` iterations (`DEFAULT_N_BOOT`) |
| Metric | macro-averaged: mean of per-question rates `s / (4n)` |

See [`reproducibility.md`](reproducibility.md) for the full pinned configuration.
