# coding=utf-8
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

"""Aggregation, bootstrap CIs, and per-axis breakdowns over judge scores.

Reads the per-question records produced by :mod:`financegym.judge.rubric_judge`
and produces:

* :func:`macro` — macro-averaged rate: the mean of per-question rates, so every
  question is weighted equally regardless of its rubric-item count. This is the
  headline metric (matches the published leaderboard).
* :func:`bootstrap_ci` — non-parametric 95% CI on the same rate, with the
  default ``n_boot=1000`` matching the published bootstrap budget.
* :func:`paired_ci` — paired bootstrap on the difference between two agents,
  matched by question text.
* :func:`axis_breakdown` — per-value bins on any axis the judge stamped
  on its records (topic, sector, reasoning_type, situation_type, etc.).
* :func:`score_dist` — aggregate score-frequency distribution.
"""

from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean

DEFAULT_N_BOOT = 1_000
DEFAULT_AXIS_N_BOOT = 400
DEFAULT_SEED = 42

SUM_KEY_TOTAL = "total_sum"
MAX_KEY_TOTAL = "total_max"
SUM_KEY_ANT = "antecedent_sum"
MAX_KEY_ANT = "antecedent_max"
SUM_KEY_CON = "consequent_sum"
MAX_KEY_CON = "consequent_max"


def _ratio(row, sum_key, max_key):
  """One question's normalized rate = score / max, or ``None`` if max==0."""
  m = row.get(max_key, 0)
  return row.get(sum_key, 0) / m if m else None


def macro(
    rows, sum_key = SUM_KEY_TOTAL, max_key = MAX_KEY_TOTAL
):
  """Macro rate = mean of per-question rates (each question weighted equally).

  ``None`` if no question has a defined rate.
  """
  rs = [r for r in (_ratio(x, sum_key, max_key) for x in rows) if r is not None]
  return sum(rs) / len(rs) if rs else None


def bootstrap_ci(
    rows,
    sum_key = SUM_KEY_TOTAL,
    max_key = MAX_KEY_TOTAL,
    *,
    n_boot = DEFAULT_N_BOOT,
    seed = DEFAULT_SEED,
):
  """Non-parametric bootstrap 95% CI on the macro rate (resamples questions)."""
  base = [
      r for r in (_ratio(x, sum_key, max_key) for x in rows) if r is not None
  ]
  if not base:
    return None, None
  rng = random.Random(seed)
  n = len(base)
  means = sorted(
      sum(base[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
  )
  return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def paired_ci(
    rows_a,
    rows_b,
    sum_key = SUM_KEY_TOTAL,
    max_key = MAX_KEY_TOTAL,
    *,
    n_boot = DEFAULT_N_BOOT,
    seed = DEFAULT_SEED,
):
  """Paired bootstrap CI on ``A_rate − B_rate``, matched by question text.

  Returns ``(point_estimate, lo, hi)``. Returns all-``None`` if there are no
  common questions or both arms produce zero max.
  """
  by_q_a = {r["question"]: r for r in rows_a if "question" in r}
  by_q_b = {r["question"]: r for r in rows_b if "question" in r}
  # Keep questions where both agents have a defined per-question rate.
  pairs: list[tuple[float, float]] = []
  for q in sorted(set(by_q_a) & set(by_q_b)):
    ra = _ratio(by_q_a[q], sum_key, max_key)
    rb = _ratio(by_q_b[q], sum_key, max_key)
    if ra is not None and rb is not None:
      pairs.append((ra, rb))
  if not pairs:
    return None, None, None

  n = len(pairs)
  point = sum(a for a, _ in pairs) / n - sum(b for _, b in pairs) / n

  rng = random.Random(seed)
  diffs: list[float] = []
  for _ in range(n_boot):
    idx = [rng.randrange(n) for _ in range(n)]
    da = sum(pairs[i][0] for i in idx) / n
    db = sum(pairs[i][1] for i in idx) / n
    diffs.append(da - db)
  diffs.sort()
  return point, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def axis_breakdown(
    rows,
    axis_key,
    *,
    sum_key = SUM_KEY_TOTAL,
    max_key = MAX_KEY_TOTAL,
    n_boot = DEFAULT_AXIS_N_BOOT,
    seed = DEFAULT_SEED,
):
  """Per-axis-value breakdown: ``{value: {n, rate, lo, hi}}``."""
  by_val: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    v = r.get(axis_key)
    if v is None:
      continue
    by_val[v].append(r)
  out: dict[str, dict] = {}
  for v, grp in by_val.items():
    lo, hi = bootstrap_ci(grp, sum_key, max_key, n_boot=n_boot, seed=seed)
    out[v] = {
        "n": len(grp),
        "rate": macro(grp, sum_key, max_key),
        "lo": lo,
        "hi": hi,
    }
  return out


def score_dist(rows):
  """Aggregate 0..4 frequency counts across every score in every record."""
  agg = {str(k): 0 for k in range(5)}
  for r in rows:
    for k, v in (r.get("score_dist") or {}).items():
      agg[k] = agg.get(k, 0) + v
  return agg


# ---------------------------------------------------------------------------
# Cost / compute helpers
# ---------------------------------------------------------------------------


def cost_stats(answer_rows):
  """Mean/median compute stats from raw answer records (not score records)."""
  if not answer_rows:
    return {}
  times = [r.get("elapsed_s") or 0 for r in answer_rows]
  docs = [r.get("docs_retrieved") or 0 for r in answer_rows]
  chars = [len(r.get("report") or "") for r in answer_rows]
  words = [len((r.get("report") or "").split()) for r in answer_rows]
  steps = [r.get("steps") or 0 for r in answer_rows]
  return {
      "elapsed_s_mean": mean(times),
      "elapsed_s_median": sorted(times)[len(times) // 2],
      "docs_mean": mean(docs),
      "chars_mean": mean(chars),
      "words_mean": mean(words),
      "steps_mean": mean(steps),
  }
