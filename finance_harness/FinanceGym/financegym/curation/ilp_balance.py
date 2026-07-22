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

"""ILP-balanced subset selection across topic × sector × reasoning × month.

The selection step turns a quality-filtered pool (the output of the
feasibility / relevance / coherence stages) into a balanced evaluation
subset. The ILP maximizes total rubric-item count (more scoring signal
per question) subject to per-axis fraction bounds.

Used twice in the documented funnel:

* ``select_balanced(target=1000)`` — a curated training pool.
* ``select_balanced(target=500)``  — the published ILP-balanced eval set.

Both call the same :func:`select_balanced` solver; only the bounds change.
"""

from __future__ import annotations

from collections import Counter
import math

import pulp

AXES = ("topic", "sector", "reasoning_type", "_month")


def month_bucket(cutoff):
  return cutoff[:7] if cutoff else "unknown"


def prepare(questions):
  """Keep questions with all three axis labels + a cutoff; stamp ``_month``."""
  out: list[dict] = []
  for q in questions:
    if not (
        q.get("topic")
        and q.get("sector")
        and q.get("reasoning_type")
        and q.get("cutoff")
    ):
      continue
    q = dict(q)
    q["_month"] = month_bucket(q["cutoff"])
    out.append(q)
  return out


def _bounds(
    pool_counts,
    target,
    *,
    min_pct,
    max_pct,
):
  out: dict[str, tuple[int, int]] = {}
  for k, cnt in pool_counts.items():
    lo = max(int(math.floor(target * min_pct)), 1)
    hi = int(math.ceil(target * max_pct))
    lo = min(lo, cnt)
    hi = min(hi, cnt)
    if lo > hi:
      lo = hi
    out[k] = (lo, hi)
  return out


def select_balanced(
    pool,
    *,
    target,
    topic_min_pct = 0.04,
    topic_max_pct = 0.20,
    sector_min_pct = 0.01,
    sector_max_pct = 0.22,
    reason_min_pct = 0.05,
    reason_max_pct = 0.35,
    month_min_pct = 0.01,
    month_max_pct = 0.15,
    solver_msg = False,
):
  """Solve the balanced ILP and return the selected subset."""
  pool = prepare(pool)
  if len(pool) < target:
    raise ValueError(f"pool ({len(pool)}) smaller than target ({target})")

  topic_pool = Counter(q["topic"] for q in pool)
  sector_pool = Counter(q["sector"] for q in pool)
  reason_pool = Counter(q["reasoning_type"] for q in pool)
  month_pool = Counter(q["_month"] for q in pool)

  topic_b = _bounds(
      topic_pool, target, min_pct=topic_min_pct, max_pct=topic_max_pct
  )
  sector_b = _bounds(
      sector_pool, target, min_pct=sector_min_pct, max_pct=sector_max_pct
  )
  reason_b = _bounds(
      reason_pool, target, min_pct=reason_min_pct, max_pct=reason_max_pct
  )
  month_b = _bounds(
      month_pool, target, min_pct=month_min_pct, max_pct=month_max_pct
  )

  def _weight(q):
    return float(len(q.get("rubric", [])))

  prob = pulp.LpProblem("FinanceGymBalance", pulp.LpMaximize)
  x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(pool))]
  prob += pulp.lpSum(_weight(pool[i]) * x[i] for i in range(len(pool)))
  prob += pulp.lpSum(x) == target

  for axis_key, bounds in (
      ("topic", topic_b),
      ("sector", sector_b),
      ("reasoning_type", reason_b),
      ("_month", month_b),
  ):
    for v, (lo, hi) in bounds.items():
      idx = [i for i, q in enumerate(pool) if q[axis_key] == v]
      prob += pulp.lpSum(x[i] for i in idx) >= lo
      prob += pulp.lpSum(x[i] for i in idx) <= hi

  prob.solve(pulp.PULP_CBC_CMD(msg=int(solver_msg)))
  if prob.status != 1:
    raise RuntimeError(
        f"ILP infeasible: {pulp.LpStatus[prob.status]} — relax bounds"
    )

  return [
      pool[i] for i in range(len(pool)) if x[i].varValue and x[i].varValue > 0.5
  ]


# ---------------------------------------------------------------------------
# Balance reporting
# ---------------------------------------------------------------------------


def entropy(counts):
  total = sum(counts.values())
  if total == 0:
    return 0.0
  probs = [c / total for c in counts.values() if c > 0]
  return -sum(p * math.log2(p) for p in probs)


def balance_score(counts):
  """Normalized Shannon entropy: 1.0 when perfectly uniform."""
  n_nonzero = len([c for c in counts.values() if c > 0])
  if n_nonzero <= 1:
    return 1.0
  return entropy(counts) / math.log2(n_nonzero)


def report(selected):
  """Return a balance summary for a selected subset (matches build_eval500.report)."""
  if not selected:
    return {"total": 0}
  topic = Counter(q["topic"] for q in selected)
  sector = Counter(q["sector"] for q in selected)
  reason = Counter(q["reasoning_type"] for q in selected)
  month = Counter(
      q.get("_month") or month_bucket(q.get("cutoff", "")) for q in selected
  )
  cross = {(q["topic"], q["sector"]) for q in selected}
  possible = len(topic) * len(sector) or 1
  return {
      "total": len(selected),
      "topic_distribution": dict(topic),
      "sector_distribution": dict(sector),
      "reasoning_distribution": dict(reason),
      "month_distribution": dict(month),
      "topic_balance": round(balance_score(topic), 3),
      "sector_balance": round(balance_score(sector), 3),
      "reasoning_balance": round(balance_score(reason), 3),
      "month_balance": round(balance_score(month), 3),
      "cross_coverage": round(len(cross) / possible, 3),
  }
