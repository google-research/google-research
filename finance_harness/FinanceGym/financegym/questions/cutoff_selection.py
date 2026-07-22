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

"""Per-month cutoff selection: pick the K best days to anchor a question on.

Each day is scored as ``volume_z × (1 + entity_diversity) × (1 +
relation_entropy / 10)``.
We then pick the top K within each month (with K varying by how balanced the
month is between pre- and post-cutoff data), enforce a minimum cross-month gap,
and return the day-before-event as the published cutoff so that the event itself
becomes consequent evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from math import log2


# Cutoff budget per month, indexed by the pre/post-data balance score.
# Months with the most data on both sides get the most cutoffs.
def cutoff_budget(balance):
  """Map the pre/post balance ratio to a K (number of cutoffs that month)."""
  if balance >= 0.85:
    return 5
  if balance >= 0.6:
    return 4
  if balance >= 0.3:
    return 3
  return 2


def score_day(
    count,
    entity_count,
    rel_counts,
    *,
    mean_volume,
    std_volume,
):
  """Return the day's importance score.

  The three factors are designed to penalize:

  * low-volume days (volume z-score)
  * days dominated by a single entity (entity diversity per edge)
  * days dominated by one relation type (relation entropy)
  """
  volume_z = (count - mean_volume) / std_volume if std_volume > 0 else 0.0
  entity_diversity = entity_count / max(count, 1)
  if rel_counts and len(rel_counts) > 1:
    total = sum(rel_counts.values())
    probs = [c / total for c in rel_counts.values()]
    rel_entropy = -sum(p * log2(p) for p in probs if p > 0)
  else:
    rel_entropy = 0.0
  return volume_z * (1 + entity_diversity) * (1 + rel_entropy / 10)


def select_cutoffs_monthly(
    edges,
    *,
    min_date = "2025-01-01",
    max_date = "2026-01-01",
    min_cross_gap_days = 7,
    within_month_gap_days = 3,
):
  """Return cutoff dates selected algorithmically across the date range."""
  daily_counts: dict[str, int] = defaultdict(int)
  daily_entities: dict[str, set[str]] = defaultdict(set)
  daily_rel_counts: dict[str, Counter] = defaultdict(Counter)

  for e in edges:
    day = e.get("pub_date", "")[:10]
    if not (min_date <= day < max_date):
      continue
    daily_counts[day] += 1
    daily_entities[day].add(e["head"])
    daily_entities[day].add(e["tail"])
    daily_rel_counts[day][e["relation"]] += 1

  if not daily_counts:
    return []

  values = list(daily_counts.values())
  mean = sum(values) / len(values)
  std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

  scores: dict[str, float] = {}
  for day, count in daily_counts.items():
    scores[day] = score_day(
        count,
        len(daily_entities[day]),
        daily_rel_counts[day],
        mean_volume=mean,
        std_volume=std,
    )

  # Per-month cutoff budgets based on how balanced the pre/post split is.
  monthly_edges: dict[str, int] = defaultdict(int)
  for day, count in daily_counts.items():
    monthly_edges[day[:7]] += count
  total_edges = sum(monthly_edges.values()) or 1

  cumulative = 0
  month_k: dict[str, int] = {}
  for month in sorted(monthly_edges):
    mid = cumulative + monthly_edges[month] / 2
    pre_ratio = mid / total_edges
    post_ratio = 1 - pre_ratio
    denom = max(pre_ratio, post_ratio) or 1
    balance = min(pre_ratio, post_ratio) / denom
    month_k[month] = cutoff_budget(balance)
    cumulative += monthly_edges[month]

  monthly_best: dict[str, list[tuple[str, float, int, int]]] = defaultdict(list)
  for day, score in scores.items():
    monthly_best[day[:7]].append(
        (day, score, daily_counts[day], len(daily_entities[day]))
    )
  for month in monthly_best:
    monthly_best[month].sort(key=lambda x: -x[1])

  cutoffs: list[str] = []
  selected: list[datetime] = []

  for month in sorted(monthly_best):
    budget = month_k.get(month, 2)
    n_in_month = 0
    for event_day, score, _count, _n_ents in monthly_best[month]:
      if n_in_month >= budget:
        break
      # Skip negative-score days unless we have nothing better available.
      if score <= 0 and n_in_month > 0:
        continue
      if score <= 0 and any(s > 0 for _, s, _, _ in monthly_best[month]):
        continue

      ev_dt = datetime.strptime(event_day, "%Y-%m-%d")
      too_close = False
      for prev in selected:
        gap = abs((ev_dt - prev).days)
        same_month = prev.strftime("%Y-%m") == ev_dt.strftime("%Y-%m")
        if same_month and gap < within_month_gap_days:
          too_close = True
          break
        if not same_month and gap < min_cross_gap_days:
          too_close = True
          break
      if too_close:
        continue

      cutoffs.append((ev_dt - timedelta(days=1)).strftime("%Y-%m-%d"))
      selected.append(ev_dt)
      n_in_month += 1

  return cutoffs
