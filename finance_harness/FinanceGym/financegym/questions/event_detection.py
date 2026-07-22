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

"""Per-day edge indexes + high-density event-day detection.

Two functions:

* :func:`build_indexes` walks the edge list once and produces every
  index the situation miners and cutoff selector need (entity adjacency,
  per-category neighbours, direct-pair set, daily edge counts, sampled
  per-day edge sets).
* :func:`detect_event_days` finds days whose edge count is at least
  ``sigma_threshold`` standard deviations above the daily mean and
  describes each one by its top entities and top relations.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from financegym.questions.entity_filter import (
    ENTITY_BLOCKLIST,
    categorize_relation,
    is_garbage_entity,
)

# How many edges to retain per day for entity-level analysis. Sampling
# keeps memory bounded on extremely busy days without affecting cutoff
# selection (which only needs entity diversity, not full enumeration).
DAILY_EDGE_SAMPLE_CAP = 2_000


def build_indexes(edges, categories):
  """Walk the edge list once and produce the shared per-stage indexes.

  The returned dict has:

  ``entity_edges``         entity → list of incident edges
  ``entity_rels``          entity → set of incident relations
  ``entity_cat_neighbors`` entity → category → list of (neighbor, relation)
  ``direct_pairs``         set of (min, max) entity pairs that share an edge
  ``daily_counts``         day (YYYY-MM-DD) → edge count
  ``daily_edges``          day → up to :data:`DAILY_EDGE_SAMPLE_CAP` edges
  """
  entity_edges: dict[str, list[dict]] = defaultdict(list)
  entity_rels: dict[str, set[str]] = defaultdict(set)
  entity_cat_neighbors: dict[str, dict[str, list[tuple[str, str]]]] = (
      defaultdict(lambda: defaultdict(list))
  )
  direct_pairs: set[tuple[str, str]] = set()
  daily_counts: dict[str, int] = defaultdict(int)
  daily_edges: dict[str, list[dict]] = defaultdict(list)

  for e in edges:
    h, t = e["head"], e["tail"]
    entity_edges[h].append(e)
    entity_edges[t].append(e)
    entity_rels[h].add(e["relation"])
    entity_rels[t].add(e["relation"])

    cat = categorize_relation(e["relation"], categories)
    entity_cat_neighbors[h][cat].append((t, e["relation"]))
    entity_cat_neighbors[t][cat].append((h, e["relation"]))

    direct_pairs.add((min(h, t), max(h, t)))

    day = e.get("pub_date", "")[:10]
    if day:
      daily_counts[day] += 1
      if len(daily_edges[day]) < DAILY_EDGE_SAMPLE_CAP:
        daily_edges[day].append(e)

  return {
      "entity_edges": dict(entity_edges),
      "entity_rels": dict(entity_rels),
      "entity_cat_neighbors": {
          k: dict(v) for k, v in entity_cat_neighbors.items()
      },
      "direct_pairs": direct_pairs,
      "daily_counts": dict(daily_counts),
      "daily_edges": dict(daily_edges),
  }


def detect_event_days(
    indexes,
    *,
    sigma_threshold = 2.0,
    min_date = "2025-01-01",
    max_date = "2026-01-01",
    supplement_to = 30,
):
  """Return days whose edge volume is above ``mean + sigma_threshold * std``.

  Each event entry has ``date``, ``edge_count``, ``sigma``, ``top_entities``
  (filtered for garbage + blocklist), and ``top_relations``. If fewer than
  10 days pass the threshold the list is padded with high-volume days
  (still filtered for blocklisted entities) up to ``supplement_to``.
  """
  daily_counts = {
      d: c
      for d, c in indexes["daily_counts"].items()
      if min_date <= d < max_date
  }
  daily_edges = indexes["daily_edges"]
  if not daily_counts:
    return []

  values = list(daily_counts.values())
  mean = sum(values) / len(values)
  variance = sum((v - mean) ** 2 for v in values) / len(values)
  std = variance**0.5
  if std == 0:
    return []
  threshold = mean + sigma_threshold * std

  events: list[dict] = []
  for day, count in daily_counts.items():
    if count < threshold:
      continue
    day_e = daily_edges.get(day, [])
    ent_counts: Counter = Counter()
    rel_counts: Counter = Counter()
    for e in day_e:
      ent_counts[e["head"]] += 1
      ent_counts[e["tail"]] += 1
      rel_counts[e["relation"]] += 1
    top_ents = [
        ent
        for ent, _ in ent_counts.most_common(20)
        if not is_garbage_entity(ent) and ent not in ENTITY_BLOCKLIST
    ][:5]
    if not top_ents:
      continue
    events.append({
        "date": day,
        "edge_count": count,
        "sigma": round((count - mean) / std, 1),
        "top_entities": top_ents,
        "top_relations": [r for r, _ in rel_counts.most_common(3)],
    })

  events.sort(key=lambda x: -x["sigma"])

  if len(events) < 10:
    seen = {ev["date"] for ev in events}
    for day, count in sorted(daily_counts.items(), key=lambda x: -x[1]):
      if day in seen:
        continue
      day_e = daily_edges.get(day, [])
      ent_counts = Counter()
      for e in day_e:
        ent_counts[e["head"]] += 1
        ent_counts[e["tail"]] += 1
      top_ents = [
          ent
          for ent, _ in ent_counts.most_common(20)
          if not is_garbage_entity(ent) and ent not in ENTITY_BLOCKLIST
      ][:5]
      if not top_ents:
        continue
      events.append({
          "date": day,
          "edge_count": count,
          "sigma": round((count - mean) / std, 1),
          "top_entities": top_ents,
          "top_relations": [],
      })
      if len(events) >= supplement_to:
        break
    events.sort(key=lambda x: -x["sigma"])
  return events
