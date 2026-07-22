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

"""Three situation-mining modes used by the v3 question pipeline.

Each miner reads the shared indexes produced by
:func:`financegym.questions.event_detection.build_indexes` and returns a list
of situation dicts (``situation_type``, ``focus_entities``, ``edges``,
``signal``, ``score``, etc.). Generation reads the same dict shape.

The three modes are intentionally complementary:

* :func:`mine_multihop_paths` — 2-hop graph paths that cross relation
  categories, surfacing causal chains the LLM would otherwise miss.
* :func:`mine_temporal_narratives` — entities whose dominant relation
  category changes across months, capturing storylines.
* :func:`mine_tension_situations` — single-entity polar divergences
  (beat/miss, upgrade/downgrade, etc.). Demoted in v3 to a capped role.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from financegym.questions.entity_filter import (
    ENTITY_BLOCKLIST,
    categorize_relation,
    is_garbage_entity,
)

SKIP_CATEGORIES: set[str] = {"other", "uncategorized"}


# ---------------------------------------------------------------------------
# Multi-hop path mining (2hop, cross-category)
# ---------------------------------------------------------------------------


def mine_multihop_paths(
    edges,  # noqa: ARG001 - kept for API symmetry with the other miners
    indexes,
    categories,
    *,
    max_paths = 500,
    min_edges_per_hop = 3,
    min_entity_edges = 50,
):
  """Find A →[cat1]→ B →[cat2]→ C paths with disjoint ends."""
  entity_cat = indexes["entity_cat_neighbors"]
  entity_edges = indexes["entity_edges"]
  direct_pairs = indexes["direct_pairs"]

  bridges: list[tuple[str, list[str]]] = []
  for ent, cat_dict in entity_cat.items():
    if is_garbage_entity(ent) or ent in ENTITY_BLOCKLIST:
      continue
    if len(entity_edges.get(ent, [])) < min_entity_edges:
      continue
    good_cats = [
        c
        for c, neighbours in cat_dict.items()
        if c not in SKIP_CATEGORIES and len(neighbours) >= min_edges_per_hop
    ]
    if len(good_cats) >= 2:
      bridges.append((ent, good_cats))

  paths: list[dict] = []
  seen: set[tuple[str, str, str]] = set()

  for bridge, cats in bridges:
    cat_dict = entity_cat[bridge]
    for cat1 in cats:
      for cat2 in cats:
        if cat1 == cat2:
          continue
        if (bridge, cat1, cat2) in seen or (bridge, cat2, cat1) in seen:
          continue
        seen.add((bridge, cat1, cat2))

        n1_counts: Counter = Counter(n for n, _ in cat_dict[cat1])
        n2_counts: Counter = Counter(n for n, _ in cat_dict[cat2])

        def _valid(name, c):
          return (
              c >= min_edges_per_hop
              and not is_garbage_entity(name)
              and name not in ENTITY_BLOCKLIST
              and name != bridge  # noqa: B023 - captured by design
          )

        valid_1 = [(n, c) for n, c in n1_counts.most_common(10) if _valid(n, c)]
        valid_2 = [(n, c) for n, c in n2_counts.most_common(10) if _valid(n, c)]
        if not valid_1 or not valid_2:
          continue

        placed = False
        for ent_a, _ in valid_1[:5]:
          for ent_c, _ in valid_2[:5]:
            if ent_a == ent_c:
              continue
            if (min(ent_a, ent_c), max(ent_a, ent_c)) in direct_pairs:
              continue
            bridge_e = entity_edges[bridge]
            hop1 = [e for e in bridge_e if ent_a in (e["head"], e["tail"])]
            hop2 = [e for e in bridge_e if ent_c in (e["head"], e["tail"])]
            all_e = hop1 + hop2
            months = sorted({e["pub_date"][:7] for e in all_e})
            paths.append({
                "situation_type": "multihop_path",
                "path_type": "2hop",
                "focus_entities": [ent_a, bridge, ent_c],
                "category_sequence": [cat1, cat2],
                "edges": all_e,
                "signal": f"{ent_a} →[{cat1}]→ {bridge} →[{cat2}]→ {ent_c}",
                "months": months,
                "n_edges": len(all_e),
                "score": len(all_e) * len(months),
            })
            placed = True
            break
          if placed:
            break

  paths.sort(key=lambda p: -p["score"])
  return paths[:max_paths]


# ---------------------------------------------------------------------------
# Temporal narrative mining
# ---------------------------------------------------------------------------


def mine_temporal_narratives(
    edges,  # noqa: ARG001
    indexes,
    categories,
    *,
    min_categories = 3,
    min_months_span = 3,
    min_entity_edges = 100,
    max_narratives = 300,
):
  """Find entities whose dominant category changes across months."""
  entity_edges = indexes["entity_edges"]
  out: list[dict] = []

  for ent, ent_e in entity_edges.items():
    if is_garbage_entity(ent) or ent in ENTITY_BLOCKLIST:
      continue
    if len(ent_e) < min_entity_edges:
      continue

    monthly_cats: dict[str, Counter] = defaultdict(Counter)
    for e in ent_e:
      month = e["pub_date"][:7]
      if not month.startswith("20"):
        continue
      cat = categorize_relation(e["relation"], categories)
      if cat not in SKIP_CATEGORIES:
        monthly_cats[month][cat] += 1

    if len(monthly_cats) < min_months_span:
      continue

    arc: list[tuple[str, str]] = []
    for month in sorted(monthly_cats):
      if monthly_cats[month]:
        arc.append((month, monthly_cats[month].most_common(1)[0][0]))
    if not arc:
      continue

    distinct = {c for _, c in arc}
    transitions = sum(
        1 for i in range(1, len(arc)) if arc[i][1] != arc[i - 1][1]
    )
    if len(distinct) < min_categories or transitions < 2:
      continue

    out.append({
        "situation_type": "temporal_narrative",
        "focus_entities": [ent],
        "category_arc": arc,
        "edges": ent_e,
        "signal": f"{ent}: " + " → ".join(f"{c}({m})" for m, c in arc[:5]),
        "months": [m for m, _ in arc],
        "n_edges": len(ent_e),
        "n_categories": len(distinct),
        "n_transitions": transitions,
        "score": len(distinct) * len(arc) * len(ent_e),
    })

  out.sort(key=lambda s: -s["score"])
  return out[:max_narratives]


# ---------------------------------------------------------------------------
# Tension mining (polar divergence). Demoted role in v3.
# ---------------------------------------------------------------------------


TENSION_PAIRS: list[tuple[str, str, str]] = [
    ("beat_estimate", "missed_estimate", "earnings_surprise"),
    ("beat_revenue_estimate", "missed_revenue_estimate", "revenue_surprise"),
    ("upgraded", "downgraded", "analyst_disagreement"),
    ("raised_price_target", "lowered_price_target", "price_target_divergence"),
    ("raised_guidance", "lowered_guidance", "guidance_reversal"),
    ("raised_dividend", "cut_dividend", "dividend_policy_shift"),
    ("acquired", "divested", "strategic_portfolio_shift"),
    ("hired", "laid_off", "workforce_restructuring"),
    ("outperformed", "underperformed", "performance_divergence"),
    ("increased_stake", "decreased_stake", "ownership_churn"),
]


def mine_tension_situations(
    edges,  # noqa: ARG001
    indexes,
    *,
    max_situations = 50,
    min_edges = 30,
    min_months = 2,
):
  """Find entities that display both poles of a tension pair."""
  entity_edges = indexes["entity_edges"]
  entity_rels = indexes["entity_rels"]
  raw: list[dict] = []

  for pos, neg, signal_type in TENSION_PAIRS:
    for ent, rels in entity_rels.items():
      if is_garbage_entity(ent) or ent in ENTITY_BLOCKLIST:
        continue
      has_pos = any(pos in r for r in rels)
      has_neg = any(neg in r for r in rels)
      if not (has_pos and has_neg):
        continue
      ent_e = entity_edges[ent]
      if len(ent_e) < min_edges:
        continue
      months = sorted({e["pub_date"][:7] for e in ent_e})
      if len(months) < min_months:
        continue
      raw.append({
          "situation_type": f"tension_{signal_type}",
          "focus_entities": [ent],
          "edges": ent_e,
          "signal": f"{ent} has both {pos} and {neg} signals",
          "months": months,
          "n_edges": len(ent_e),
          "score": len(ent_e) * len(months),
      })

  # Keep one per entity: the highest-scoring tension type.
  best: dict[str, dict] = {}
  for s in raw:
    ent = s["focus_entities"][0]
    if ent not in best or s["score"] > best[ent]["score"]:
      best[ent] = s

  deduped = sorted(best.values(), key=lambda s: -s["score"])
  return deduped[:max_situations]
