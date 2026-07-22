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

"""Garbage-entity filter and edge-loading helpers shared by every question stage.

The LLM extractor sometimes returns dollar amounts, percentages, dates, or
generic role names ("CEO", "Management") as entities. These are filtered
once at edge-load time and again whenever an entity is about to be
referenced in a generated question or a downstream miner.

The two relation-categorization helpers (:func:`load_rel_categories`,
:func:`categorize_relation`) are stateless: callers pass a categories dict
rather than depending on a module-level global, which keeps the situation
miners testable without a JSON file on disk.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Entity blocklist + garbage patterns
# ---------------------------------------------------------------------------

ENTITY_BLOCKLIST: set[str] = {
    "Zacks",
    "Zacks Research",
    "Zacks Investment Research",
    "Zacks Consensus Estimate",
    "Zacks Rank #1 (Strong Buy)",
    "Zacks Rank #2 (Buy)",
    "Zacks Rank #3 (Hold)",
    "Zacks Rank #4 (Sell)",
    "Zacks Rank #5 (Strong Sell)",
    "Validea",
    "The Motley Fool",
    "Seeking Alpha",
    "InvestorPlace",
    "Hold",
    "Buy",
    "Sell",
    "Strong Buy",
    "Strong Sell",
    "#1 (Strong Buy)",
    "#2 (Buy)",
    "#3 (Hold)",
    "#4 (Sell)",
    "#5 (Strong Sell)",
    "Wall Street Zen",
    "StockNews.com",
}

_BLOCKLIST_PREFIXES: tuple[str, Ellipsis] = ("Zacks ",)

_GARBAGE_PATTERNS = [
    re.compile(r"^\$[\d,.]+\s*(billion|million|trillion)?$", re.I),
    re.compile(r"^[\d,.]+%$"),
    re.compile(r"^\d{4}$"),
    re.compile(r"^Q[1-4]\s+\d{4}$"),
    re.compile(r"^H[12]\s+\d{4}$"),
    re.compile(
        r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)$",
        re.I,
    ),
    re.compile(r"^\d+$"),
    re.compile(r"^\d[\d,.]+\s*(million|billion)$", re.I),
]

_GARBAGE_EXACT: set[str] = {
    "CEO",
    "CFO",
    "CTO",
    "COO",
    "Management",
    "Board of Directors",
    "new CEO",
    "inflation",
    "recession",
    "IPO",
    "SPAC",
}


def is_garbage_entity(name):
  """``True`` if the LLM-emitted entity is noise we should drop."""
  if name in _GARBAGE_EXACT:
    return True
  if name in ENTITY_BLOCKLIST:
    return True
  if any(name.startswith(p) for p in _BLOCKLIST_PREFIXES):
    return True
  if len(name) <= 1:
    return True
  return any(p.match(name) for p in _GARBAGE_PATTERNS)


# ---------------------------------------------------------------------------
# Relation taxonomy (loaded from financegym.graph.relation_categories output)
# ---------------------------------------------------------------------------


FALLBACK_REL_CATEGORIES: dict[str, list[str]] = {
    "corporate_action": [
        "acquired",
        "partnered_with",
        "invested_in",
        "launched",
        "spun_off",
    ],
    "financial_report": [
        "reported_revenue",
        "reported_eps",
        "beat_estimate",
        "missed_estimate",
    ],
    "people_governance": ["ceo_of", "appointed", "resigned", "board_member_of"],
    "market_competition": [
        "competes_with",
        "subsidiary_of",
        "supplier_to",
        "owns",
    ],
    "analyst_action": ["upgraded", "downgraded", "set_price_target"],
    "regulatory": ["sued", "fined", "settled_with", "regulated_by"],
    "macro_policy": ["raised_rate", "set_target", "signed_bill"],
}


def load_rel_categories(path):
  """Load the categories JSON from the graph stage; fall back to a small set."""
  if path is not None:
    p = Path(path)
    if p.exists():
      data = json.loads(p.read_text())
      cats = data.get("categories", {})
      cats.pop("other", None)
      return cats
  return dict(FALLBACK_REL_CATEGORIES)


def _build_lookup(categories):
  out: dict[str, str] = {}
  for cat, rels in categories.items():
    for r in rels:
      out[r] = cat
  return out


def categorize_relation(rel, categories):
  """Map a raw relation label to its category, or ``"other"``."""
  lookup = _build_lookup(categories)
  if rel in lookup:
    return lookup[rel]
  rel_lower = rel.lower()
  for cat, patterns in categories.items():
    if any(p in rel_lower for p in patterns):
      return cat
  return "other"


# ---------------------------------------------------------------------------
# Edge loading
# ---------------------------------------------------------------------------


def load_edges(path):
  """Load + clean an edge CSV written by :mod:`financegym.graph.extract_triples`.

  Filters out:
    - rows without a year-prefixed ``pub_date`` (parsing errors / nulls)
    - rows where head or tail is blocklisted or ≤ 2 chars
    - rows where head or tail matches the garbage pattern set
  """
  out: list[dict] = []
  with open(path) as f:
    for row in csv.DictReader(f):
      if not row.get("pub_date", "").startswith("20"):
        continue
      if row["head"] in ENTITY_BLOCKLIST or row["tail"] in ENTITY_BLOCKLIST:
        continue
      if len(row["head"]) <= 2 or len(row["tail"]) <= 2:
        continue
      if is_garbage_entity(row["head"]) or is_garbage_entity(row["tail"]):
        continue
      out.append(row)
  return out


# ---------------------------------------------------------------------------
# Sector normalization (v3 taxonomy — 10 sectors)
# ---------------------------------------------------------------------------


_V3_SECTORS: set[str] = {
    "equity",
    "fixed_income",
    "macro",
    "commodities",
    "fx",
    "crypto",
    "healthcare",
    "financials",
    "technology",
    "cross_asset",
}

_V3_SECTOR_MAP: dict[str, str] = {
    "cryptocurrency": "crypto",
    "digital_assets": "crypto",
    "crypto_digital_assets": "crypto",
    "pharma": "healthcare",
    "biotech": "healthcare",
    "healthcare_pharma": "healthcare",
    "banking": "financials",
    "fintech": "financials",
    "banking_financial": "financials",
    "semiconductors": "technology",
    "ai": "technology",
    "tech": "technology",
    "monetary_policy": "macro",
    "geopolitics": "macro",
    "trade_policy": "macro",
    "fiscal_policy": "macro",
    "energy": "commodities",
    "metals": "commodities",
    "bonds": "fixed_income",
    "credit": "fixed_income",
    "currencies": "fx",
    "foreign_exchange": "fx",
    "cross-asset": "cross_asset",
    "cross asset": "cross_asset",
    "fixed income": "fixed_income",
    "fixed-income": "fixed_income",
}

_SUBSTRING_RULES: list[tuple[str, str]] = [
    ("crypto", "crypto"),
    ("bitcoin", "crypto"),
    ("blockchain", "crypto"),
    ("defi", "crypto"),
    ("digital_asset", "crypto"),
    ("pharma", "healthcare"),
    ("biotech", "healthcare"),
    ("medical", "healthcare"),
    ("health", "healthcare"),
    ("drug", "healthcare"),
    ("fda", "healthcare"),
    ("bank", "financials"),
    ("fintech", "financials"),
    ("insurance", "financials"),
    ("asset_manage", "financials"),
    ("lending", "financials"),
    ("semiconductor", "technology"),
    ("chip", "technology"),
    ("software", "technology"),
    ("cloud", "technology"),
    ("artificial_intelligence", "technology"),
    ("energy", "commodities"),
    ("oil", "commodities"),
    ("metal", "commodities"),
    ("gold", "commodities"),
    ("mining", "commodities"),
    ("bond", "fixed_income"),
    ("credit", "fixed_income"),
    ("yield", "fixed_income"),
    ("debt", "fixed_income"),
    ("treasury", "fixed_income"),
    ("currency", "fx"),
    ("forex", "fx"),
    ("exchange_rate", "fx"),
    ("monetary", "macro"),
    ("fiscal", "macro"),
    ("inflation", "macro"),
    ("gdp", "macro"),
    ("geopolit", "macro"),
    ("trade_war", "macro"),
    ("tariff", "macro"),
    ("central_bank", "macro"),
]


def normalize_sector(sector):
  """Map an LLM-emitted sector label to one of the 10 canonical sectors."""
  if not sector:
    return "equity"
  s = sector.strip().lower().replace("-", "_").replace(" ", "_")
  if s in _V3_SECTORS:
    return s
  if s in _V3_SECTOR_MAP:
    return _V3_SECTOR_MAP[s]
  for keyword, target in _SUBSTRING_RULES:
    if keyword in s:
      return target
  if "/" in sector:
    return "cross_asset"
  return "equity"
