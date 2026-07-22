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

"""data.equity.comps — relative valuation against a peer set.

Fetches the trading multiples of a target and a model-supplied peer group
(yfinance), takes the peer medians, and applies them to the target's own metrics
to
imply a per-share value per multiple (P/E, EV/EBITDA, EV/revenue, P/S). Returns
the
peer table, the implied values, and the range vs the current price. The model
picks
the peers (it knows the comparables); the tool does the arithmetic faithfully.

It also surfaces per-company fundamentals (market cap, margins, growth, ROE,
leverage) — already in the fetched ``.info``, so a comps report can tabulate
them
from grounded data instead of improvising the peer table from prior knowledge.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import Any

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import (
    PROVIDER,
    TICKER_MAX_LEN,
    is_num,
    money,
    num,
    pct,
    yf_call,
)
from pydantic import BaseModel, Field

_DESCRIPTION = (
    "Comparable-company (relative) valuation. Pass the target as `ticker` and"
    " the comparison set as `peers` (a list of tickers, e.g."
    " ['AMD','AVGO','INTC']). It pulls each company's trading multiples (P/E,"
    " EV/EBITDA, EV/revenue, P/S), takes the peer medians, and applies them to"
    " the target to imply a value per share per multiple, with the range vs the"
    " current price. It also returns each company's fundamentals (market cap,"
    " margins, growth, ROE, net-debt/EBITDA), so the peer table can be built"
    " from these returned figures. The medians, implied per-share values, and"
    " range it returns are exact and auditable — reference them directly rather"
    " than re-deriving the math by hand. Pair with a DCF for a triangulated"
    " valuation."
)
_MAX_PEERS = 10

# our key -> provider .info field. Trading multiples drive the median/implied math.
_MULTIPLES = {
    "pe": "trailingPE",
    "ev_ebitda": "enterpriseToEbitda",
    "ev_rev": "enterpriseToRevenue",
    "ps": "priceToSalesTrailing12Months",
}

# Descriptive per-company fields a comps report tabulates — already present in the
# fetched .info, so surfacing them is free and keeps the peer table grounded.
_STATS = {
    "market_cap": "marketCap",
    "forward_pe": "forwardPE",
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "profit_margin": "profitMargins",
    "revenue_growth": "revenueGrowth",
    "return_on_equity": "returnOnEquity",
}


class EquityCompsRequest(BaseModel):
  """Input for a comparable-company relative valuation."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Target ticker (e.g. NVDA).",
  )
  peers: list[str] = Field(
      Ellipsis,
      min_length=1,
      max_length=_MAX_PEERS,
      description="Peer tickers to value against, e.g. ['AMD','AVGO','INTC'].",
  )


def _multiples(info):
  out: dict[str, float] = {}
  for key, field in _MULTIPLES.items():
    v = info.get(field)
    if is_num(v) and v > 0:
      out[key] = float(v)
  return out


def _stats(info):
  """Per-company fundamentals for the peer table (all from the fetched .info)."""
  out: dict[str, float] = {}
  for key, field in _STATS.items():
    v = info.get(field)
    if is_num(v):
      out[key] = float(v)
  debt, cash, ebitda = (
      info.get("totalDebt"),
      info.get("totalCash"),
      info.get("ebitda"),
  )
  if is_num(debt) and is_num(cash):
    net_debt = float(debt) - float(cash)
    out["net_debt"] = net_debt
    if is_num(ebitda) and ebitda:
      out["net_debt_to_ebitda"] = net_debt / float(ebitda)
  return out


def _target_metrics(info):
  debt, cash = info.get("totalDebt"), info.get("totalCash")
  net_debt = (debt - cash) if is_num(debt) and is_num(cash) else None
  return {
      "eps": info.get("trailingEps"),
      "ebitda": info.get("ebitda"),
      "revenue": info.get("totalRevenue"),
      "shares": info.get("sharesOutstanding"),
      "net_debt": net_debt,
      "price": info.get("currentPrice") or info.get("regularMarketPrice"),
  }


def _implied_per_share(
    metric, mult, t
):
  """Implied per-share value from one peer-median multiple + the target metric."""
  eps, ebitda, rev, shares, nd = (
      t.get("eps"),
      t.get("ebitda"),
      t.get("revenue"),
      t.get("shares"),
      t.get("net_debt"),
  )

  def _ev_to_share(ev):
    if not is_num(nd) or not is_num(shares) or not shares:
      return None
    return (ev - nd) / shares

  if metric == "pe":
    return mult * eps if is_num(eps) else None
  if metric == "ev_ebitda":
    return _ev_to_share(mult * ebitda) if is_num(ebitda) else None
  if metric == "ev_rev":
    return _ev_to_share(mult * rev) if is_num(rev) else None
  if metric == "ps":
    return (
        mult * rev / shares
        if is_num(rev) and is_num(shares) and shares
        else None
    )
  return None


def compute_comps(
    target, peer_multiples
):
  """Pure: peer medians → implied per-share values + range.

  ``target`` is the target's metrics; ``peer_multiples`` maps peer ticker → its
  multiples.
  """
  medians: dict[str, float] = {}
  for key in _MULTIPLES:
    vals = [m[key] for m in peer_multiples.values() if key in m]
    if vals:
      medians[key] = statistics.median(vals)

  implied = {
      key: _implied_per_share(key, med, target) for key, med in medians.items()
  }
  implied = {k: v for k, v in implied.items() if v is not None and v > 0}
  vals = list(implied.values())
  price = target.get("price")
  return {
      "peer_medians": medians,
      "implied_per_share": implied,
      "low": min(vals) if vals else None,
      "high": max(vals) if vals else None,
      "median_implied": statistics.median(vals) if vals else None,
      "current_price": price,
      "n_peers": len(peer_multiples),
  }


_LABELS = {
    "pe": "P/E",
    "ev_ebitda": "EV/EBITDA",
    "ev_rev": "EV/Rev",
    "ps": "P/S",
}
_STAT_LABELS = {
    "market_cap": "MktCap",
    "forward_pe": "FwdP/E",
    "gross_margin": "Gross",
    "operating_margin": "Op",
    "profit_margin": "Net",
    "revenue_growth": "RevGr",
    "return_on_equity": "ROE",
    "net_debt_to_ebitda": "NetDebt/EBITDA",
}


def _fmt_stat(key, v):
  if not is_num(v):
    return "n/a"
  if key == "market_cap":
    return money(v)
  if key == "net_debt_to_ebitda":
    return num(v) + "x"
  if key == "forward_pe":
    return num(v)
  return pct(v)  # margins, growth, ROE are decimal fractions


def _row(label, cells):
  return f"{label} | " + " | ".join(cells)


def _markdown(
    ticker,
    res,
    target_multiples,
    peer_multiples,
    stats,
):
  lines = [f"**{ticker}** relative valuation vs {res['n_peers']} peer(s):", ""]
  # Trading multiples — target, each peer, then the peer median (drives the math).
  lines.append("_Trading multiples_")
  lines.append(_row("Company", [_LABELS[k] for k in _MULTIPLES]))
  lines.append(_row(ticker, [num(target_multiples.get(k)) for k in _MULTIPLES]))
  for peer, m in peer_multiples.items():
    lines.append(_row(peer, [num(m.get(k)) for k in _MULTIPLES]))
  lines.append(
      _row("median", [num(res["peer_medians"].get(k)) for k in _MULTIPLES])
  )
  lines.append("")
  # Fundamentals — same row order, the grounded per-peer figures for the report.
  lines.append("_Fundamentals_")
  lines.append(_row("Company", [_STAT_LABELS[k] for k in _STAT_LABELS]))
  for sym in [ticker, *peer_multiples]:
    s = stats.get(sym, {})
    lines.append(_row(sym, [_fmt_stat(k, s.get(k)) for k in _STAT_LABELS]))
  lines.append("")
  if res["implied_per_share"]:
    lines.append("Implied value/share:")
    for k, v in res["implied_per_share"].items():
      lines.append(f"  via {_LABELS[k]}: {money(v)}")
    lines.append(
        f"**Range: {money(res['low'])} – {money(res['high'])}** "
        f"(median {money(res['median_implied'])})"
    )
    if res["current_price"]:
      mid = res["median_implied"]
      up = (mid - res["current_price"]) / res["current_price"] if mid else None
      lines.append(
          f"Current price {money(res['current_price'])} · "
          f"median-implied upside {pct(up) if up is not None else 'n/a'}"
      )
  else:
    lines.append(
        "No implied values (target metrics or peer multiples unavailable)."
    )
  return "\n".join(lines)


async def _handler(req):
  import yfinance as yf

  target = req.ticker.upper()
  peers = [p.upper() for p in req.peers if p.strip().upper() != target]
  tickers = [target, *peers]

  async def _info(sym):
    info = await yf_call(lambda: asyncio.to_thread(lambda: yf.Ticker(sym).info))
    return sym, (info or {})

  results = dict(await asyncio.gather(*(_info(s) for s in tickers)))
  t_info = results.get(target, {})
  if (
      not t_info
      or t_info.get("trailingEps") is None
      and t_info.get("ebitda") is None
  ):
    raise ToolError(f"no data for target '{target}' — check the ticker")

  peer_multiples = {p: _multiples(results.get(p, {})) for p in peers}
  peer_multiples = {
      p: m for p, m in peer_multiples.items() if m
  }  # drop peers with no multiples
  if not peer_multiples:
    raise ToolError("no usable peer multiples — check the peer tickers")

  target_multiples = _multiples(t_info)
  stats = {
      target: _stats(t_info),
      **{p: _stats(results.get(p, {})) for p in peer_multiples},
  }
  res = compute_comps(_target_metrics(t_info), peer_multiples)
  return ToolResponse(
      markdown=_markdown(target, res, target_multiples, peer_multiples, stats),
      structured={
          "ticker": target,
          **res,
          "target_multiples": target_multiples,
          "peer_multiples": peer_multiples,
          "stats": stats,
          "peers": list(peer_multiples),
      },
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_equity_comps",
    display_name="data.equity.comps",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=EquityCompsRequest,
    handler=_handler,
    tags=("equity", "comps", "relative-valuation", "data"),
)
