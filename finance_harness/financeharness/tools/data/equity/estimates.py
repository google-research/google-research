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

"""data.equity.estimates — analyst consensus: price targets, forward estimates, ratings.

The forward-looking, sell-side view yfinance exposes: analyst price targets
(current/mean/median/high/low), consensus EPS and revenue estimates by period
(current & next quarter, current & next fiscal year, with analyst counts and
growth), and
the latest buy/hold/sell recommendation spread. Numbers only — and clearly
marked as
*estimates*, which the model presents as consensus expectations, not facts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import (
    PROVIDER,
    TICKER_MAX_LEN,
    df_cell,
    is_num,
    money,
    num,
    pct,
    yf_call,
)
from pydantic import BaseModel, Field

_DESCRIPTION = (
    "Analyst consensus for a public equity: price targets (mean/median/high/low"
    " vs current), forward EPS and revenue estimates by period (current & next"
    " quarter, current & next fiscal year, with analyst counts and expected"
    " growth), and the latest buy/hold/sell recommendation spread."
    " Forward-looking sell-side estimates, not reported results — present as"
    " consensus expectations."
)
_PERIODS = ("0q", "+1q", "0y", "+1y")  # current Q, next Q, current FY, next FY


class EquityEstimatesRequest(BaseModel):
  """Input for fetching analyst consensus estimates."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Equity ticker (e.g. NVDA).",
  )


def _df_records(df, fields):
  """Rows of a period-indexed estimate frame → list of {period, ...mapped fields}."""
  if df is None or getattr(df, "empty", True):
    return []
  out: list[dict[str, Any]] = []
  for period in df.index:
    if str(period) not in _PERIODS:
      continue
    row: dict[str, Any] = {"period": str(period)}
    for col, key in fields.items():
      if col in df.columns:
        row[key] = df_cell(df.loc[period, col])
    out.append(row)
  return out


def _recommendation(df):
  if df is None or getattr(df, "empty", True):
    return None
  # select the current period ("0m") explicitly rather than trusting row order
  if "period" in df.columns and (df["period"] == "0m").any():
    row = df[df["period"] == "0m"].iloc[0]
  else:
    row = df.iloc[0]
  keys = {
      "strongBuy": "strong_buy",
      "buy": "buy",
      "hold": "hold",
      "sell": "sell",
      "strongSell": "strong_sell",
  }
  return {
      our: int(row[prov])
      for prov, our in keys.items()
      if prov in df.columns and is_num(row[prov])
  }


def _markdown(
    ticker, pt, eps, rev, rec
):
  lines = [f"**{ticker}** analyst consensus:"]
  if pt:
    lines.append(
        f"Price target — mean {money(pt.get('mean'))} (median"
        f" {money(pt.get('median'))},"
        f" {money(pt.get('low'))}–{money(pt.get('high'))}) vs current"
        f" {money(pt.get('current'))}"
    )
  rev_by = {r["period"]: r for r in rev}
  if eps:
    lines.append("Estimates (consensus, avg):")
    for e in eps:
      r = rev_by.get(e["period"], {})
      n = int(e["num_analysts"]) if e.get("num_analysts") else "n/a"
      lines.append(
          f"  {e['period']}: EPS {num(e.get('avg'))} (n={n}, "
          f"growth {pct(e.get('growth'))}) · revenue {money(r.get('avg'))}"
      )
  if rec:
    total = sum(rec.values()) or 1
    buys = rec.get("strong_buy", 0) + rec.get("buy", 0)
    lines.append(
        f"Ratings — {rec.get('strong_buy', 0)} strong buy · {rec.get('buy', 0)}"
        f" buy · {rec.get('hold', 0)} hold · {rec.get('sell', 0)} sell ·"
        f" {rec.get('strong_sell', 0)} strong sell ({pct(buys / total)}"
        " bullish)"
    )
  return "\n".join(lines)


async def _handler(req):
  import yfinance as yf

  ticker = req.ticker.upper()

  def _sync():
    tk = yf.Ticker(ticker)
    return (
        tk.analyst_price_targets,
        tk.earnings_estimate,
        tk.revenue_estimate,
        tk.recommendations,
    )

  pt_raw, eps_df, rev_df, rec_df = await yf_call(
      lambda: asyncio.to_thread(_sync)
  )
  # keep only real numbers — NaN would poison the JSON-serialized structured payload
  pt = (
      {
          k: pt_raw[k]
          for k in ("current", "mean", "median", "high", "low")
          if is_num(pt_raw.get(k))
      }
      if pt_raw
      else {}
  )
  eps = _df_records(
      eps_df,
      {
          "avg": "avg",
          "low": "low",
          "high": "high",
          "numberOfAnalysts": "num_analysts",
          "growth": "growth",
      },
  )
  rev = _df_records(rev_df, {"avg": "avg", "growth": "growth"})
  rec = _recommendation(rec_df)

  if not pt and not eps and not rec:
    raise ToolError(f"no analyst estimates for '{ticker}' — check the ticker")

  return ToolResponse(
      markdown=_markdown(ticker, pt, eps, rev, rec),
      structured={
          "ticker": ticker,
          "price_targets": pt,
          "eps_estimates": eps,
          "revenue_estimates": rev,
          "recommendations": rec,
      },
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_equity_estimates",
    display_name="data.equity.estimates",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=EquityEstimatesRequest,
    handler=_handler,
    tags=("equity", "estimates", "analyst", "consensus", "data"),
)
