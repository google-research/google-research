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

"""data.equity.fundamentals — financial summary + multi-year income trend.

Curated TTM financials from the provider snapshot plus a 4-year income trend
(revenue / operating income / net income / EBITDA). The structured payload is a
chaining surface for valuation tools (e.g. revenue + margins + FCF into a DCF).
Numbers only — the model interprets.
"""

from __future__ import annotations

import asyncio
from typing import Any

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import (
    PROVIDER,
    TICKER_MAX_LEN,
    df_cell,
    money,
    num,
    pct,
    yf_call,
)
from pydantic import BaseModel, Field

_DESCRIPTION = (
    "Fundamental financials for a public equity: revenue, EBITDA, margins, free"
    " and operating cash flow, total cash and debt, ROE, growth, and EPS, plus"
    " a multi-year income trend (revenue/operating income/net income/EBITDA)."
    " Feeds valuation tools like DCF."
)

# provider .info field -> our snake_case key
_INFO_MAP = {
    "totalRevenue": "total_revenue",
    "ebitda": "ebitda",
    "grossMargins": "gross_margin",
    "operatingMargins": "operating_margin",
    "profitMargins": "profit_margin",
    "freeCashflow": "free_cashflow",
    "operatingCashflow": "operating_cashflow",
    "totalCash": "total_cash",
    "totalDebt": "total_debt",
    "returnOnEquity": "return_on_equity",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "trailingEps": "trailing_eps",
    "forwardEps": "forward_eps",
    "priceToBook": "price_to_book",
}

_INCOME_ROWS = (
    ("Total Revenue", "revenue"),
    ("Operating Income", "operating_income"),
    ("Net Income", "net_income"),
    ("EBITDA", "ebitda"),
)


class EquityFundamentalsRequest(BaseModel):
  """Input for fetching equity fundamentals and income-statement history."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Equity ticker (e.g. NVDA).",
  )


def _income_trend(inc):
  if inc is None or getattr(inc, "empty", True):
    return []
  years: list[dict[str, Any]] = []
  for col in list(inc.columns)[:4]:
    entry: dict[str, Any] = {
        "period": str(col.date()) if hasattr(col, "date") else str(col)
    }
    for label, key in _INCOME_ROWS:
      if label in inc.index:
        entry[key] = df_cell(inc.loc[label, col])
    years.append(entry)
  return years


def _markdown(
    snap, trend, ticker
):
  lines = [
      f"**{ticker}** fundamentals (TTM):",
      (
          f"Revenue {money(snap.get('total_revenue'))} · EBITDA"
          f" {money(snap.get('ebitda'))} · FCF"
          f" {money(snap.get('free_cashflow'))} · OCF"
          f" {money(snap.get('operating_cashflow'))}"
      ),
      (
          f"Margins — gross {pct(snap.get('gross_margin'))}, op"
          f" {pct(snap.get('operating_margin'))}, net"
          f" {pct(snap.get('profit_margin'))} · ROE"
          f" {pct(snap.get('return_on_equity'))}"
      ),
      (
          f"Cash {money(snap.get('total_cash'))} · Debt"
          f" {money(snap.get('total_debt'))} · Rev growth"
          f" {pct(snap.get('revenue_growth'))} · EPS ttm/fwd"
          f" {num(snap.get('trailing_eps'))}/{num(snap.get('forward_eps'))}"
      ),
  ]
  if trend:
    lines.append("\nIncome trend (revenue / op income / net income):")
    for y in trend:
      lines.append(
          f"  {y['period']}: {money(y.get('revenue'))} / "
          f"{money(y.get('operating_income'))} / "
          f"{money(y.get('net_income'))}"
      )
  return "\n".join(lines)


async def _handler(req):
  import yfinance as yf

  ticker = req.ticker.upper()

  def _sync():
    tk = yf.Ticker(ticker)
    return tk.info, tk.income_stmt

  info, inc = await yf_call(lambda: asyncio.to_thread(_sync))
  if (
      not info
      or info.get("totalRevenue") is None
      and not (inc is not None and not inc.empty)
  ):
    raise ToolError(f"no fundamentals for '{ticker}' — check the ticker")

  snap = {
      our: info.get(prov)
      for prov, our in _INFO_MAP.items()
      if info.get(prov) is not None
  }
  trend = _income_trend(inc)
  structured = {"ticker": ticker, "ttm": snap, "income_trend": trend}
  return ToolResponse(
      markdown=_markdown(snap, trend, ticker),
      structured=structured,
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_equity_fundamentals",
    display_name="data.equity.fundamentals",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=EquityFundamentalsRequest,
    handler=_handler,
    tags=("equity", "fundamentals", "financials", "data"),
)
