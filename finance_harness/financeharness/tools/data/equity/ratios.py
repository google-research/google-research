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

"""data.equity.ratios — financial-health ratios + valuation multiples.

Complements `data.equity.fundamentals` (size, margins, cash flow) with the ratio
view an analyst reaches for: liquidity, leverage, returns on assets/equity, and
the
company's own valuation multiples. Numbers from the provider snapshot (yfinance)
plus
a couple of safe derived figures (net debt, net-debt/EBITDA). Numbers only — the
model interprets and compares.
"""

from __future__ import annotations

import asyncio
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
    "Financial-health ratios and valuation multiples for a public equity:"
    " liquidity (current, quick), leverage (debt/equity, net debt,"
    " net-debt/EBITDA), returns (ROA, ROE), and multiples (P/E trailing &"
    " forward, EV/EBITDA, EV/revenue, P/S, PEG, dividend yield). Complements"
    " fundamentals for valuation and credit work."
)

# provider .info field -> our snake_case key (ratios already in usable form)
_RATIO_MAP = {
    "currentRatio": "current_ratio",
    "quickRatio": "quick_ratio",
    "returnOnAssets": "return_on_assets",
    "returnOnEquity": "return_on_equity",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "enterpriseToEbitda": "ev_to_ebitda",
    "enterpriseToRevenue": "ev_to_revenue",
    "priceToSalesTrailing12Months": "price_to_sales",
}


class EquityRatiosRequest(BaseModel):
  """Input for fetching ratio and valuation-multiple snapshots."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Equity ticker (e.g. NVDA).",
  )


def _derive(info):
  """Safe derived figures the provider doesn't give directly."""
  out: dict[str, Any] = {}
  # yfinance reports debtToEquity as a percentage (e.g. 168.6 → 1.69x).
  dte = info.get("debtToEquity")
  if is_num(dte):
    out["debt_to_equity"] = dte / 100.0
  debt, cash = info.get("totalDebt"), info.get("totalCash")
  if is_num(debt) and is_num(cash):
    net_debt = debt - cash
    out["net_debt"] = net_debt
    ebitda = info.get("ebitda")
    if is_num(ebitda) and ebitda:
      out["net_debt_to_ebitda"] = net_debt / ebitda
  # peg: newer yfinance exposes trailingPegRatio; fall back to pegRatio
  peg = info.get("trailingPegRatio")
  if not is_num(peg):
    peg = info.get("pegRatio")
  if is_num(peg):
    out["peg_ratio"] = peg
  # yfinance now reports dividendYield as a percent number (2.99 → 2.99%), not a
  # fraction — normalize to a fraction so it renders and chains consistently.
  dy = info.get("dividendYield")
  if is_num(dy):
    out["dividend_yield"] = dy / 100.0
  return out


def _markdown(r, ticker):
  return "\n".join([
      f"**{ticker}** ratios & multiples:",
      (
          f"Liquidity — current {num(r.get('current_ratio'))} · "
          f"quick {num(r.get('quick_ratio'))}"
      ),
      (
          f"Leverage — debt/equity {num(r.get('debt_to_equity'))}x · "
          f"net debt {money(r.get('net_debt'))} · "
          f"net-debt/EBITDA {num(r.get('net_debt_to_ebitda'))}x"
      ),
      (
          f"Returns — ROA {pct(r.get('return_on_assets'))} · "
          f"ROE {pct(r.get('return_on_equity'))}"
      ),
      (
          f"Multiples — P/E {num(r.get('trailing_pe'))} (fwd"
          f" {num(r.get('forward_pe'))}) · EV/EBITDA"
          f" {num(r.get('ev_to_ebitda'))} · EV/rev"
          f" {num(r.get('ev_to_revenue'))} · P/S {num(r.get('price_to_sales'))}"
          f" · PEG {num(r.get('peg_ratio'))} · div yield"
          f" {pct(r.get('dividend_yield'))}"
      ),
  ])


async def _handler(req):
  import yfinance as yf

  ticker = req.ticker.upper()
  info = await yf_call(
      lambda: asyncio.to_thread(lambda: yf.Ticker(ticker).info)
  )
  if (
      not info
      or info.get("currentRatio") is None
      and info.get("trailingPE") is None
  ):
    raise ToolError(f"no ratio data for '{ticker}' — check the ticker")

  ratios = {
      our: info.get(prov)
      for prov, our in _RATIO_MAP.items()
      if is_num(info.get(prov))
  }
  ratios.update(_derive(info))
  return ToolResponse(
      markdown=_markdown(ratios, ticker),
      structured={"ticker": ticker, "ratios": ratios},
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_equity_ratios",
    display_name="data.equity.ratios",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=EquityRatiosRequest,
    handler=_handler,
    tags=("equity", "ratios", "valuation", "credit", "data"),
)
