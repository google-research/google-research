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

"""data.equity.reference — identity / classification / market snapshot."""

from __future__ import annotations

import asyncio
from typing import Any

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import PROVIDER, TICKER_MAX_LEN, money, num, yf_call
from pydantic import BaseModel, Field

_DESCRIPTION = (
    "Reference data for a public equity: name, sector/industry, country,"
    " exchange, market cap, shares outstanding, beta, current price, 52-week"
    " range, and P/E. Use to identify and classify a ticker (see"
    " data.equity.ratios for full multiples incl. dividend yield)."
)

_STABLE_FIELDS = (
    "shortName",
    "longName",
    "sector",
    "industry",
    "country",
    "currency",
    "exchange",
    "quoteType",
    "marketCap",
    "sharesOutstanding",
    "beta",
    "currentPrice",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "trailingPE",
    "forwardPE",
    "longBusinessSummary",
    "website",
)


class EquityReferenceRequest(BaseModel):
  """Input for the equity reference-data tool."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Public equity ticker (e.g. NVDA, AAPL, 005930.KS).",
  )


def _curate(info, ticker):
  out = {k: info.get(k) for k in _STABLE_FIELDS if info.get(k) is not None}
  # canonical identifier under our `ticker` key (yfinance returns it as "symbol").
  out["ticker"] = (info.get("symbol") or ticker).upper()
  return out


def _markdown(d):
  name = d.get("longName") or d.get("shortName") or d.get("ticker")
  shares = d.get("sharesOutstanding")
  shares_str = money(shares).lstrip("$") if shares else "n/a"
  lines = [
      (
          f"**{name}** ({d.get('ticker')}) — {d.get('exchange', '?')} ·"
          f" {d.get('currency', '?')}"
      ),
      (
          f"Sector: {d.get('sector', 'n/a')} / {d.get('industry', 'n/a')} · "
          f"Country: {d.get('country', 'n/a')}"
      ),
      (
          f"Market cap: {money(d.get('marketCap'))} · Shares out: {shares_str}"
          f" · Beta: {num(d.get('beta'))}"
      ),
      (
          f"Price: {num(d.get('currentPrice'))} · 52w:"
          f" {num(d.get('fiftyTwoWeekLow'))}–{num(d.get('fiftyTwoWeekHigh'))} ·"
          " P/E (ttm/fwd):"
          f" {num(d.get('trailingPE'))}/{num(d.get('forwardPE'))}"
      ),
  ]
  summary = d.get("longBusinessSummary")
  if summary:
    lines.append("\n" + summary[:600])
  return "\n".join(lines)


async def _handler(req):
  import yfinance as yf

  ticker = req.ticker.upper()

  async def _fetch():
    return await asyncio.to_thread(lambda: yf.Ticker(ticker).info)

  info = await yf_call(_fetch)
  if not info or not (
      info.get("shortName") or info.get("longName") or info.get("marketCap")
  ):
    raise ToolError(f"no reference data for '{ticker}' — check the ticker")
  curated = _curate(info, ticker)
  return ToolResponse(
      markdown=_markdown(curated),
      structured=curated,
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_equity_reference",
    display_name="data.equity.reference",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=EquityReferenceRequest,
    handler=_handler,
    tags=("equity", "reference", "data"),
)
