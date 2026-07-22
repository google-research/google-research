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

"""data.equity.prices — OHLCV history + derived technicals.

`bars[*]` is the stable chaining surface (`prev:<id>.bars[*].close` into compute
tools). Derived measures are named canon only — SMA(20/50/200), golden/death
cross, realized volatility (per-bar stdev + annualized at 252 trading days,
factor shown). No interpretive tier labels; the model owns interpretation.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import PROVIDER, TICKER_MAX_LEN, is_num, num, yf_call
from pydantic import BaseModel, Field, field_validator

_VALID_PERIODS = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
_VALID_INTERVALS = ("1d", "5d", "1wk", "1mo")
_BARS_PER_YEAR = 252  # equities: trading days
_RECENT_ROWS = 8

_DESCRIPTION = (
    "Daily OHLCV price history for a public equity plus derived technicals:"
    " SMA(20/50/200), golden/death cross, realized volatility, and 52-week"
    " range. Exposes bars[*] for chaining price series into compute tools."
)


class EquityPricesRequest(BaseModel):
  """Input for fetching equity OHLCV history and technical measures."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Equity ticker (e.g. NVDA).",
  )
  period: str = Field(
      "1y", description=f"Window: one of {', '.join(_VALID_PERIODS)}."
  )
  interval: str = Field(
      "1d", description=f"Bar interval: one of {', '.join(_VALID_INTERVALS)}."
  )

  @field_validator("period")
  @classmethod
  def _vp(cls, v):
    if v not in _VALID_PERIODS:
      raise ValueError(f"period must be one of {_VALID_PERIODS}")
    return v

  @field_validator("interval")
  @classmethod
  def _vi(cls, v):
    if v not in _VALID_INTERVALS:
      raise ValueError(f"interval must be one of {_VALID_INTERVALS}")
    return v


def _sma_last(values, window):
  return sum(values[-window:]) / window if len(values) >= window else None


def _sma_series(values, window):
  out: list[float | None] = []
  running = 0.0
  for i, v in enumerate(values):
    running += v
    if i >= window:
      running -= values[i - window]
    out.append(running / window if i + 1 >= window else None)
  return out


def _realized_vol(closes):
  if len(closes) < 2:
    return None, None
  rets = []
  for i in range(1, len(closes)):
    a, b = closes[i - 1], closes[i]
    if a <= 0 or b <= 0:
      return None, None
    rets.append(math.log(b / a))
  n = len(rets)
  mean = sum(rets) / n
  var = sum((x - mean) ** 2 for x in rets) / (n - 1) if n > 1 else 0.0
  per_bar = math.sqrt(var)
  return per_bar, per_bar * math.sqrt(_BARS_PER_YEAR)


def _last_cross(closes):
  """Most recent SMA50 vs SMA200 crossover: 'golden' (50 over 200) or 'death'."""
  s50, s200 = _sma_series(closes, 50), _sma_series(closes, 200)
  last: str | None = None
  for i in range(1, len(closes)):
    if None in (s50[i], s50[i - 1], s200[i], s200[i - 1]):
      continue
    if s50[i - 1] <= s200[i - 1] and s50[i] > s200[i]:
      last = "golden"
    elif s50[i - 1] >= s200[i - 1] and s50[i] < s200[i]:
      last = "death"
  return last


def _build(ticker, rows):
  closes = [r["close"] for r in rows]
  per_bar, annualized = _realized_vol(closes)
  structured = {
      "ticker": ticker,
      "bars": rows,
      "current": closes[-1],
      "sma": {
          "20": _sma_last(closes, 20),
          "50": _sma_last(closes, 50),
          "200": _sma_last(closes, 200),
      },
      "last_cross": _last_cross(closes),
      "realized_vol": {
          "per_bar": per_bar,
          "annualized": annualized,
          "periods_per_year": _BARS_PER_YEAR,
      },
      "high": max(closes),
      "low": min(closes),
      "n_bars": len(rows),
  }
  sma = structured["sma"]
  recent = "\n".join(
      f"  {r['date']}: {num(r['close'])}" for r in rows[-_RECENT_ROWS:]
  )
  md = (
      f"**{ticker}** — {len(rows)} bars · current"
      f" {num(closes[-1])}\nSMA20/50/200:"
      f" {num(sma['20'])}/{num(sma['50'])}/{num(sma['200'])}{' · ' + structured['last_cross'] + ' cross' if structured['last_cross'] else ''}\nRealized"
      f" vol: per-bar {num(per_bar, 4)}, annualized {num(annualized, 3)}"
      f" (×√{_BARS_PER_YEAR}) · range"
      f" {num(structured['low'])}–{num(structured['high'])}\nRecent"
      f" closes:\n{recent}"
  )
  return ToolResponse(
      markdown=md, structured=structured, meta={"provider": PROVIDER}
  )


async def _handler(req):
  import yfinance as yf

  ticker = req.ticker.upper()

  def _sync():
    h = yf.Ticker(ticker).history(period=req.period, interval=req.interval)
    rows: list[dict[str, Any]] = []
    for idx, row in h.iterrows():
      # yfinance emits NaN-filled bars (halts, sparse intervals); skip them so
      # int(NaN) can't crash and NaN can't poison technicals / the payload.
      if not all(
          is_num(row[c]) for c in ("Open", "High", "Low", "Close", "Volume")
      ):
        continue
      rows.append({
          "date": idx.strftime("%Y-%m-%d"),
          "open": round(float(row["Open"]), 4),
          "high": round(float(row["High"]), 4),
          "low": round(float(row["Low"]), 4),
          "close": round(float(row["Close"]), 4),
          "volume": int(row["Volume"]),
      })
    return rows

  rows = await yf_call(lambda: asyncio.to_thread(_sync))
  if not rows:
    raise ToolError(f"no price data for '{ticker}' — check the ticker / period")
  return _build(ticker, rows)


SPEC = ToolSpec(
    name="data_equity_prices",
    display_name="data.equity.prices",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=EquityPricesRequest,
    handler=_handler,
    tags=("equity", "prices", "technical", "data"),
)
