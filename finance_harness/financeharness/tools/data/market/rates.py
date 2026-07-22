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

"""data.market.rates — the US Treasury yield curve (the risk-free backdrop).

3-month, 5-, 10-, and 30-year Treasury yields (yfinance ^IRX/^FVX/^TNX/^TYX).
The
10-year is the canonical risk-free rate for CAPM/WACC — exposed both as a
percent
and as a decimal ready to chain into compute.valuation.wacc. Numbers only.
"""

from __future__ import annotations

import asyncio

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import PROVIDER, num
from financeharness.tools.data.market.quotes import last_quote
from pydantic import BaseModel

_DESCRIPTION = (
    "US Treasury yield curve — 3-month, 5-, 10-, and 30-year yields (percent)."
    " The 10-year is the standard risk-free rate for CAPM/WACC; it's also"
    " returned as a decimal (risk_free_rate) ready to chain into"
    " compute.valuation.wacc. Use for the rate backdrop and to ground a"
    " valuation's discount rate."
)
# tenor label → yfinance yield symbol (value is the yield in percent)
_TENORS = {"3m": "^IRX", "5y": "^FVX", "10y": "^TNX", "30y": "^TYX"}


class MarketRatesRequest(BaseModel):
  """Input for the Treasury-yield snapshot tool."""


async def _handler(_req):
  syms = list(_TENORS.values())
  quotes = dict(
      zip(
          syms,
          await asyncio.gather(*(last_quote(s) for s in syms)),
          strict=True,
      )
  )
  tenors = {
      label: quotes[sym][0]
      for label, sym in _TENORS.items()
      if quotes.get(sym) is not None
  }
  if not tenors:
    raise ToolError(
        "no Treasury yield data available right now (provider may be down)"
    )

  ten_year = tenors.get("10y")
  structured = {
      "tenors_pct": tenors,
      "ten_year_pct": ten_year,
      # decimal form for direct WACC chaining (4.45% → 0.0445)
      "risk_free_rate": (
          round(ten_year / 100, 5) if ten_year is not None else None
      ),
  }
  line = " · ".join(
      f"{label} {num(tenors[label])}%" for label in _TENORS if label in tenors
  )
  rf = (
      f"\nRisk-free (10y) = {num(ten_year)}% → {structured['risk_free_rate']}"
      " for WACC."
  )
  return ToolResponse(
      markdown=(
          f"**US Treasury yields:** {line}{rf if ten_year is not None else ''}"
      ),
      structured=structured,
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_market_rates",
    display_name="data.market.rates",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=MarketRatesRequest,
    handler=_handler,
    tags=("market", "rates", "treasury", "risk-free", "data"),
)
