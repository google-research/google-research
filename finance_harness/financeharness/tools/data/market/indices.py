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

"""data.market.indices — major US equity indices + the VIX.

The market backdrop for equity research: S&P 500, Nasdaq Composite, Dow, Russell
2000, and the VIX volatility index — last level + day change. Use to frame a
single name against the broad market and the volatility regime. Numbers only.
"""

from __future__ import annotations

import asyncio

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import PROVIDER, num
from financeharness.tools.data.market.quotes import last_quote
from pydantic import BaseModel

_DESCRIPTION = (
    "Major US market indices + volatility: S&P 500, Nasdaq Composite, Dow"
    " Jones, Russell 2000, and the VIX — last level and day change. Use to"
    " frame an equity against the broad market and the current volatility"
    " regime."
)
# display name → yfinance index symbol
_INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
}


class MarketIndicesRequest(BaseModel):
  """Input for the market-indices snapshot tool."""


async def _handler(_req):
  syms = list(_INDICES.values())
  quotes = dict(
      zip(
          syms,
          await asyncio.gather(*(last_quote(s) for s in syms)),
          strict=True,
      )
  )
  out: dict[str, dict] = {}
  for name, sym in _INDICES.items():
    q = quotes.get(sym)
    if q is not None:
      out[name] = {"level": q[0], "change_pct": q[1]}
  if not out:
    raise ToolError("no index data available right now (provider may be down)")

  lines = [
      f"  {name}: {num(d['level'])} ({d['change_pct']:+.2f}%)"
      for name, d in out.items()
  ]
  return ToolResponse(
      markdown="**Market indices:**\n" + "\n".join(lines),
      structured={"indices": out},
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_market_indices",
    display_name="data.market.indices",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=MarketIndicesRequest,
    handler=_handler,
    tags=("market", "indices", "benchmark", "volatility", "data"),
)
