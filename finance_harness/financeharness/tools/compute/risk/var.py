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

"""compute.risk.var — Value-at-Risk of a return series (historical + parametric).

Pure-math. From a price series (chain from data.equity.prices), the worst
expected
loss at a confidence level over a horizon: historical (empirical left-tail
quantile)
and parametric (normal: −(μ + zσ)), each scaled by √horizon. Reported as a
positive
% loss. A downside-risk lens on a single name.
"""

from __future__ import annotations

import statistics
from typing import Any

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.compute.risk.returns import pct_returns
from financeharness.tools.format import pct
from pydantic import BaseModel, Field

_DESCRIPTION = (
    "Value-at-Risk for a price series (pass prices, e.g. from"
    " data.equity.prices): the worst expected loss at a confidence level over a"
    " horizon, both historical (empirical tail) and parametric (normal)."
    " Returns positive % losses. confidence default 0.95, horizon_days"
    " default 1."
)
_MIN_POINTS = 5


class VarRequest(BaseModel):
  """Input for historical value-at-risk over a price series."""

  prices: list[float] = Field(
      Ellipsis,
      min_length=_MIN_POINTS,
      description="Price series (>=5), e.g. [180, 182, 179, 185, 183].",
  )
  confidence: float = Field(
      0.95, gt=0.5, lt=1, description="Confidence level, e.g. 0.95."
  )
  horizon_days: int = Field(
      1, ge=1, le=252, description="Horizon in periods (sqrt-scaled)."
  )


def _percentile(xs, q):
  s = sorted(xs)
  idx = q * (len(s) - 1)
  lo = int(idx)
  if lo + 1 >= len(s):
    return s[-1]
  return s[lo] * (1 - (idx - lo)) + s[lo + 1] * (idx - lo)


def compute_var(
    prices, confidence, horizon_days
):
  """Compute historical VaR from percentage returns."""

  rets = pct_returns(prices)
  scale = horizon_days**0.5
  tail = 1 - confidence
  hist = -_percentile(rets, tail) * scale
  mu = statistics.fmean(rets)
  sigma = statistics.pstdev(rets)
  z = statistics.NormalDist().inv_cdf(tail)  # negative
  param = -(mu + z * sigma) * scale
  return {
      "confidence": confidence,
      "horizon_days": horizon_days,
      "historical_var_pct": round(max(hist, 0.0), 6),
      "parametric_var_pct": round(max(param, 0.0), 6),
      "mean_return": round(mu, 6),
      "volatility": round(sigma, 6),
      "n_obs": len(rets),
  }


def _markdown(res):
  return (
      f"**Value-at-Risk** ({pct(res['confidence'])} conf,"
      f" {res['horizon_days']}d, n={res['n_obs']}):\n  historical"
      f" {pct(res['historical_var_pct'])} · parametric"
      f" {pct(res['parametric_var_pct'])} loss\n  (per-period vol"
      f" {pct(res['volatility'])})"
  )


async def _handler(req):
  res = compute_var(req.prices, req.confidence, req.horizon_days)
  return ToolResponse(markdown=_markdown(res), structured=res, meta={})


SPEC = ToolSpec(
    name="compute_risk_var",
    display_name="compute.risk.var",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=VarRequest,
    handler=_handler,
    tags=("compute", "risk", "var", "downside"),
)
