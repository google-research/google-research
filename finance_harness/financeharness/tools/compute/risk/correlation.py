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

"""compute.risk.correlation — return-correlation matrix across named price series.

Pure-math. Takes a set of named price series (chain them from
data.equity.prices'
``bars[*].close``), converts to returns, aligns to the common window, and
returns the
pairwise Pearson correlation matrix. Use to see how names co-move
(diversification,
crowding, pair risk).
"""

from __future__ import annotations

import statistics
from typing import Any, Self

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.compute.risk.returns import align, pct_returns
from financeharness.tools.format import num
from pydantic import BaseModel, Field, model_validator

_DESCRIPTION = (
    "Return-correlation matrix across two or more named price series (pass each"
    " as a list of prices, e.g. from data.equity.prices bars). Returns the"
    " pairwise Pearson correlations of their returns over the common window."
    " Use for co-movement, diversification, and pair risk."
)
_MIN_POINTS = 3


class CorrelationRequest(BaseModel):
  """Input for a correlation matrix over named price series."""

  series: dict[str, list[float]] = Field(
      Ellipsis,
      min_length=2,
      description=(
          "Map of ticker → price list (>=2 tickers, >=3 prices each), "
          'e.g. {"AAPL": [180, 182, 179], "MSFT": [400, 405, 402]}.'
      ),
  )

  @model_validator(mode="after")
  def _check_lengths(self):
    short = [k for k, v in self.series.items() if len(v) < _MIN_POINTS]
    if short:
      raise ValueError(
          f"series need >= {_MIN_POINTS} prices; too short: {short}"
      )
    return self


def compute_correlation(series):
  """Compute pairwise correlations from one common aligned return window."""

  names = list(series)
  rets = {k: pct_returns(v) for k, v in series.items()}
  # Align ALL series to one common (most-recent) window, so every pairwise
  # correlation is measured over the same observations — the standard for a
  # correlation matrix — and n_obs is the actual count used.
  aligned = dict(zip(names, align(*(rets[k] for k in names)), strict=True))
  n_obs = len(next(iter(aligned.values()))) if aligned else 0
  matrix: dict[str, dict[str, float | None]] = {a: {} for a in names}
  for i, a in enumerate(names):
    for b in names[i:]:
      ra, rb = aligned[a], aligned[b]
      # correlation is undefined (and statistics.correlation raises) when either
      # series is constant — report None rather than erroring on valid input.
      computable = (
          n_obs >= 2
          and statistics.variance(ra) > 0
          and statistics.variance(rb) > 0
      )
      corr = statistics.correlation(ra, rb) if computable else None
      c = round(corr, 4) if corr is not None else None
      matrix[a][b] = c
      matrix[b][a] = c
  return {"names": names, "matrix": matrix, "n_obs": n_obs}


def _markdown(res):
  names = res["names"]
  head = "corr | " + " | ".join(names)
  rows = [head, "—" * len(head)]
  for a in names:
    rows.append(
        f"{a} | " + " | ".join(num(res["matrix"][a].get(b)) for b in names)
    )
  return f"**Return correlation** (n={res['n_obs']}):\n" + "\n".join(rows)


async def _handler(req):
  res = compute_correlation(req.series)
  return ToolResponse(markdown=_markdown(res), structured=res, meta={})


SPEC = ToolSpec(
    name="compute_risk_correlation",
    display_name="compute.risk.correlation",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=CorrelationRequest,
    handler=_handler,
    tags=("compute", "risk", "correlation"),
)
