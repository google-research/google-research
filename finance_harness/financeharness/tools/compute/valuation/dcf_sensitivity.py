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

"""compute.valuation.dcf_sensitivity — a DCF value across a 2-D driver grid.

Reuses the DCF core (`compute_dcf`) to value every cell of a discount-rate ×
terminal-driver grid, so an analyst sees how intrinsic value moves with the two
assumptions that dominate it — and reads a bear/base/bull range straight off the
table. Pure-math; each cell is a faithful DCF, invalid cells (e.g. g ≥ r) are
left
blank rather than fudged.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.compute.valuation.dcf import DCFRequest, compute_dcf
from financeharness.tools.format import money, num, pct
from pydantic import BaseModel, Field, ValidationError, model_validator

_DESCRIPTION = (
    "DCF sensitivity grid: values intrinsic value across a range of discount"
    " rates (rows) and a terminal driver (columns) — terminal_growth_rates for"
    " gordon_growth, or terminal_ebitda_multiples for exit_multiple. Takes the"
    " same FCF schedule / net_debt / shares as the DCF tool plus the two axes,"
    " and returns the value matrix with the bear/base/bull range. Use it to"
    " show how the valuation depends on WACC and terminal assumptions."
)


class DCFSensitivityRequest(BaseModel):
  """Input for a DCF sensitivity matrix over discount and terminal assumptions."""

  fcf_schedule: list[float] = Field(
      Ellipsis,
      min_length=1,
      max_length=30,
      description=(
          "Explicit-period FCFs as a flat list, year 1 → N (USD), "
          "e.g. [1200, 1320, 1450]."
      ),
  )
  discount_rates: list[float] = Field(
      Ellipsis,
      min_length=1,
      max_length=12,
      description=(
          "Discount-rate axis (rows), decimals e.g. [0.09, 0.10, 0.11]."
      ),
  )
  terminal_method: Literal["gordon_growth", "exit_multiple"] = "gordon_growth"
  terminal_growth_rates: list[float] | None = Field(
      None,
      max_length=12,
      description=(
          "Terminal growth axis (cols) for gordon_growth, decimals "
          "e.g. [0.02, 0.03, 0.04]."
      ),
  )
  terminal_ebitda: float | None = Field(
      None, ge=0, description="Year-N EBITDA for exit_multiple cells."
  )
  terminal_ebitda_multiples: list[float] | None = Field(
      None,
      max_length=12,
      description=(
          "EV/EBITDA multiple axis (cols) for exit_multiple, e.g. [10, 12, 14]."
      ),
  )
  net_debt: float = Field(
      0.0, description="Net debt subtracted from EV for equity value."
  )
  shares_outstanding: float | None = Field(
      None,
      gt=0,
      description=(
          "Diluted shares → per-share grid; omit for equity-value grid."
      ),
  )

  @model_validator(mode="after")
  def _check_axes(self):
    if self.terminal_method == "gordon_growth":
      if not self.terminal_growth_rates:
        raise ValueError("terminal_growth_rates is required for gordon_growth.")
    elif self.terminal_ebitda is None or not self.terminal_ebitda_multiples:
      raise ValueError(
          "terminal_ebitda and terminal_ebitda_multiples are required for"
          " exit_multiple."
      )
    return self


def _col_axis(req):
  return (
      req.terminal_growth_rates
      if req.terminal_method == "gordon_growth"
      else req.terminal_ebitda_multiples
  ) or []


def _cell_value(
    req, dr, col
):
  """One DCF at (discount_rate=dr, terminal driver=col).

  None if the cell is not a valid DCF (e.g. g ≥ r) — surfaced as a blank, never
  fudged.
  """
  kw: dict[str, Any] = {
      "fcf_schedule": req.fcf_schedule,
      "discount_rate": dr,
      "terminal_method": req.terminal_method,
      "net_debt": req.net_debt,
      "shares_outstanding": req.shares_outstanding,
  }
  if req.terminal_method == "gordon_growth":
    kw["terminal_growth_rate"] = col
  else:
    kw["terminal_ebitda"] = req.terminal_ebitda
    kw["terminal_ebitda_multiple"] = col
  try:
    res = compute_dcf(DCFRequest(**kw))
  except (ValidationError, ValueError, ZeroDivisionError):
    return None
  return (
      res["intrinsic_per_share"]
      if req.shares_outstanding
      else res["equity_value"]
  )


def compute_sensitivity(req):
  """Compute a DCF grid for the request's discount-rate and terminal axes."""

  cols = _col_axis(req)
  grid = [[_cell_value(req, dr, c) for c in cols] for dr in req.discount_rates]
  valid = [v for row in grid for v in row if v is not None]
  metric = "intrinsic_per_share" if req.shares_outstanding else "equity_value"
  return {
      "metric": metric,
      "terminal_method": req.terminal_method,
      "discount_rates": req.discount_rates,
      "terminal_axis": cols,
      "terminal_axis_label": (
          "terminal_growth_rate"
          if req.terminal_method == "gordon_growth"
          else "terminal_ebitda_multiple"
      ),
      "grid": grid,
      "low": min(valid) if valid else None,
      "high": max(valid) if valid else None,
      "n_valid": len(valid),
  }


def _fmt(v):
  return "  —  " if v is None else money(v)


def _markdown(res):
  per_share = res["metric"] == "intrinsic_per_share"
  is_gordon = res["terminal_method"] == "gordon_growth"
  col_fmt = (lambda c: pct(c)) if is_gordon else (lambda c: f"{num(c, 1)}x")
  header = (
      "WACC \\ "
      + ("g" if is_gordon else "exit×")
      + " | "
      + " | ".join(col_fmt(c) for c in res["terminal_axis"])
  )
  rows = [header, "—" * len(header)]
  for dr, row in zip(res["discount_rates"], res["grid"], strict=True):
    rows.append(f"{pct(dr):>6} | " + " | ".join(_fmt(v) for v in row))
  label = "intrinsic value/share" if per_share else "equity value"
  summary = (
      f"\n**Range ({label}): {money(res['low'])} – {money(res['high'])}**"
      if res["n_valid"]
      else "\nNo valid cells (every combination had g ≥ r)."
  )
  title = (
      "**DCF sensitivity** ("
      + ("Gordon growth" if is_gordon else "exit multiple")
      + f") — {label}\n"
  )
  return title + "\n".join(rows) + summary


async def _handler(req):
  res = compute_sensitivity(req)
  return ToolResponse(
      markdown=_markdown(res), structured=res, meta={"metric": res["metric"]}
  )


SPEC = ToolSpec(
    name="compute_valuation_dcf_sensitivity",
    display_name="compute.valuation.dcf_sensitivity",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=DCFSensitivityRequest,
    handler=_handler,
    tags=("compute", "valuation", "dcf", "sensitivity"),
)
