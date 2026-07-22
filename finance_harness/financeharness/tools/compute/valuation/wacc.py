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

"""compute.valuation.wacc — CAPM cost of equity + optional WACC blend.

The discount-rate input to the valuation pipeline (wacc → dcf →
dcf_sensitivity).
Pure-math. CAPM only (risk_free_rate, equity_risk_premium, beta) → cost of
equity
(WACC = Re). Add the debt trio (cost_of_debt, tax_rate, debt_to_equity,
all-or-none)
→ the weighted blend. Re = Rf + β·ERP; WACC = E/V·Re + D/V·Rd·(1−Tc) with
weights
from D/E. Exposes cost_of_equity + wacc as top-level fields for
chaining/sensitivity,
plus implied_equity_risk_premium and wacc_premium_over_rf.
"""

from __future__ import annotations

from typing import Any, Self

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.format import num, pct
from pydantic import BaseModel, Field, model_validator

_DESCRIPTION = (
    "Cost of capital via CAPM + optional WACC blend — the discount rate for"
    " DCF. Always: risk_free_rate, equity_risk_premium, beta (CAPM cost of"
    " equity). Optionally cost_of_debt, tax_rate, debt_to_equity (all-or-none)"
    " for a WACC blend with after-tax cost of debt. β can come from"
    " data.equity.reference; the risk-free rate from data.market.rates. Returns"
    " cost_of_equity and wacc as top-level fields plus"
    " implied_equity_risk_premium and wacc_premium_over_rf. Pass rates as"
    " decimals (0.045 not 4.5)."
)


class CostOfCapitalRequest(BaseModel):
  """Input for CAPM cost of equity and optional WACC blend."""

  risk_free_rate: float = Field(
      Ellipsis, ge=0, lt=1, description="Annual risk-free rate, decimal (0.045)."
  )
  equity_risk_premium: float = Field(
      Ellipsis, ge=0, lt=1, description="Equity risk premium (Rm−Rf), decimal."
  )
  beta: float = Field(
      Ellipsis,
      ge=-2,
      le=5,
      description="Equity beta. From data.equity.reference or supplied.",
  )
  cost_of_debt: float | None = Field(
      None,
      ge=0,
      lt=1,
      description="Pre-tax cost of debt, decimal (all-or-none).",
  )
  tax_rate: float | None = Field(
      None, ge=0, le=1, description="Effective tax rate, decimal (all-or-none)."
  )
  debt_to_equity: float | None = Field(
      None, ge=0, description="D/E ratio (all-or-none)."
  )

  @model_validator(mode="after")
  def _check_debt_all_or_none(self):
    set_count = sum(
        1
        for f in (self.cost_of_debt, self.tax_rate, self.debt_to_equity)
        if f is not None
    )
    if 0 < set_count < 3:
      raise ValueError(
          "cost_of_debt, tax_rate, and debt_to_equity must all be set (WACC"
          f" blend) or all unset (CAPM only). Got {set_count}/3."
      )
    return self


def compute_cost_of_capital(req):
  """Pure-math core (Pydantic enforced ranges + all-or-none)."""
  rf, erp, beta = req.risk_free_rate, req.equity_risk_premium, req.beta
  cost_of_equity = rf + beta * erp
  components: dict[str, Any] = {
      "risk_free_rate": rf,
      "equity_risk_premium": erp,
      "beta": beta,
  }

  if req.cost_of_debt is not None:
    rd, tc, de = req.cost_of_debt, req.tax_rate, req.debt_to_equity
    equity_weight = 1.0 / (1.0 + de)
    debt_weight = de / (1.0 + de)
    after_tax_cost_of_debt = rd * (1.0 - tc)
    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
    components.update({
        "cost_of_debt": rd,
        "tax_rate": tc,
        "debt_to_equity": de,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
    })
    method = "CAPM + WACC"
  else:
    wacc = cost_of_equity
    method = "CAPM only"

  return {
      "cost_of_equity": cost_of_equity,
      "wacc": wacc,
      "implied_equity_risk_premium": cost_of_equity - rf,
      "wacc_premium_over_rf": wacc - rf,
      "method": method,
      "components": components,
  }


def _markdown(res):
  c = res["components"]
  re_ = pct(res["cost_of_equity"])
  lines = [
      f"**Cost of capital** ({res['method']})",
      f"Re = Rf + b*ERP = {pct(c['risk_free_rate'])} + {num(c['beta'])}*{pct(c['equity_risk_premium'])} = **{re_}**",  # noqa: E501
      f"Implied ERP (Re-Rf = b*ERP): {pct(res['implied_equity_risk_premium'])}",
  ]
  if res["method"] == "CAPM + WACC":
    atrd = pct(c["after_tax_cost_of_debt"])
    lines += [
        (
            f"After-tax Rd = {pct(c['cost_of_debt'])}*(1-{pct(c['tax_rate'])})"
            f" = {atrd}"
        ),
        (
            f"Weights (D/E={num(c['debt_to_equity'], 3)}): E/V"
            f" {pct(c['equity_weight'])}, D/V {pct(c['debt_weight'])}"
        ),
        (
            f"**WACC = {pct(res['wacc'])}** · premium over Rf"
            f" {pct(res['wacc_premium_over_rf'])}"
        ),
    ]
  return "\n".join(lines)


async def _handler(req):
  res = compute_cost_of_capital(req)
  return ToolResponse(
      markdown=_markdown(res), structured=res, meta={"method": res["method"]}
  )


SPEC = ToolSpec(
    name="compute_valuation_wacc",
    display_name="compute.valuation.wacc",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=CostOfCapitalRequest,
    handler=_handler,
    tags=("compute", "valuation", "wacc", "capm", "cost-of-capital"),
)
