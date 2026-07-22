---
name: dcf-valuation
description: Estimate a public equity's intrinsic value with a discounted-cash-flow model — pull fundamentals, set a discount rate via CAPM/WACC, project free cash flow, and discount it.
tags: [equity, valuation, dcf, intrinsic-value]
requires_tools:
  - data_equity_reference
  - data_equity_fundamentals
  - compute_valuation_wacc
  - compute_valuation_dcf
---

# DCF valuation

Estimate intrinsic value per share with a transparent, grounded DCF.

## Workflow

1.  **Anchor the company.** `data_equity_reference` for shares outstanding,
    beta, and current price; `data_equity_fundamentals` for revenue, free cash
    flow, margins, growth, and total cash/debt.
2.  **Discount rate.** `compute_valuation_wacc` — CAPM cost of equity from a
    risk-free rate, an equity risk premium, and the beta you pulled; add the
    debt trio for a WACC blend when leverage is material. Use the resulting
    rate.
3.  **Project FCF.** Build a short explicit free-cash-flow schedule (typically
    3–5 years) from the latest FCF and a growth path grounded in the
    fundamentals and trend. State the assumptions you used.
4.  **Value it.** `compute_valuation_dcf` with the FCF schedule, the discount
    rate, a terminal method (Gordon growth or exit multiple), net debt (debt −
    cash), and shares outstanding. Chain bulk inputs with `prev:` references
    where a prior tool already produced the series.
5.  **Report.** Intrinsic value per share vs. the current price, the discount
    rate and key assumptions, and how much of the value rests on the terminal
    value.

## Principles

-   Make every assumption explicit (growth, discount rate, terminal method) — a
    DCF is only as credible as its inputs, and the reader needs to see them.
-   Prefer chaining real figures (fundamentals → FCF → DCF) over restating
    numbers by hand; let the tools carry the arithmetic.
-   A high terminal-value share of enterprise value is worth flagging, not
    hiding — report the number and let the reader weigh it.
