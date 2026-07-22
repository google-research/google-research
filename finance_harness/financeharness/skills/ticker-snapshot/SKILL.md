---
name: ticker-snapshot
description: Quick structured overview of a single public equity — identity, sector, market cap, valuation ratios, current price, recent trend.
tags: [equity, snapshot]
requires_tools:
  - data_equity_reference
  - data_equity_prices
---

# Ticker snapshot

Produce a tight, grounded overview of one equity.

## Workflow

1.  Pull `data_equity_reference` for identity, sector/industry, market cap,
    beta, current price, 52-week range, and P/E.
2.  Pull `data_equity_prices` (1y) for the recent trend — SMA(20/50/200), any
    golden/death cross, realized volatility.
3.  Write a short snapshot: what the company is, its scale and classification,
    where the price sits in its range, and the recent technical picture.

## Principles

-   Report the numbers and named canon the tools return (SMA, golden/death
    cross); let the figures speak. Note when a field is unavailable rather than
    guessing.
-   Keep it to the essentials a reader needs to orient on the name quickly.
