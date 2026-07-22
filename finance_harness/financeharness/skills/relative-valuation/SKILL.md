---
name: relative-valuation
description: Value an equity against its peers — peer-median trading multiples applied to the company's metrics for an implied range, cross-read against its own valuation ratios.
tags: [equity, valuation, comps, relative-valuation, multiples]
requires_tools:
  - data_equity_reference
  - data_equity_ratios
  - data_equity_comps
---

# Relative valuation

Where a name trades versus comparable companies, and whether that's deserved.

## Workflow

1.  **Identify the peer set.** `data_equity_reference` to classify the company
    (sector/industry); choose genuine comparables — similar business, scale, and
    growth/margin profile, not just same-sector names.
2.  **The company's own multiples.** `data_equity_ratios` for its P/E,
    EV/EBITDA, EV/revenue, P/S, plus the leverage/returns context that justifies
    a premium or discount.
3.  **Comps.** `data_equity_comps` with the peer tickers — peer-median multiples
    applied to the company's metrics give an implied value-per-share range vs
    the current price. Report the medians, implied values, and range it returns
    directly; they're already computed, so there's no need to re-derive them.
4.  **Read it.** Is the name rich or cheap versus peers, and is the gap
    warranted by superior growth, margins, or returns? State the implied range
    and the call.

## Principles

-   Peer choice drives the answer — name the comparables and why they fit; a bad
    peer set makes a precise-looking range meaningless.
-   A premium or discount is only a finding once you tie it to fundamentals
    (growth, margins, returns) — otherwise it's just a number.
-   Relative value complements, not replaces, an intrinsic (DCF) view — say
    which you're giving and pair them when it matters.
