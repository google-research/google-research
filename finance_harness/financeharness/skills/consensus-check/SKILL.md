---
name: consensus-check
description: Summarize the sell-side view on an equity — analyst price targets, forward EPS/revenue estimates and ratings — and corroborate it against recent guidance and estimate revisions from the web.
tags: [equity, estimates, consensus, analyst]
requires_tools:
  - data_equity_estimates
---

# Consensus check

What the Street expects for a name, and whether recent news supports it.

## Workflow

1.  **Consensus snapshot.** `data_equity_estimates` for price targets
    (mean/median, range vs current), forward EPS and revenue by period with
    analyst counts and expected growth, and the buy/hold/sell spread.
2.  **Corroborate.** `search` + `visit` recent, authoritative coverage — the
    latest earnings/guidance, notable upgrades/downgrades, and the direction of
    estimate revisions. Ground each specific claim in a page you read.
3.  **Report.** The consensus expectation and the dispersion (tight vs wide
    targets), the rating skew, and whether recent developments are pushing
    estimates up or down.

## Principles

-   These are *estimates* — present them as consensus expectations, not facts,
    and note the dispersion (a wide target range is itself the signal).
-   Pair the tool's consensus numbers with what changed recently; a stale
    consensus and a fresh guidance cut tell different stories.
-   Attribute web-sourced specifics to the page you read; mark anything you
    couldn't confirm as unverified.
