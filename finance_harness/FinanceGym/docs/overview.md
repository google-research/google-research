# Overview

FinanceGym is the **benchmark + environment** half of the FinanceHarness project. It is intentionally separated from the agent harness so that any agent — third-party, hosted, or our reference harness — can be evaluated against the same retrieval contract and the same scoring contract.

## The three artifacts

The published work distinguishes three artifacts, and the boundaries are load-bearing:

| Artifact | What it is | Repo |
|---|---|---|
| **FinanceEnv** | The point-in-time search sandbox: a finance-relevant slice of a web corpus, embedded with a pinned model, indexed with FAISS, served over a small REST API that enforces a `max_date` cutoff. | `financegym.env` (this repo) |
| **FinanceGym** | The benchmark: a question + rubric set built from a finance entity graph, situation mining, and a multi-stage curation funnel. Verified by expert annotation. | `financegym.questions` + `financegym.curation` + `financegym.judge` (this repo) |
| **FinanceHarness** | The agent harness: a layered orchestrator with tools and skills that runs *against* FinanceEnv + FinanceGym. | not in this repo |

## What this repo covers

In scope:
- Corpus ingestion (web corpus → clean text + publication date + embedding).
- FAISS index build and the PIT search server.
- Finance entity graph construction (domain whitelist → triple extraction → relation taxonomy).
- Question generation (event detection → situation mining → unconstrained LLM generation).
- Curation funnel (feasibility audit → relevance + naturalness → coherence → bottom-up taxonomy → ILP balancing → rubric/evidence packaging).
- Rubric judge and aggregation.

Out of scope:
- The agent that *answers* benchmark questions.
- The RL training pipeline.
- Leaderboard hosting.

## Reading order

- [`architecture.md`](architecture.md) — service boundaries and data flow.
- [`setup.md`](setup.md) — install, environment variables, hardware notes.
- [`pipeline.md`](pipeline.md) — end-to-end run order.
- [`api-contract.md`](api-contract.md) — locked API and JSON schemas.
- [`data-formats.md`](data-formats.md) — on-disk JSONL formats.
- [`reproducibility.md`](reproducibility.md) — seeds, model versions, and expected token costs.
