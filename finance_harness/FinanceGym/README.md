# FinanceGym

A point-in-time agentic finance research benchmark with a public retrieval
environment and a verifiable rubric judge.

FinanceGym ships:

-   **An end-to-end pipeline** to construct the benchmark from a public news
    corpus — corpus ingestion, finance entity graph extraction, situation-driven
    question generation, multi-stage curation, and rubric packaging.
-   **A point-in-time (PIT) search environment** (the `financegym.env` server)
    that agents query during evaluation, with strict `max_date` filtering so no
    future information leaks.
-   **A 5-tier rubric judge** with bootstrap-CI aggregation used to score agent
    outputs against expert-annotated rubrics.

FinanceGym is the **benchmark + retrieval environment**: any agent can be
evaluated against the same point-in-time retrieval and rubric-scoring contract.

## Status

The public **400-question** slim set ships in
[`data/`](data/benchmark_400_public.jsonl) and the
[leaderboard](docs/leaderboard.md) lives in this repo. The withheld full-rubric
set, grading results, and trained checkpoints are not part of this repository.

## Benchmark & Leaderboard

FinanceGym hosts the point-in-time finance deep-research benchmark (400
questions).

-   🏆 **[Leaderboard](docs/leaderboard.md)** — current standings across 18
    systems.
-   📥 **[Participate](docs/participate.md)** — download the questions + search
    environment, run your agent, and submit a report.
-   🧪 **[Grading](docs/grading.md)** — how submissions are scored
    (maintainer-facing, reproducible).
-   📦 **[Questions](data/benchmark_400_public.jsonl)** — the public 400-question
    slim set (rubric withheld).

## Quickstart

```bash
git clone git@github.com:FinanceHarness/FinanceGym.git
cd FinanceGym
pip install -e .

# Configure (Gemini API key for the LLM-as-judge and structured-output paths)
export GOOGLE_API_KEY=...
```

See [`docs/setup.md`](docs/setup.md) for hardware requirements (the PIT search
env serves a 100M+ document corpus and benefits from a GPU for embedding and
FAISS).

## Pipeline

```
web corpus (WARC archives, provided)
   └─► financegym.corpus      (extract clean text + publication date + embed)
       └─► financegym.index   (FAISS IVF-SQ8 index over embeddings)
           └─► financegym.env (FastAPI /search + /fetch with PIT max_date)

texts.jsonl + domain whitelist
   └─► financegym.graph       (Gemini structured-output → entity-relation edges)
       └─► financegym.questions (situation mining → cutoff → unconstrained gen)
           └─► financegym.curation (feasibility → relevance → coherence → ILP)
               └─► financegym.judge (5-tier rubric, bootstrap CI)
```

See [`docs/pipeline.md`](docs/pipeline.md) for the run order.

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License (CC BY-NC 4.0).

*Non-Commercial Use Only:* FinanceGym, its benchmark datasets, and retrieval
tools are provided strictly for non-commercial research and evaluation.
Commercial use, including integration into commercial financial advisory
services or using generated outputs to provide commercial investment advice, is
strictly prohibited.

## Citation

```bibtex
@misc{financeharness2026,
  title  = {FinanceHarness: Agentic Finance Research},
  author = {Anonymous},
  year   = {2026},
  note   = {Benchmark and environment available at \url{https://github.com/google-research/google-research/tree/master/google_research/finance_harness}}
}
```

