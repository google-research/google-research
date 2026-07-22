# Contributing to FinanceGym

FinanceGym is the benchmark + environment half of the FinanceHarness project. This repo accepts contributions to the data pipeline, the PIT search server, and the rubric judge.

## Scope

In scope:
- Bug fixes and quality improvements to the corpus, graph, question, curation, and judge pipelines.
- Performance and reliability improvements to the PIT search environment.
- Reproducibility and documentation fixes.
- New examples and CI improvements.

Out of scope for this repo:
- New agent implementations or harness changes.
- Training pipelines.

## Development setup

```bash
git clone git@github.com:FinanceHarness/FinanceGym.git
cd FinanceGym
pip install -e ".[dev,faiss-cpu]"
ruff check .
```

## Pull requests

- Open one PR per logical change.
- Run `ruff check .` locally before pushing.
- Reference the relevant section of [`docs/`](docs/) if your change touches a documented contract.

## Submitting benchmark results

Run the benchmark and submit your agent's report to appear on the [leaderboard](docs/leaderboard.md).
Submit via a **[Benchmark Submission issue](https://github.com/FinanceHarness/FinanceGym/issues/new?template=benchmark-submission.yml)**
(preferred) or by email — full instructions in [`docs/participate.md`](docs/participate.md).
Maintainers grade submissions with the reproducible judge documented in [`docs/grading.md`](docs/grading.md).
