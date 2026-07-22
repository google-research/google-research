# FinanceGym Leaderboard

Normalized rubric means (%) on the **400-question** point-in-time finance deep-research benchmark.

**Evaluation set:** 400 expert-annotated questions · 2,464 rubric items · 0–4 rubric scale
normalized to %. **Hindsight** = pre-cutoff (antecedent) grounding; **Foresight** = post-cutoff
(consequent) reasoning; **±SE** = standard error of the Overall mean.

## Overall standings

| Rank | System | Category | Overall | Hindsight | Foresight | ±SE |
|-----:|--------|----------|--------:|----------:|----------:|----:|
| 1 | Claude-Opus-4.7 | Proprietary w/ search | 34.1 | 50.1 | 9.9 | 0.8 |
| 2 | Gemini-3.1-Pro | Proprietary w/ search | 33.2 | 46.8 | 12.8 | 0.7 |
| 3 | **FinanceHarness** | Open-weight w/ search | **32.4** | 45.7 | 11.8 | 0.8 |
| 4 | GPT-5.5 | Proprietary w/ search | 31.8 | 47.5 | 8.0 | 0.7 |
| 5 | TTD-DR | Agentic search | 31.5 | 45.5 | 9.8 | 0.7 |
| 6 | GLM-5 | Open-weight w/ search | 30.4 | 44.7 | 8.5 | 0.9 |
| 7 | GPT-Researcher | Agentic search | 30.4 | 42.2 | 12.0 | 0.8 |
| 8 | Gemini-3-Flash | Proprietary w/ search | 30.2 | 43.7 | 9.8 | 0.7 |
| 9 | DeepSeek-v3.2 | Open-weight w/ search | 28.9 | 42.4 | 8.4 | 0.8 |
| 10 | Tongyi-DR | Fine-tuned open-weight | 28.2 | 39.7 | 10.7 | 0.8 |
| 11 | deepagents | Agentic search | 28.1 | 40.7 | 8.6 | 1.1 |
| 12 | OpenClaw | Agentic search | 27.7 | 40.2 | 8.3 | 0.7 |
| 13 | STORM | Agentic search | 27.4 | 39.3 | 9.4 | 0.6 |
| 14 | OpenResearcher | Fine-tuned open-weight | 27.2 | 39.9 | 8.1 | 0.7 |
| 15 | Qwen3-235B-A22B | Open-weight w/ search | 26.8 | 39.7 | 7.3 | 0.7 |
| 16 | Gemma-4-26B | Open-weight w/ search | 25.7 | 37.6 | 7.7 | 0.6 |
| 17 | MiroThinker | Fine-tuned open-weight | 21.2 | 29.6 | 7.8 | 0.6 |
| 18 | gpt-oss-120b | Open-weight w/ search | 18.7 | 27.8 | 5.2 | 0.8 |

**FinanceHarness** is the top open-weight-with-search system, trailing only the two frontier
proprietary agents overall while leading its own category on every axis.

## By system category

Within each panel, **bold** marks the best and _underline_ the second best per column
(Overall / Hindsight / Foresight).

### Fine-tuned open-weight

| System | Overall | Hindsight | Foresight | ±SE |
|--------|--------:|----------:|----------:|----:|
| Tongyi-DR | **28.2** | **39.7** | **10.7** | 0.8 |
| OpenResearcher | _27.2_ | _39.9_ | _8.1_ | 0.7 |
| MiroThinker | 21.2 | 29.6 | 7.8 | 0.6 |

_(Note: OpenResearcher's Hindsight 39.9 slightly exceeds Tongyi-DR's 39.7; bold follows the
paper's Overall-led ranking.)_

### Agentic search

| System | Overall | Hindsight | Foresight | ±SE |
|--------|--------:|----------:|----------:|----:|
| TTD-DR | **31.5** | **45.5** | _9.8_ | 0.7 |
| GPT-Researcher | _30.4_ | _42.2_ | **12.0** | 0.8 |
| deepagents | 28.1 | 40.7 | 8.6 | 1.1 |
| OpenClaw | 27.7 | 40.2 | 8.3 | 0.7 |
| STORM | 27.4 | 39.3 | 9.4 | 0.6 |

### Proprietary with search

| System | Overall | Hindsight | Foresight | ±SE |
|--------|--------:|----------:|----------:|----:|
| Claude-Opus-4.7 | **34.1** | **50.1** | _9.9_ | 0.8 |
| Gemini-3.1-Pro | _33.2_ | 46.8 | **12.8** | 0.7 |
| GPT-5.5 | 31.8 | _47.5_ | 8.0 | 0.7 |
| Gemini-3-Flash | 30.2 | 43.7 | 9.8 | 0.7 |

### Open-weight with search

| System | Overall | Hindsight | Foresight | ±SE |
|--------|--------:|----------:|----------:|----:|
| **FinanceHarness** | **32.4** | **45.7** | **11.8** | 0.8 |
| GLM-5 | _30.4_ | _44.7_ | _8.5_ | 0.9 |
| DeepSeek-v3.2 | 28.9 | 42.4 | 8.4 | 0.8 |
| Qwen3-235B-A22B | 26.8 | 39.7 | 7.3 | 0.7 |
| Gemma-4-26B | 25.7 | 37.6 | 7.7 | 0.6 |
| gpt-oss-120b | 18.7 | 27.8 | 5.2 | 0.8 |

Per-topic, per-sector, and per-reasoning-type breakdowns are reported in the paper appendix.

## How to appear on this leaderboard

Run the benchmark and submit a report — see **[participate.md](participate.md)**. Submissions are
scored by maintainers using the reproducible pipeline in **[grading.md](grading.md)**.
