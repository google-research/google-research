# Setup

## Requirements

* Python 3.12+.
* For the search environment: an OpenAI-compatible embedding server (e.g.
  vLLM hosting `Qwen3-Embedding-4B`) reachable from the box that runs
  `financegym.corpus.extract_embed` and `financegym.env.server`.
* For the index build at corpus scale: enough RAM to memory-map
  `embeddings.bin` (122M × 2,560 × 4 bytes ≈ 1.2 TB on disk; mmap keeps
  the resident set small). A GPU is optional but speeds up FAISS IVF-SQ8
  k-means training by 10–50×.
* For the LLM-driven stages (graph, questions, curation, judge): a
  Google GenAI API key in `GOOGLE_API_KEY`.

## Install

```bash
pip install -e ".[dev,faiss-cpu]"
```

Swap `faiss-cpu` for `faiss-gpu` to enable GPU-accelerated k-means.

## Environment variables

| Variable | Used by | Default |
|----------|---------|---------|
| `GOOGLE_API_KEY` | every Gemini call site | (required) |
| `GEMINI_MAX_ATTEMPTS` / `GEMINI_BASE_DELAY` / `GEMINI_MAX_DELAY` | `financegym.common.llm` | 6 / 2.0 / 64.0 |
| `EMBED_URL` | `corpus.extract_embed` | `http://127.0.0.1:8888/v1/embeddings` |
| `EMBED_MODEL` | same | `Qwen/Qwen3-Embedding-4B` |
| `EMBED_DIM` | same | `2560` |
| `JUDGE_CALL_TIMEOUT_S` | `judge.rubric_judge` | `300` |

The FinanceGym search server takes the point-in-time cutoff per-request as
`max_date`.

## Hardware footprint per stage

| Stage | Disk | RAM | Notes |
|-------|------|-----|-------|
| Corpus extract + embed | ≈10× the WARC size | depends on extractor workers (~4 GB / worker) | GPU bound on the embedder |
| `build_db` | ≈SUM(`texts.jsonl`) | small | one-time, sequential |
| `index.build` | ≈40% of `embeddings.bin` (IVF-SQ8) | mmap; peak in training only | GPU optional |
| `env.server` | reads `corpus.db` + `faiss_index.bin` lazily | metadata + index resident | 1 process |

## Sanity check

After install, lint to confirm the package imports and is well-formed:

```bash
ruff check .
```
