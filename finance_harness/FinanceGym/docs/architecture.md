# Architecture

FinanceGym is split into five service-grade modules that line up with the
data flow from raw web articles to scored agent submissions. Each module
publishes a small interface that the next consumes, so any of them can be
swapped or scaled independently.

```
   web corpus WARCs               questions.jsonl
        |                               |
        v                               v
 ┌──────────────┐  metadata.jsonl  ┌──────────┐
 │   corpus     │ ───────────────► │  graph   │ ─► edges.csv
 │              │  texts.jsonl     │          │
 └──────┬───────┘                  └────┬─────┘
        | embeddings.bin                |
        v                               v
   ┌─────────┐                    ┌───────────┐
   │  index  │                    │ questions │ ─► questions.jsonl
   │ (FAISS) │                    │ (v3 gen)  │
   └────┬────┘                    └─────┬─────┘
        | faiss_index.bin               |
        v                               v
   ┌────────────────┐              ┌──────────┐
   │      env       │              │ curation │ ─► curated.jsonl
   │ (FastAPI PIT)  │              └─────┬────┘
   └───┬────────────┘                    |
       |                                 v
  /search, /fetch                  ┌─────────┐
       ▲                           │  judge  │ ─► scores.jsonl
       │       agent answers       └─────────┘
       └───────────────────────────────┘
```

## Modules

### `financegym.corpus`

Streams web corpus WARCs through three concurrent stages:

1. **Reader** — one thread iterates WARC records.
2. **Extractors** — process pool runs trafilatura + htmldate per record.
3. **Embedders** — thread pool batches extracted articles and calls the
   embedding server.

Outputs `embeddings.bin` (8-byte `<ii>` header followed by float32),
`metadata.jsonl`, `texts.jsonl`, and a `checkpoint.txt` for resume.
`build_db` consolidates the text store into `corpus.db` so the search
server doesn't have to hold texts in RAM.

### `financegym.index`

Memory-maps `embeddings.bin` and builds a FAISS index (`ivf_sq8` by
default, also `ivf_flat`, `flat_sq8`, `flat`). Auto-detects faiss-gpu
and accelerates k-means training on GPUs when present, falling back
gracefully to CPU.

### `financegym.env`

The point-in-time search server. State (FAISS index, in-RAM metadata,
SQLite text store) is encapsulated in a `ServerState` object and the
FastAPI app is produced by a `create_app(state)` factory — clean for
testing and for any deployment that wants multiple servers per host.

The API contract is small and pinned (see [`api-contract.md`](api-contract.md)).

### `financegym.graph`

Builds the finance entity graph from the corpus output. Three
subcommands:

1. **`domain_whitelist`** — count → broad LLM curate → strict LLM cleanup.
2. **`extract_triples`** — Gemini structured output over whitelisted
   articles. The relation taxonomy (snake_case verbs) and the
   classify-then-extract prompt are pinned in-module.
3. **`relation_categories`** — count + LLM-categorize raw relations into
   the small taxonomy the question miners depend on.

### `financegym.questions`

The v3 question pipeline split into focused modules:

* **`entity_filter`** — shared blocklist, garbage patterns, sector
  normalization, edge loader, relation categorization.
* **`event_detection`** — `build_indexes` (one pass over edges produces
  every per-stage index) and `detect_event_days` (sigma-threshold day
  filter with a supplement path).
* **`cutoff_selection`** — per-day score (volume × diversity × entropy),
  per-month K from balance, gap enforcement.
* **`situation_mining`** — multi-hop paths, temporal narratives,
  tension situations.
* **`generate`** — evidence packers per situation type, pinned
  `QUESTION_PROMPT_TEMPLATE`, garbage-entity scrubbing, dedup.
* **`quality_judge`** — five-dimension judge with `passes_quality_gate`
  enforcing the hard accept rule in code.

### `financegym.curation`

The multi-stage filter funnel that turns the generated pool into the
balanced eval set:

| Stage | Module | Documented gate |
|-------|--------|-----------------|
| Feasibility | `feasibility_audit` | All 4 dims ≥ 5, flag == "none" |
| Relevance | `relevance_filter` | rel ≥ 5 AND naturalness ≥ 4 |
| Coherence | `coherence_filter` | coherence ≥ 4 |
| Reclassify | `reclassify_taxonomy` | discovered topic + reasoning labels |
| ILP balance | `ilp_balance` | per-axis fraction bounds |
| Package | `package_rubric_evidence` | rubric ↔ source-edge mapping + article text |

### `financegym.judge`

* **`rubric_judge`** — five-tier (0–4) Gemini judge with pinned
  `JUDGE_SYSTEM` and `build_prompt`. `_align_scores` defends against
  the LLM dropping items. Persistent failures emit placeholder
  records that `is_judge_failure` flags.
* **`aggregate`** — `macro`, `bootstrap_ci` (n=1000), `paired_ci`
  (matched by question), `axis_breakdown`, `score_dist`, `cost_stats`.

## Service boundaries

Three boundaries are load-bearing:

1. **Embedder ↔ index.** The embedding server is hot-swappable as long
   as it serves an OpenAI-compatible `/v1/embeddings` endpoint and
   produces 2,560-dim normalized vectors (the canonical contract;
   see [`reproducibility.md`](reproducibility.md)).
2. **Env ↔ agent.** Agents see only `/search`, `/fetch`, `/stats`,
   `/health`. They never reach into the on-disk format directly.
3. **Judge ↔ aggregator.** The judge writes one score record per
   (question, agent) pair. The aggregator consumes whatever axis
   labels the records carry, so adding a new axis only means stamping
   it on questions; no aggregator change is needed.
