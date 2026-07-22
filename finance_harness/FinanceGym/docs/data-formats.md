# Data formats

Every on-disk JSONL format used between pipeline stages, with the
pydantic / TypedDict shape it mirrors and the module that owns it.

## `embeddings.bin`

Binary, owned by `financegym.corpus.extract_embed`.

```
offset  size  field
0       4     int32  nvecs
4       4     int32  dim
8       nvecs * dim * 4   float32 row-major embeddings
```

`pack_header` / `unpack_header` in `extract_embed` (and the matching
helpers in `financegym.index.build`) are the canonical encoders.

## `metadata.jsonl`

One line per article. Owned by `financegym.corpus.extract_embed`. Mirrors
`financegym.common.schemas.Document`.

```json
{
  "doc_id": "doc_0",
  "url": "https://example.com/x",
  "domain": "example.com",
  "pub_date": "2025-03-01",
  "crawl_date": "2025-03-01T00:00:00Z",
  "text_len": 7350
}
```

## `texts.jsonl`

One line per article, indexed by `doc_id`. The full article body is
deliberately kept out of `metadata.jsonl` so the index server can hold
metadata in RAM without paying for texts.

```json
{"doc_id": "doc_0", "text": "<full extracted article body>"}
```

## `corpus.db`

A SQLite database produced by `financegym.corpus.build_db`. Single table
`docs` with `doc_id`, `url`, `domain`, `pub_date`, `crawl_date`,
`text_len`, `text`. Indexes on `doc_id`, `domain`, and `pub_date`.

## `edges.csv`

Header + one row per extracted relation. Columns pinned in
`financegym.graph.extract_triples.EDGE_COLUMNS`:

```
head, relation, tail, context, url, domain, pub_date, crawl_date
```

## `finance_domains_clean.json`

The strict-pass domain whitelist:

```json
{
  "finance_domains": ["bloomberg.com", "ft.com", ...],
  "domain_doc_counts": {"bloomberg.com": 41234, ...},
  "removed_domains": ["..."],
  "metadata": {...}
}
```

## `relation_categories.json`

```json
{
  "categories": {
    "corporate_action":  ["acquired", "partnered_with", ...],
    "financial_report":  ["reported_revenue", ...],
    ...
  },
  "metadata": {...}
}
```

## `questions.jsonl`

The output of question generation (and the input to curation, judging,
and the leaderboard). Each line:

```json
{
  "question": "Why ...?",
  "thesis": "...",
  "entities": ["AAPL", "..."],
  "cutoff": "2025-03-31",
  "topic": "earnings",
  "sector": "technology",
  "reasoning_type": "causal",
  "situation_type": "tension_earnings_surprise",
  "rubric": [
    {"category": "antecedent", "criterion": "...", "source_edge_indices": [3, 5]},
    {"category": "consequent", "criterion": "...", "source_edge_indices": [1]}
  ],
  "metadata": {
    "pre_edge_evidence":  [...],
    "post_edge_evidence": [...],
    "source_urls_pre":    ["..."],
    "source_urls_post":   ["..."],
    "source_domains":     ["..."],
    "n_months": 6, "n_entities_in_evidence": 12, "n_categories_in_evidence": 3,
    "total_pre_edges": 47, "total_post_edges": 18
  }
}
```

`topic` and `reasoning_type` are populated by
`financegym.curation.reclassify_taxonomy.apply_taxonomy`.
`source_edge_indices` is populated by
`financegym.curation.package_rubric_evidence.annotate_question`.

## `answers.jsonl`

One line per agent submission, owned by the agent. The judge expects at
minimum:

```json
{
  "question": "<matches a question.question text>",
  "report": "<full agent report>",
  "elapsed_s": 123.4,
  "docs_retrieved": 17,
  "steps": 8
}
```

## `scores.jsonl`

One line per (question, agent) pair, written by
`financegym.judge.rubric_judge.judge_pair_to_record` and consumed by
`financegym.judge.aggregate`:

```json
{
  "agent": "agent-x",
  "question": "Why ...?",
  "cutoff": "2025-03-31",
  "topic": "earnings",
  "sector": "technology",
  "reasoning_type": "causal",
  "situation_type": "tension_earnings_surprise",
  "report_chars": 7421,
  "report_words": 1232,
  "scores": [
    {"item_idx": 0, "score": 3, "reasoning": "...", "category": "antecedent", "criterion": "..."}
  ],
  "antecedent_sum": 12, "antecedent_max": 16, "antecedent_norm": 0.75,
  "consequent_sum":  4, "consequent_max":  8, "consequent_norm": 0.5,
  "total_sum": 16, "total_max": 24, "total_norm": 0.6667,
  "score_dist": {"0": 1, "1": 0, "2": 2, "3": 1, "4": 2}
}
```
