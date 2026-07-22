# API contract

The four endpoints below are the public contract between any agent and
the FinanceGym search environment. Schemas are pinned in
`financegym.env.server` as pydantic models so callers in any language
can mirror them.

## `POST /search`

**Request**

```json
{
  "query_embedding": [0.012, ...],
  "k": 10,
  "max_date": "2025-06-30"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `query_embedding` | list of floats | Must match the index's dim (2,560 for the canonical Qwen3-Embedding-4B contract). |
| `k` | int | Default 10. The server over-retrieves internally when a `max_date` filter is set. |
| `max_date` | string `YYYY-MM-DD` \| null | PIT cutoff. Documents with `pub_date > max_date` are excluded. |

**Response**

```json
{
  "results": [
    {
      "doc_id": "doc_3214",
      "url": "https://example.com/x",
      "domain": "example.com",
      "pub_date": "2025-05-12",
      "score": 0.873,
      "text_preview": "First 200 chars of the article ..."
    }
  ],
  "total_candidates": 50,
  "query_time_ms": 32.4
}
```

503 is returned if the index has not loaded yet. `query_time_ms` excludes
network roundtrip.

## `POST /fetch`

**Request**: `{"doc_id": "doc_3214"}`

**Response**:

```json
{
  "doc_id": "doc_3214",
  "text": "...full article text...",
  "url": "https://example.com/x",
  "domain": "example.com",
  "pub_date": "2025-05-12"
}
```

404 if the doc id is unknown.

## `GET /stats`

```json
{"total_docs": 122000000, "index_loaded": true, "db_loaded": true}
```

## `GET /health`

`{"status": "ok"}` once both the index and the text store have loaded.

## Judge contract

The judge consumes a question dict and an agent report string. Question
shape (the writable fields the judge depends on):

```json
{
  "question": "Why ...?",
  "thesis": "...",
  "cutoff": "2025-03-31",
  "rubric": [
    {"category": "antecedent",  "criterion": "..."},
    {"category": "consequent",  "criterion": "..."}
  ],
  "metadata": {
    "pre_edge_evidence":  [{"head": "...", "relation": "...", "tail": "...", "context": "..."}],
    "post_edge_evidence": [...],
    "source_urls_pre":  ["..."],
    "source_urls_post": ["..."]
  }
}
```

Per-item judge output:

```json
{"item_idx": 0, "score": 3, "reasoning": "...", "category": "antecedent", "criterion": "..."}
```

The score is a 0–4 integer (`0 = not addressed` through `4 = fully
grounded`). A record where every item starts with `"judge failed: "` is
a placeholder marking a persistent judge error; `is_judge_failure()`
flags it.
