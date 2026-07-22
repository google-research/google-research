# Participate: Run the Benchmark & Submit a Report

FinanceGym is a **point-in-time (PIT)** finance deep-research benchmark. Each task gives your agent
a `question` and a `cutoff` date; the agent must produce a cited research report using **only**
information available on or before the cutoff. Reports are graded against expert rubrics, and
results appear on the [leaderboard](leaderboard.md).

There are four steps: **download the questions → download the search environment → run your agent
→ submit a report**.

---

## 1. Download the questions

The public question set lives in this repo:

- [`../data/benchmark_400_public.jsonl`](../data/benchmark_400_public.jsonl) — 400 questions.

Each line is `{"task_id": ..., "question": ..., "cutoff": "YYYY-MM-DD"}`. Rubrics are withheld
(grading is maintainer-run). Schema details: [`../data/README.md`](../data/README.md).

```json
{"task_id": "69f904b728538874c086db16", "question": "How does Air France-KLM's accelerated transition ...", "cutoff": "2025-08-05"}
```

---

## 2. Download the point-in-time search environment

Agents retrieve evidence from a frozen news corpus served behind a search API. The corpus ships as
a **3-file bundle** (~932.8 GB total), hosted on Google Drive:

| File | What it is |
|------|-----------|
| `metadata.jsonl` | Per-document metadata (id, url, publication date). |
| `faiss_ivfsq8.bin` | FAISS vector index for semantic search. |
| `corpus.db` | SQLite store of full article text. |

```bash
mkdir -p search_env
# Download the 3 files from the FinanceGym Drive folder into search_env/:
#   <FILL: public Drive folder link — published by maintainers>
md5sum search_env/metadata.jsonl search_env/faiss_ivfsq8.bin search_env/corpus.db
# Compare against the published checksums:
#   <FILL: md5 checksums — published by maintainers>
```

> **Follow-up (maintainers):** publish the public Drive link + md5 checksums here.

---

## 3. Serve the search API & run your agent

Bring up the search services against the downloaded bundle (see the environment setup docs), then
have your agent talk to the point-in-time API. Three endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/embeddings` (port `8888`) | POST | Embed a query string. |
| `/search` (port `8889`) | POST | Semantic search. **Requires `max_date`** (= the task `cutoff`). |
| `/fetch` (port `8889`) | POST | Retrieve full article text by document id. |

**`max_date` is mandatory** — the server never returns documents published after the cutoff, which
is what enforces point-in-time compliance. This is the single hard rule; you may otherwise run any
agent architecture you like.

Minimal reference client:

```python
import requests

HOST = "http://<search-host>"

def encode(query: str) -> list[float]:
    r = requests.post(f"{HOST}:8888/v1/embeddings", json={"input": query}, timeout=30)
    return r.json()["data"][0]["embedding"]

def search(query: str, k: int, cutoff: str) -> list[dict]:
    emb = encode(query)
    r = requests.post(f"{HOST}:8889/search",
                      json={"query_embedding": emb, "k": k, "max_date": cutoff}, timeout=30)
    return r.json()["results"]

def fetch(doc_id: str) -> str:
    r = requests.post(f"{HOST}:8889/fetch", json={"id": doc_id}, timeout=30)
    return r.json()["text"]
```

For each benchmark question, pass its `cutoff` as `max_date` on every `search` call, gather
evidence, and write a cited report.

---

## 4. Submission format

Produce a **JSONL** file, one object per question:

| Field | Required | Description |
|-------|----------|-------------|
| `question` | ✅ | The benchmark question **verbatim** (used to match your answer to its rubric). |
| `cutoff` | ✅ | The task's cutoff (`YYYY-MM-DD`). |
| `report` | ✅ | Your agent's full cited research report. |
| `searches` | optional | List of query strings your agent issued. |
| `docs_retrieved` | optional | Number of documents fetched. |
| `steps` | optional | Number of search/reason iterations. |
| `elapsed_s` | optional | Wall-clock time in seconds. |

**Do not include the rubric in submissions.**

```json
{"question": "How does Air France-KLM's accelerated transition ...", "cutoff": "2025-08-05", "report": "**Equity Research Note**\n\n...", "searches": ["Air France-KLM SAS stake ..."], "docs_retrieved": 16, "steps": 7, "elapsed_s": 62.2}
```

---

## 5. Submit your report

Submit through one of two channels — **a GitHub issue (preferred)** or **email**. You keep your
`answers.jsonl`; maintainers run the judge and post the result.

### Option A — GitHub issue (preferred)

Open a **[Benchmark Submission issue](https://github.com/FinanceHarness/FinanceGym/issues/new?template=benchmark-submission.yml)**
on this repo and fill in the template:

- **Agent name** and **organization**
- **Base model** (the backbone your agent used, e.g. `gemini-flash-latest`)
- **A link to your `answers.jsonl`** (a release asset, gist, or cloud link) — attach the file to
  the issue if it is small enough
- **Contact** and any notes (harness, worker count, PIT-compliance confirmation)

### Option B — Email

Email your `answers.jsonl` plus the same metadata to the maintainers at
**`<FILL: submission email>`** with the subject `FinanceGym submission: <agent-name>`.

### What happens next

Maintainers run the reproducible judge over your `answers.jsonl` (see [grading.md](grading.md)),
then add your row to the [leaderboard](leaderboard.md). Submissions are single-blind: the rubric is
never shared, so no tuning to the rubric is possible.

Please include this metadata with either channel:

```yaml
agent: my-research-agent
org: Example Lab
base_model: your-model        # e.g. gemini-flash-latest
date: 2026-07-01
contact: you@example.com
notes: agent loop, 8 workers, PIT-compliant (max_date enforced on every search).
```
