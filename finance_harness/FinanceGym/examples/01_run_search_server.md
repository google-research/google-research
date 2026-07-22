# Run the PIT search server

End-to-end walkthrough from a prepared web corpus to a running search server.
You provide the raw corpus as WARC archives; this pipeline extracts, embeds,
indexes, and serves it.

## 1. Extract clean text + embed

`financegym.corpus.extract_embed` reads every `.warc.gz` under the input dir,
runs trafilatura + htmldate per record, and embeds it via an
OpenAI-compatible `/v1/embeddings` endpoint. The canonical embedder is
`Qwen3-Embedding-4B` (2,560 dim).

```bash
# Stand up the embedding server first (e.g. vLLM on port 8888),
# then point extract_embed at it via EMBED_URL.
EMBED_URL=http://127.0.0.1:8888/v1/embeddings \
python -m financegym.corpus.extract_embed \
    path/to/warc/ \
    --output output/search/ \
    --workers 96
```

Output: `embeddings.bin`, `metadata.jsonl`, `texts.jsonl`,
`checkpoint.txt` (resume-safe).

## 2. Build the SQLite text store

```bash
python -m financegym.corpus.build_db --input output/search/
```

Output: `output/search/corpus.db`.

## 3. Build the FAISS index

```bash
python -m financegym.index.build --input output/search/ --index-type ivf_sq8
```

Output: `output/search/faiss_index.bin`.

## 4. Serve

```bash
python -m financegym.env.server --data-dir output/search/ --host 0.0.0.0 --port 8889
```

Hit it from Python:

```python
from financegym.env.client import EnvClient

client = EnvClient("http://localhost:8889")
print(client.stats())            # {'total_docs': ..., 'index_loaded': True, 'db_loaded': True}
print(client.health())           # {'status': 'ok'}
```

Or run the whole sequence with one wrapper:

```bash
scripts/build_search_env.sh
```

Skip individual stages with `SKIP_EMBED=1`, `SKIP_DB=1`, `SKIP_INDEX=1`, or `SKIP_SERVE=1`.
