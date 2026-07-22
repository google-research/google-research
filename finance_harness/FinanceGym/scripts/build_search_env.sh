#!/usr/bin/env bash
# Build the search environment end-to-end from a prepared web corpus:
#   extract + embed -> build SQLite -> build FAISS index -> serve.
#
# You provide the raw corpus as WARC archives in WARC_DIR.
#
# Env vars (all optional, sensible defaults):
#   WARC_DIR                         directory of .warc.gz archives you provide
#   SEARCH_DIR                       where embeddings.bin / metadata.jsonl /
#                                    texts.jsonl / corpus.db / faiss_index.bin go
#   EMBED_URL, EMBED_MODEL, EMBED_DIM  see financegym.corpus.extract_embed
#   INDEX_TYPE                       see financegym.index.build
#   HOST, PORT                       server bind
#   SKIP_EMBED / SKIP_DB / SKIP_INDEX / SKIP_SERVE  truthy to skip

set -euo pipefail

WARC_DIR="${WARC_DIR:-output/corpus/warc}"
SEARCH_DIR="${SEARCH_DIR:-output/search}"
INDEX_TYPE="${INDEX_TYPE:-ivf_sq8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8889}"
PYTHON="${PYTHON:-python}"

if [ -z "${SKIP_EMBED:-}" ]; then
    "$PYTHON" -m financegym.corpus.extract_embed "$WARC_DIR" --output "$SEARCH_DIR"
fi

if [ -z "${SKIP_DB:-}" ]; then
    "$PYTHON" -m financegym.corpus.build_db --input "$SEARCH_DIR"
fi

if [ -z "${SKIP_INDEX:-}" ]; then
    "$PYTHON" -m financegym.index.build --input "$SEARCH_DIR" --index-type "$INDEX_TYPE"
fi

if [ -z "${SKIP_SERVE:-}" ]; then
    "$PYTHON" -m financegym.env.server --data-dir "$SEARCH_DIR" --host "$HOST" --port "$PORT"
fi
