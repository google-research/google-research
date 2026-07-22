#!/usr/bin/env bash
# Build the finance entity graph end-to-end:
#   domain_whitelist count -> curate -> cleanup
#   extract_triples
#   relation_categories count -> categorize.
#
# Env vars (all optional):
#   SEARCH_DIR        where metadata.jsonl / texts.jsonl live (from corpus stage)
#   CORPUS_OUT        output dir for whitelist + categorization artifacts
#   GRAPH_DIR         where the edge CSV lands
#   MODEL             Gemini model for the LLM steps
#   WORKERS           LLM concurrency for triple extraction
#   SKIP_*            truthy to skip a substep

set -euo pipefail

SEARCH_DIR="${SEARCH_DIR:-output/search}"
CORPUS_OUT="${CORPUS_OUT:-output/corpus}"
GRAPH_DIR="${GRAPH_DIR:-data/graphs}"
MODEL="${MODEL:-gemini-flash-latest}"
WORKERS="${WORKERS:-32}"
PYTHON="${PYTHON:-python}"

mkdir -p "$CORPUS_OUT" "$GRAPH_DIR"

if [ -z "${SKIP_WHITELIST:-}" ]; then
    "$PYTHON" -m financegym.graph.domain_whitelist count \
        "$SEARCH_DIR/metadata.jsonl" \
        --output "$CORPUS_OUT/domain_counts.json"
    "$PYTHON" -m financegym.graph.domain_whitelist curate \
        "$CORPUS_OUT/domain_counts.json" \
        --output "$CORPUS_OUT/finance_domains_curated.json" \
        --model "$MODEL"
    "$PYTHON" -m financegym.graph.domain_whitelist cleanup \
        "$CORPUS_OUT/finance_domains_curated.json" \
        --output "$CORPUS_OUT/finance_domains_clean.json" \
        --model "$MODEL"
fi

if [ -z "${SKIP_EXTRACT:-}" ]; then
    "$PYTHON" -m financegym.graph.extract_triples \
        --texts "$SEARCH_DIR/texts.jsonl" \
        --metadata "$SEARCH_DIR/metadata.jsonl" \
        --whitelist "$CORPUS_OUT/finance_domains_clean.json" \
        --output "$GRAPH_DIR/edges.csv" \
        --jsonl-output "$CORPUS_OUT/graph_results.jsonl" \
        --model "$MODEL" --workers "$WORKERS"
fi

if [ -z "${SKIP_CATEGORIES:-}" ]; then
    "$PYTHON" -m financegym.graph.relation_categories count \
        --edges "$GRAPH_DIR/edges.csv" \
        --output "$CORPUS_OUT/relation_counts.json"
    "$PYTHON" -m financegym.graph.relation_categories categorize \
        --counts "$CORPUS_OUT/relation_counts.json" \
        --output "$CORPUS_OUT/relation_categories.json" \
        --model "$MODEL"
fi
