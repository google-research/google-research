#!/usr/bin/env bash
# Run the full benchmark-construction pipeline end-to-end:
#   questions -> feasibility audit -> relevance + coherence -> reclassify
#   -> ILP balanced subset -> rubric/evidence packaging.
#
# This wrapper expects the per-stage Python entry points to exist as CLI
# helpers; for now it documents the call order and lets each stage be
# scripted directly via `python -c "from financegym.curation... import ..."`.

set -euo pipefail

QUESTIONS_IN="${QUESTIONS_IN:-output/benchmark/questions_raw.jsonl}"
OUT_DIR="${OUT_DIR:-output/benchmark}"
MODEL="${MODEL:-gemini-flash-latest}"
TARGET="${TARGET:-500}"

mkdir -p "$OUT_DIR"

echo "FinanceGym curation funnel"
echo "  input    : $QUESTIONS_IN"
echo "  output   : $OUT_DIR"
echo "  model    : $MODEL"
echo "  target N : $TARGET"
echo
echo "Run each stage via python -m / -c using the modules under financegym.curation:"
echo "  feasibility_audit.audit_one     -> audit per question"
echo "  relevance_filter.score_relevance"
echo "  coherence_filter.score_coherence"
echo "  reclassify_taxonomy.discover_taxonomy + apply_taxonomy"
echo "  ilp_balance.select_balanced(target=\$TARGET)"
echo "  package_rubric_evidence.annotate_question + export_article_texts"
