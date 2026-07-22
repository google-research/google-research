# Pipeline

End-to-end run order. Each stage is independently scriptable; the
defaults in `configs/default.yaml` reproduce the canonical FinanceGym
build.

## 1 — Web corpus

Extract clean text + publication date from a web corpus (WARC archives you
provide) and embed it into the search store.

```bash
EMBED_URL=http://127.0.0.1:8888/v1/embeddings \
python -m financegym.corpus.extract_embed path/to/warc/ --output output/search/

python -m financegym.corpus.build_db --input output/search/
```

Output: `output/search/{embeddings.bin, metadata.jsonl, texts.jsonl, corpus.db}`.

## 2 — Index + env

Build the FAISS index and start the PIT search server.

```bash
python -m financegym.index.build --input output/search/ --index-type ivf_sq8
python -m financegym.env.server --data-dir output/search/ --port 8889
```

Verify with `EnvClient("http://localhost:8889").stats()`.

## 3 — Finance entity graph

Count → curate → cleanup the domain whitelist, then run the LLM
classify/extract over the whitelisted articles.

```bash
python -m financegym.graph.domain_whitelist count   output/search/metadata.jsonl \
    --output output/corpus/domain_counts.json
python -m financegym.graph.domain_whitelist curate  output/corpus/domain_counts.json \
    --output output/corpus/finance_domains_curated.json
python -m financegym.graph.domain_whitelist cleanup output/corpus/finance_domains_curated.json \
    --output output/corpus/finance_domains_clean.json

python -m financegym.graph.extract_triples \
    --texts    output/search/texts.jsonl \
    --metadata output/search/metadata.jsonl \
    --whitelist output/corpus/finance_domains_clean.json \
    --output    data/graphs/edges.csv

python -m financegym.graph.relation_categories count     --edges data/graphs/edges.csv \
    --output output/corpus/relation_counts.json
python -m financegym.graph.relation_categories categorize --counts output/corpus/relation_counts.json \
    --output output/corpus/relation_categories.json
```

## 4 — Question generation

Build the per-stage indexes, mine situations across three modes, score
each candidate (situation × cutoff) for viability, generate one
question per viable task, judge for quality.

```python
from financegym.questions.entity_filter import load_edges, load_rel_categories
from financegym.questions.event_detection import build_indexes
from financegym.questions.cutoff_selection import select_cutoffs_monthly
from financegym.questions.situation_mining import (
    mine_multihop_paths, mine_temporal_narratives, mine_tension_situations,
)
from financegym.questions.generate import generate_question
from financegym.questions.quality_judge import judge_question, passes_quality_gate

edges = load_edges("data/graphs/edges.csv")
cats  = load_rel_categories("output/corpus/relation_categories.json")
idx   = build_indexes(edges, cats)
cutoffs = select_cutoffs_monthly(edges)

situations = (
    mine_multihop_paths(edges, idx, cats)
    + mine_temporal_narratives(edges, idx, cats)
    + mine_tension_situations(edges, idx)
)

candidates = []
for sit in situations:
    for cutoff in cutoffs:
        q = generate_question(sit, cutoff, categories=cats)
        if q is None:
            continue
        verdict = judge_question(q)
        if passes_quality_gate(verdict):
            q["judgment"] = verdict
            candidates.append(q)
```

## 5 — Curation funnel

Each filter is one Gemini call per question; gates are enforced in
code via `passes_*` helpers.

```python
from financegym.curation.feasibility_audit  import audit_one, passes_audit
from financegym.curation.relevance_filter   import score_relevance, passes_relevance
from financegym.curation.coherence_filter   import score_coherence, passes_coherence
from financegym.curation.reclassify_taxonomy import discover_taxonomy, apply_taxonomy
from financegym.curation.ilp_balance        import select_balanced
from financegym.curation.package_rubric_evidence import annotate_question, export_article_texts

# 1) feasibility -> relevance -> coherence
gated = [q for q in candidates if passes_audit(audit_one(q))]
gated = [q for q in gated if passes_relevance(score_relevance(q))]
gated = [q for q in gated if passes_coherence(score_coherence(q))]

# 2) bottom-up taxonomy
tax = discover_taxonomy(gated)
gated, _ = apply_taxonomy(gated, tax)

# 3) ILP-balanced eval subset
eval_set = select_balanced(gated, target=500)

# 4) rubric ↔ source-edge mapping + article text export
for q in eval_set:
    annotate_question(q)
export_article_texts(eval_set, "output/search/metadata.jsonl", "output/search/corpus.db",
                     "output/benchmark/eval500_texts.jsonl")
```

## 6 — Scoring

Score one agent against the eval set, then aggregate.

```python
from financegym.judge.rubric_judge import judge_pair_to_record
from financegym.judge.aggregate    import macro, bootstrap_ci, paired_ci, axis_breakdown

questions = {q["question"]: q for q in eval_set}
records = []
for answer in agent_answers:                      # each row: {"question", "report", ...}
    q = questions.get(answer["question"])
    if q is None:
        continue
    rec = judge_pair_to_record("agent-x", q, answer)
    if rec is not None:
        records.append(rec)

print("macro:", macro(records))
print("95% CI:", bootstrap_ci(records))
print("by sector:", axis_breakdown(records, "sector"))
```

`scripts/build_search_env.sh`, `scripts/build_finance_graph.sh`,
`scripts/build_benchmark.sh`, and `scripts/run_judge.sh` wrap the
common invocations.
