# Reproducibility

Everything a third party needs to reproduce a FinanceGym submission and
its scoring is pinned in code; this document lists those pins together
so they are easy to audit.

## Pinned models

| Stage | Model | Notes |
|-------|-------|-------|
| Embedding (corpus) | `Qwen/Qwen3-Embedding-4B` | 2,560-dim, L2-normalized. Override via `EMBED_MODEL` / `EMBED_DIM` only when running an alternative contract. |
| LLM (graph + questions + curation + judge) | configurable via `FINANCEGYM_MODEL` | A current Gemini model (`financegym.common.llm.DEFAULT_MODEL`); not pinned to a version. Set `FINANCEGYM_MODEL` to a fixed id for a reproducible run. |

## Pinned prompts

The benchmark contract is the set of system prompts and question
templates used at each LLM call site. Each is exported as a
module-level constant so a single grep tells you whether anything has
changed:

| Stage | Module | Constant |
|-------|--------|----------|
| Domain whitelist (broad) | `financegym.graph.domain_whitelist` | `CURATE_SYSTEM` |
| Domain whitelist (strict) | `financegym.graph.domain_whitelist` | `CLEANUP_SYSTEM` |
| Triple extraction | `financegym.graph.extract_triples` | `SYSTEM_INSTRUCTION` |
| Relation categorization | `financegym.graph.relation_categories` | `CATEGORIZE_SYSTEM` |
| Question generation | `financegym.questions.generate` | `QUESTION_PROMPT_TEMPLATE` |
| Question quality judge | `financegym.questions.quality_judge` | `JUDGE_PROMPT_TEMPLATE` |
| Feasibility audit | `financegym.curation.feasibility_audit` | `AUDIT_PROMPT_TEMPLATE` |
| Relevance filter | `financegym.curation.relevance_filter` | `RELEVANCE_PROMPT_TEMPLATE` |
| Coherence filter | `financegym.curation.coherence_filter` | `COHERENCE_PROMPT_TEMPLATE` |
| Taxonomy discovery | `financegym.curation.reclassify_taxonomy` | `DISCOVER_PROMPT_TEMPLATE` |
| Taxonomy classify | `financegym.curation.reclassify_taxonomy` | `CLASSIFY_PROMPT_TEMPLATE` |
| Rubric ↔ edge mapping | `financegym.curation.package_rubric_evidence` | `MAP_SYSTEM_INSTRUCTION` |
| Rubric judge (system) | `financegym.judge.rubric_judge` | `JUDGE_SYSTEM` |
| Rubric judge (per-pair) | `financegym.judge.rubric_judge` | `build_prompt()` |

Any change to one of these is a change to the benchmark. Bump the
package minor version and note it in the release.

## Pinned gates

The hard accept rules are enforced in code, not by trusting the LLM's
verdict string:

| Gate | Module | Function |
|------|--------|----------|
| Question quality | `financegym.questions.quality_judge` | `passes_quality_gate` (avg ≥ 3.8, no dim == 1, naturalness ≥ 3) |
| Feasibility | `financegym.curation.feasibility_audit` | `passes_audit` (all 4 dims ≥ 5, flag == "none") |
| Relevance | `financegym.curation.relevance_filter` | `passes_relevance` (rel ≥ 5 AND naturalness ≥ 4) |
| Coherence | `financegym.curation.coherence_filter` | `passes_coherence` (coherence ≥ 4) |

## Seeds

| Source of randomness | Default seed | Where |
|----------------------|--------------|-------|
| Bootstrap CI | 42 | `financegym.judge.aggregate.DEFAULT_SEED` |
| Taxonomy discovery batches | 42 | `discover_taxonomy(seed=42)` |
| IVF training sample | 42 | `numpy.random.default_rng(42)` in `index.build` |

## Index parameters

| Parameter | Default | Where |
|-----------|---------|-------|
| Index type | `ivf_sq8` | `financegym.index.build.DEFAULT_INDEX_TYPE` |
| `nlist` | `clamp(sqrt(nvecs), 256, 65_536)` | `auto_nlist` |
| `nprobe` | 32 | both at build time and at server-load time |

## Bootstrap budget

The published bootstrap CIs use `n_boot = 1_000`. Per-axis breakdowns
use `n_boot = 400` to keep the per-axis cost bounded. Both defaults are
exported as `DEFAULT_N_BOOT` and `DEFAULT_AXIS_N_BOOT` in
`financegym.judge.aggregate`.

## Expected token cost

Rough order-of-magnitude figures using a current Gemini flash model. Real
cost depends on the corpus size and pool size:

| Stage | Calls per question | Total at 500 Q |
|-------|--------------------|----------------|
| Question quality judge | 1 | 500 |
| Feasibility audit | 1 | 500 |
| Relevance filter | 1 | 500 |
| Coherence filter | 1 | 500 |
| Taxonomy classify | 1 | 500 |
| Rubric ↔ edge mapping | `len(rubric)` | ≈3,000 |
| Rubric judge (per agent) | 1 | 500 per agent |

Token usage scales with the embedded evidence; a single rubric judge call
on a richly evidenced question uses around 8K input tokens.
