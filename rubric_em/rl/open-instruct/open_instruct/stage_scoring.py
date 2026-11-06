# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file (_generate_stage_aware_adaptive_rubrics_batch and
# update_ground_truths_with_stage_aware_adaptive_rubrics) are adapted from
# AllenAI open-instruct / DR Tulu (https://github.com/rlresearch/dr-tulu),
# Copyright 2024 AllenAI, licensed under the Apache License, Version 2.0.

"""Stage-Wise Scoring & Adaptive Rubric Generation (v4).

Key changes from v3:
  - Schema restructured: stage_1/stage_2/stage_3/stage_4 dict (not flat list)
  - Negative rubric score INVERTED during aggregation (bug fix)
  - 2-6 rubrics PER STAGE (not total)
  - Weight field 1-3 preserved, agent self-rubric extraction preserved
"""

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types
import numpy as np
from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations
from pydantic import BaseModel
from pydantic import Field

LOGGER = logging.getLogger(__name__)


# =====================================================================
# 1. Pydantic Schemas
# =====================================================================


class RubricItem(BaseModel):
  """A single rubric without stage tag (stage comes from parent key)."""

  title: str = Field(
      description="Concise abstract label (general, not question-specific)"
  )
  description: str = Field(description="Detailed, specific description")
  weight: int = Field(
      default=2,
      description="Importance 1-3: 1=nice-to-have, 2=important, 3=critical",
  )


class StageRubrics(BaseModel):
  """Rubrics for one stage."""

  positive_rubrics: List[RubricItem] = Field(
      default_factory=list, description="Excellence indicators (2-6 items)"
  )
  negative_rubrics: List[RubricItem] = Field(
      default_factory=list,
      description="Critical flaws (0-4 items; fewer than positive is fine)",
  )


class StageAwareAdaptiveRubrics(BaseModel):
  """Rubrics organized by stage."""

  question: str = Field(description="The original question verbatim")
  stage_1: StageRubrics = Field(description="Plan stage rubrics")
  stage_2: StageRubrics = Field(description="Research stage rubrics")
  stage_3: StageRubrics = Field(description="Review stage rubrics")
  stage_4: StageRubrics = Field(description="Answer stage rubrics")


class StageAwareRubricEvaluation(BaseModel):
  rubric_index: int = Field(
      description="The index of the rubric within this stage's list."
  )
  score: float = Field(description="Score from 0 to 2.")
  justification: str = Field(description="Brief justification for the score.")


class StageEvaluationBlock(BaseModel):
  evaluations: List[StageAwareRubricEvaluation] = Field(
      default_factory=list,
      description="Evaluations for all rubrics in this stage.",
  )


class StageGroupedScoringResult(BaseModel):
  stage_1: Optional[StageEvaluationBlock] = Field(
      default=None, description="Plan stage evaluations"
  )
  stage_2: Optional[StageEvaluationBlock] = Field(
      default=None, description="Research stage evaluations"
  )
  stage_3: Optional[StageEvaluationBlock] = Field(
      default=None, description="Review stage evaluations"
  )
  stage_4: Optional[StageEvaluationBlock] = Field(
      default=None, description="Answer stage evaluations"
  )


# Legacy flat schema (kept for backward compatibility)
class StageAwareMultiRubricResult(BaseModel):
  evaluations: List[StageAwareRubricEvaluation] = Field(
      description="Evaluations for all provided rubrics."
  )


# Legacy schemas (for fixed process judge fallback)
class SingleStageEval(BaseModel):
  stage: int = Field(description="Stage number (1=Plan, 2=Research, 3=Review)")
  score: float = Field(description="Score from 0 to 2 for this stage")
  justification: str = Field(description="Brief justification for the score")


class ProcessStagesEvaluation(BaseModel):
  evaluations: List[SingleStageEval]


# =====================================================================
# 2. Stage-Aware Adaptive Rubric GENERATION
# =====================================================================

STAGE_AWARE_RUBRIC_GENERATION_PROMPT = """You are an expert evaluator generating stagewise comparative rubrics for multiple full trajectories produced by the SAME research agent on the SAME question.

## Goal
Identify the most discriminative, stage-local criteria that explain why some trajectories are better than others. Evaluate real process quality and answer quality — not XML compliance, verbosity, or shallow self-consistency.

## How to Read the Trajectories
Each trajectory is a highly-structured full execution trace, not just a final answer:

- **Stage 1 — Plan:** `<structured_plan>` with `<deep_analysis>`, `<rubric>`, `<research_plan>`
- **Stage 2 — Research:** iterative `<state_evaluation>`, tool calls, and possible plan/rubric updates
- **Stage 3 — Review:** `<review>` with `<rubric_review>` and `<writing_plan>`
- **Stage 4 — Answer:** final `<answer>`

Do not reward a stage just because the relevant tags exist. Judge whether that stage actually performs its intended function well.

## How to Use Different Stages
Use the stages in different ways:

- **Stage 1 and Stage 4 can be used as soft references across trajectories.**
  - Strong Stage 1 analyses/rubrics can reveal important task dimensions, hidden constraints, and failure modes.
  - Strong Stage 4 answers can reveal what good coverage, synthesis, support, and format compliance look like.
  - These are useful for discovering candidate rubric dimensions.

- **Stage 2 and Stage 3 are more trajectory-dependent.**
  - Judge them mainly from what that trajectory actually did:
    - how it searched,
    - how it updated or corrected itself,
    - how it assessed evidence,
    - how honestly it prepared to answer.

## Important Principle
A trajectory's own Stage 1 `<rubric>` is never the scoring standard.

Use it only as a reference signal:
- it shows what the agent recognized as important,
- it may surface useful task dimensions,
- it may expose shallow planning if it misses critical requirements.

But:
- never reward a trajectory simply for following a weak self-rubric,
- never let a narrow self-rubric lower the real task standard,
- never penalize evidence-based improvement beyond the original plan.

## Stage-by-Stage Focus

### Stage 1 — PLAN
Ask: Did the agent understand the problem and set itself up for success?

Focus on:
- whether `<deep_analysis>` identifies the task dimensions that later prove necessary,
- whether it goes beyond paraphrasing and surfaces implicit needs, hidden constraints, or likely failure modes,
- whether `<rubric>` is specific, testable, and genuinely useful,
- whether `<research_plan>` is adaptive and proportional to the task,
- whether the planning effort is well-calibrated.

Use strong Stage 1 plans/rubrics across trajectories as references for what good task understanding looks like.

### Stage 2 — RESEARCH
Ask: Did the agent search effectively and adapt to evidence?

Focus on:
- specificity and usefulness of tool queries,
- whether searches are aligned with actual information gaps,
- whether `<state_evaluation>` correctly judges evidence quality and sufficiency,
- whether the agent pivots when assumptions fail,
- whether it performs multi-hop triangulation instead of isolated or redundant searches,
- whether it stops at the right time.

This stage should be judged mainly from the trajectory's actual research behavior, not from reference answers.

### Stage 3 — REVIEW
Ask: Did the agent honestly audit readiness and convert evidence into a concrete drafting plan?

Focus on:
- whether `<rubric_review>` meaningfully maps evidence back to real task needs,
- whether it acknowledges unresolved gaps or uncertainty honestly,
- whether it turns abstract requirements into concrete drafting directives,
- whether `<writing_plan>` maps evidence and citations to answer structure in a useful way,
- whether the review is appropriately scaled to task complexity.

This stage should also be judged mainly from the trajectory's own review quality.

### Stage 4 — ANSWER
Ask: Is the final `<answer>` itself high-quality enough for the user's question?

Stage 4 should be evaluated independently from the earlier trajectory. The scoring target is the final `<answer>` itself.

The main standard is:
- the original question,
- the requested output format,
- and the quality of the final answer as supported by evidence.

When designing Stage 4 rubrics, you may compare multiple final answers to discover the most objective and discriminative answer-quality dimensions. As a secondary aid, you may also read strong Stage 1 rubrics across trajectories, since they may reveal useful candidate dimensions.

However, once the Stage 4 rubric is formulated, it should judge only the final `<answer>`, not whether that trajectory followed its own earlier `<deep_analysis>`, `<rubric>`, `<rubric_review>`, or `<writing_plan>`.

Focus on:
- whether the answer directly and adequately addresses the question,
- whether it is factually correct and appropriately scoped,
- whether it is sufficiently complete for the task,
- whether citations are used correctly and actually support the attached claims,
- whether uncertainty or conflicting evidence is handled appropriately,
- whether the answer synthesizes rather than merely lists facts,
- whether it respects the requested format and depth,
- whether it avoids unsupported claims, contradiction, overstatement, and citation misuse.

## What Makes a Good Rubric
A good rubric:
- separates stronger and weaker trajectories for the SAME question,
- is specific to that stage's real responsibility,
- captures genuine competence rather than style preference,
- is observable from the trajectories,
- is not redundant with existing rubrics,
- is not a mirror duplicate of another rubric.

Before keeping a rubric, check:
- Does it actually separate provided trajectories?
- Would it still matter if the model had written a different self-rubric?
- Is it stage-local rather than vaguely global?
- Is it specific enough to be actionable?

## Anti-Hack Rules
Do NOT create rubrics that:
- only check whether a tag exists,
- reward formal obedience to a bad plan,
- overfit to one trajectory's wording or idiosyncratic rubric,
- confuse self-consistency with true quality,
- punish evidence-based improvement,
- reward unnecessary overhead on simple tasks.

## Selection Rules
- Prefer fewer, sharper rubrics over many generic ones.
- Include only criteria that meaningfully discriminate among the provided trajectories.
- Skip criteria where all trajectories perform similarly.
- Never create positive and negative mirror versions of the same idea within a stage.
- Use conservative negative rubrics: target clear failure modes, not mere absence of excellence.

## Weights
Use:
- **3** = critical
- **2** = important
- **1** = minor but meaningful

Guidance:
- Positive rubrics may use 1-3.
- Negative rubrics should usually be 1.
- Use negative 2 only for serious flaws such as fabricated citations, contradiction of retrieved evidence, or clearly premature stopping.
- Use negative 3 only when the flaw nearly invalidates the stage.

## Output Format
Return rubrics by stage:
- `stage_1`
- `stage_2`
- `stage_3`
- `stage_4`

Each stage contains:
- `positive_rubrics`
- `negative_rubrics`

Each rubric item must include:
- `title`
- `description`
- `weight`

Target counts:
- Stage 1: 2-6 total
- Stage 2: 2-6 total
- Stage 3: 2-6 total
- Stage 4: 3-8 total

These are targets, not quotas. Return fewer if more would be redundant.

## Inputs
1. **Question**
2. **Trajectories**
3. **Existing Rubrics** (optional)

## Final Reminder
This is a stagewise comparative rubric generator.

- Use **Stage 1** and **Stage 4** across trajectories as reference pools for discovering high-value rubric dimensions.
- Judge **Stage 2** and **Stage 3** more directly from each trajectory's actual behavior.
- For **Stage 4**, design the rubric with cross-trajectory insight, but score only the final answer itself.

Generate only the most discriminative, non-redundant, stage-specific rubrics."""


# pylint: disable=invalid-name
STAGE_AWARE_RUBRIC_GENERATION_PROMPT_v2 = """You are an expert evaluator generating stagewise comparative rubrics for multiple full trajectories produced by the SAME research agent on the SAME question.

## Goal
Identify the most discriminative, stage-local criteria that explain why some trajectories are better than others. Evaluate real process quality and answer quality — not XML compliance, verbosity, or shallow self-consistency.

## How to Read the Trajectories
Each trajectory is a highly-structured full execution trace, not just a final answer:

- **Stage 1 — Plan:** `<structured_plan>` with `<deep_analysis>`, `<rubric>`, `<research_plan>`
- **Stage 2 — Research:** iterative `<state_evaluation>`, tool calls, and possible plan/rubric updates
- **Stage 3 — Review:** `<review>` with `<rubric_review>` and `<writing_plan>`
- **Stage 4 — Answer:** final `<answer>`

Do not reward a stage just because the relevant tags exist. Judge whether that stage actually performs its intended function well.

## How to Use Different Stages
Use the stages in different ways:

- **Stage 1 and Stage 4 can be used as soft references across trajectories.**
  - Strong Stage 1 analyses/rubrics can reveal important task dimensions, hidden constraints, and failure modes.
  - Strong Stage 4 answers can reveal what good coverage, synthesis, support, and format compliance look like.
  - These are useful for discovering candidate rubric dimensions.

- **Stage 2 and Stage 3 rubrics should capture trajectory-dependent behaviors.**
  - Compare how different trajectories searched, adapted, and reviewed to discover the dimensions that actually separate better from worse execution.
  - The rubrics you generate will be applied to each trajectory independently — so focus on criteria that are observable from a single trajectory's behavior, even though you use cross-trajectory comparison to identify them.

## Important Principle
A trajectory's own Stage 1 `<rubric>` is never the scoring standard.

Use it only as a reference signal:
- it shows what the agent recognized as important,
- it may surface useful task dimensions,
- it may expose shallow planning if it misses critical requirements.

But:
- never reward a trajectory simply for following a weak self-rubric,
- never let a narrow self-rubric lower the real task standard,
- never penalize evidence-based improvement beyond the original plan.

## Stage-by-Stage Guidance

For each stage, the most discriminative criteria often emerge from the specific question and trajectories themselves. The examples below are common dimensions that have proven useful in the past — use them as starting points, but do not limit yourself to them. The best rubrics capture what matters for THIS question.

Each rubric must be specific, discriminative, and detailed: the description should reference concrete aspects of the question or task so that a separate scorer can unambiguously judge whether a trajectory meets the criterion. Avoid vague rubrics like "good research quality" — instead specify what concretely makes research good for this particular question.

### Stage 1 — PLAN
Ask: Did the agent understand the problem and set itself up for success?

Common useful dimensions include task understanding depth, identification of hidden constraints, rubric specificity, and plan proportionality — but the most discriminative criterion is specific to the question at hand. Use strong plans across trajectories as references to discover what good task understanding looks like for this question.

### Stage 2 — RESEARCH
Ask: Did the agent search effectively and adapt to evidence?

Common useful dimensions include query specificity, evidence-based pivoting, triangulation, and stopping criteria — but freely discover what actually separates better from worse research behavior in these specific trajectories.

### Stage 3 — REVIEW
Ask: Did the agent honestly audit readiness and prepare a concrete drafting plan?

Common useful dimensions include honest gap assessment, evidence-to-structure mapping, and quality of the drafting plan — but the most telling criterion is specific to the question at hand. The review's purpose is to prepare a strong final answer, so judge it by whether the draft plan would actually lead to good coverage, synthesis, and grounding.

### Stage 4 — ANSWER
Ask: Is the final `<answer>` itself high-quality for the user's question?

Stage 4 should be evaluated independently from the earlier trajectory. The scoring target is the final `<answer>` itself, judged against the original question.

Common useful dimensions include readability, instruction following, insightfulness, comprehensiveness, correctness, and evidence grounding — but compare the actual answers to discover what specifically separates stronger from weaker responses for this question. Do not default to generic quality checklists.

Do not create separate rubrics for overlapping concerns (e.g., "factually correct" and "avoids unsupported claims" are the same dimension). Pick only the dimensions where trajectories actually differ.

## What Makes a Good Rubric
A good rubric:
- separates stronger and weaker trajectories for the SAME question,
- is specific to that stage's real responsibility,
- captures genuine competence rather than style preference,
- is observable from the trajectories,
- is not redundant with existing rubrics,
- is not a mirror duplicate of another rubric.

Before keeping a rubric, check:
- Does it actually separate provided trajectories?
- Would it still matter if the model had written a different self-rubric?
- Is it stage-local rather than vaguely global?
- Is it specific enough to be actionable?

## Anti-Hack Rules
Do NOT create rubrics that:
- only check whether a tag exists,
- reward formal obedience to a bad plan,
- overfit to one trajectory's wording or idiosyncratic rubric,
- confuse self-consistency with true quality,
- punish evidence-based improvement,
- reward unnecessary overhead on simple tasks.

## Selection Rules
- Prefer fewer, sharper rubrics over many generic ones.
- Include only criteria that meaningfully discriminate among the provided trajectories.
- Skip criteria where all trajectories perform similarly.
- Never create positive and negative mirror versions of the same idea within a stage.
- Use conservative negative rubrics: target clear failure modes, not mere absence of excellence.

## Weights
Use:
- **3** = critical
- **2** = important
- **1** = minor but meaningful

Guidance:
- Positive rubrics may use 1-3.
- Negative rubrics should usually be 1.
- Use negative 2 only for serious flaws such as fabricated citations, contradiction of retrieved evidence, or clearly premature stopping.
- Use negative 3 only when the flaw nearly invalidates the stage.

## Output Format
Return rubrics by stage:
- `stage_1`
- `stage_2`
- `stage_3`
- `stage_4`

Each stage contains:
- `positive_rubrics`
- `negative_rubrics`

Each rubric item must include:
- `title`
- `description`
- `weight`

Target counts:
- Stage 1: 1-4 total
- Stage 2: 1-4 total
- Stage 3: 1-4 total
- Stage 4: 2-5 total

These are targets, not quotas. Return fewer if more would be redundant.

## Inputs
1. **Question**
2. **Trajectories**
3. **Existing Rubrics** (optional)

## Final Reminder
Generate only the most discriminative, non-redundant, stage-specific rubrics. Fewer sharp rubrics are always better than many generic ones."""

STAGE_AWARE_RUBRIC_GENERATION_PROMPT_v2_Q_GROUNDED = """You are an expert evaluator generating stagewise comparative rubrics for multiple full trajectories produced by the SAME research agent on the SAME question.

## Goal
Identify the most discriminative, stage-local criteria that explain why some trajectories are better than others. Evaluate real process quality and answer quality — not XML compliance, verbosity, or shallow self-consistency.

Before examining the trajectories, consider: what does this question actually need? A patient asking about side effects needs different things than a researcher asking about enzyme kinetics. Use the question's actual needs — not the trajectories' shared approach — as your anchor for what "good" looks like.

## How to Read the Trajectories
Each trajectory is a highly-structured full execution trace, not just a final answer:

- **Stage 1 — Plan:** `<structured_plan>` with `<deep_analysis>`, `<rubric>`, `<research_plan>`
- **Stage 2 — Research:** iterative `<state_evaluation>`, tool calls, and possible plan/rubric updates
- **Stage 3 — Review:** `<review>` with `<rubric_review>` and `<writing_plan>`
- **Stage 4 — Answer:** final `<answer>`

Do not reward a stage just because the relevant tags exist. Judge whether that stage actually performs its intended function well.

## How to Use Different Stages
Use the stages in different ways:

- **Stage 1 and Stage 4 can be used as soft references across trajectories.**
  - Strong Stage 1 analyses/rubrics can reveal important task dimensions, hidden constraints, and failure modes.
  - Strong Stage 4 answers can reveal what good coverage, synthesis, support, and format compliance look like.
  - These are useful for discovering candidate rubric dimensions.

- **Stage 2 and Stage 3 rubrics should capture trajectory-dependent behaviors.**
  - Compare how different trajectories searched, adapted, and reviewed to discover the dimensions that actually separate better from worse execution.
  - The rubrics you generate will be applied to each trajectory independently — so focus on criteria that are observable from a single trajectory's behavior, even though you use cross-trajectory comparison to identify them.

## Important Principle
A trajectory's own Stage 1 `<rubric>` is never the scoring standard.

Use it only as a reference signal:
- it shows what the agent recognized as important,
- it may surface useful task dimensions,
- it may expose shallow planning if it misses critical requirements.

But:
- never reward a trajectory simply for following a weak self-rubric,
- never let a narrow self-rubric lower the real task standard,
- never penalize evidence-based improvement beyond the original plan.

## Stage-by-Stage Guidance

For each stage, the most discriminative criteria often emerge from the specific question and trajectories themselves. The examples below are common dimensions that have proven useful in the past — use them as starting points, but do not limit yourself to them. The best rubrics capture what matters for THIS question.

Each rubric must be specific, discriminative, and detailed: the description should reference concrete aspects of the question or task so that a separate scorer can unambiguously judge whether a trajectory meets the criterion. Avoid vague rubrics like "good research quality" — instead specify what concretely makes research good for this particular question.

### Stage 1 — PLAN
Ask: Did the agent understand the problem and set itself up for success?

Common useful dimensions include task understanding depth, identification of hidden constraints, rubric specificity, and plan proportionality — but the most discriminative criterion is specific to the question at hand. Use strong plans across trajectories as references to discover what good task understanding looks like for this question.

Also consider whether the plan's complexity is proportionate to the question. A plan that creates multi-phase technical frameworks for a simple consumer question is over-engineered. Thoroughness on a complex research question is good; the same elaboration on a straightforward question is a flaw.

### Stage 2 — RESEARCH
Ask: Did the agent search effectively and adapt to evidence?

Common useful dimensions include query specificity, evidence-based pivoting, triangulation, and stopping criteria — but freely discover what actually separates better from worse research behavior in these specific trajectories.

### Stage 3 — REVIEW
Ask: Did the agent honestly audit readiness and prepare a concrete drafting plan?

Common useful dimensions include honest gap assessment, evidence-to-structure mapping, and quality of the drafting plan — but the most telling criterion is specific to the question at hand. The review's purpose is to prepare a strong final answer, so judge it by whether the draft plan would actually lead to good coverage, synthesis, and grounding.

### Stage 4 — ANSWER
Ask: Is the final `<answer>` itself high-quality for the user's question?

Stage 4 should be evaluated independently from the earlier trajectory. The scoring target is the final `<answer>` itself, judged against the original question.

Common useful dimensions include readability, instruction following, insightfulness, comprehensiveness, correctness, and evidence grounding — but compare the actual answers to discover what specifically separates stronger from weaker responses for this question. Do not default to generic quality checklists.

When the answer uses specific technical terminology, metrics, or named frameworks, verify they are standard domain vocabulary — not invented. Fabricated terminology (made-up metric names, invented "rules" or "principles" with dramatic capitalization) presented as authoritative is a flaw. The model's internal stages can freely use shorthand — only the final answer's claims need to be verifiable.

Also consider whether the answer's tone matches the likely audience. A patient needs empathetic plain-language guidance. A casual consumer needs direct recommendations. A domain expert expects precise terminology. Mismatched register is a flaw. The model's internal stages can use any tone — only the final answer needs appropriate register.

Do not create separate rubrics for overlapping concerns (e.g., "factually correct" and "avoids unsupported claims" are the same dimension). Pick only the dimensions where trajectories actually differ.

## What Makes a Good Rubric
A good rubric:
- separates stronger and weaker trajectories for the SAME question,
- is specific to that stage's real responsibility,
- captures genuine competence rather than style preference,
- is observable from the trajectories,
- is not redundant with existing rubrics,
- is not a mirror duplicate of another rubric.

Before keeping a rubric, check:
- Does it actually separate provided trajectories?
- Would it still matter if the model had written a different self-rubric?
- Is it stage-local rather than vaguely global?
- Is it specific enough to be actionable?

## Anti-Hack Rules
Do NOT create rubrics that:
- only check whether a tag exists,
- reward formal obedience to a bad plan,
- overfit to one trajectory's wording or idiosyncratic rubric,
- confuse self-consistency with true quality,
- punish evidence-based improvement,
- reward unnecessary overhead on simple tasks,
- reward fabricated terminology, invented framework names, or dramatic proper-noun labeling that does not correspond to real domain concepts,
- reward elaborate-looking but shallow content — models trained with RL can learn to game rubrics by producing authoritative-sounding invented terminology, extreme length, or dramatic framework names without corresponding substance.

## Selection Rules
- Prefer fewer, sharper rubrics over many generic ones.
- Include only criteria that meaningfully discriminate among the provided trajectories.
- Skip criteria where all trajectories perform similarly.
- Exception: structural quality issues (proportionality, register, fabricated terminology) should be flagged even if all trajectories share the pattern — these indicate systematic model artifacts, not universal quality standards.
- Never create positive and negative mirror versions of the same idea within a stage.
- Use conservative negative rubrics: target clear failure modes, not mere absence of excellence.

## Weights
Use:
- **3** = critical
- **2** = important
- **1** = minor but meaningful

Guidance:
- Positive rubrics may use 1-3.
- Negative rubrics should usually be 1.
- Use negative 2 only for serious flaws such as fabricated citations, fabricated terminology presented as domain-standard, contradiction of retrieved evidence, or clearly premature stopping.
- Use negative 3 only when the flaw nearly invalidates the stage.

## Output Format
Return rubrics by stage:
- `stage_1`
- `stage_2`
- `stage_3`
- `stage_4`

Each stage contains:
- `positive_rubrics`
- `negative_rubrics`

Each rubric item must include:
- `title`
- `description`
- `weight`

Target counts:
- Stage 1: 1-4 total
- Stage 2: 1-4 total
- Stage 3: 1-4 total
- Stage 4: 2-5 total

These are targets, not quotas. Return fewer if more would be redundant.

## Inputs
1. **Question**
2. **Trajectories**
3. **Existing Rubrics** (optional)

## Final Reminder
Generate only the most discriminative, non-redundant, stage-specific rubrics. Fewer sharp rubrics are always better than many generic ones."""
# pylint: enable=invalid-name

STAGE_AWARE_RUBRIC_GEN_PROMPTS = {
    "default": STAGE_AWARE_RUBRIC_GENERATION_PROMPT_v2,
    "q_grounded": STAGE_AWARE_RUBRIC_GENERATION_PROMPT_v2_Q_GROUNDED,
}

STAGE_AWARE_RUBRIC_GENERATION_PROMPT_CONSERVATIVE = """You are an expert evaluator generating stagewise comparative rubrics for multiple full trajectories produced by the SAME research agent on the SAME question.

## Goal
Identify only the most critical, high-impact criteria that explain why some trajectories are clearly better than others. Be highly selective — generate fewer, sharper rubrics rather than comprehensive coverage.

## How to Read the Trajectories
Each trajectory is a highly-structured full execution trace:

- **Stage 1 — Plan:** `<structured_plan>` with `<deep_analysis>`, `<rubric>`, `<research_plan>`
- **Stage 2 — Research:** iterative `<state_evaluation>`, tool calls, and possible plan/rubric updates
- **Stage 3 — Review:** `<review>` with `<rubric_review>` and `<writing_plan>`
- **Stage 4 — Answer:** final `<answer>`

Do not reward a stage just because the relevant tags exist. Judge whether that stage actually performs its intended function well.

## Stage-by-Stage Focus

### Stage 1 — PLAN
Did the agent understand the problem and set itself up for success?
Focus on: whether analysis surfaces implicit needs, whether rubric is specific and useful, whether plan is proportional.

### Stage 2 — RESEARCH
Did the agent search effectively and adapt to evidence?
Focus on: query specificity, evidence-based pivoting, triangulation, appropriate stopping.

### Stage 3 — REVIEW
Did the agent honestly audit readiness?
Focus on: honest gap assessment, concrete drafting directives, appropriate scaling.

### Stage 4 — ANSWER
Is the final answer high-quality for the user's question?
Focus on: factual correctness, completeness, citation quality, synthesis depth, format compliance.

## What Makes a Good Rubric
- Separates stronger and weaker trajectories for the SAME question
- Captures genuine competence, not style preference
- Is specific enough to be actionable
- Is not redundant with existing rubrics

## Anti-Hack Rules
Do NOT create rubrics that only check tag existence, reward formal obedience, overfit to one trajectory, or punish evidence-based improvement.

## Selection Rules — STRICT MINIMALISM
- **Generate the minimum number of rubrics needed to distinguish trajectory quality.**
- If one rubric per stage suffices, use one. Zero is acceptable for stages where trajectories perform similarly.
- Never create positive and negative mirror versions of the same idea.
- Negative rubrics: target only clear, serious failure modes.

## Weights
- **3** = critical
- **2** = important
- **1** = minor but meaningful
Negative rubrics should usually be 1. Use negative 2-3 only for serious flaws.

## Output Format
Return rubrics by stage (`stage_1`, `stage_2`, `stage_3`, `stage_4`), each with `positive_rubrics` and `negative_rubrics`. Each item: `title`, `description`, `weight`.

Target counts — these are MAXIMUMS, not targets:
- Stage 1: 1-2 total
- Stage 2: 1-2 total
- Stage 3: 1-2 total
- Stage 4: 2-4 total

Generating fewer is always preferred. Return 0 rubrics for a stage if all trajectories perform similarly on that stage.

## Inputs
1. **Question**
2. **Trajectories**
3. **Existing Rubrics** (optional)

Generate only the most critical, non-redundant, stage-specific rubrics. Quality over quantity."""

STAGE_AWARE_RUBRIC_GENERATION_USER_TEMPLATE = """Question: {question}

{trajectories_text}

{existing_rubrics_text}
"""


async def generate_stage_aware_adaptive_rubrics(
    question,
    response_list,
    existing_rubrics = None,
    model_name = None,
    prompt_version = "default",
):
  """Generate stage-aware adaptive rubrics for a question and trajectories."""
  if model_name is None:
    model_name = os.environ.get(
        "RUBRIC_GENERATION_MODEL", "gemini-3-flash-preview"
    )

  traj_parts = []
  for i, resp in enumerate(response_list):
    truncated = (
        resp[:100000] + "\n...[truncated]..." if len(resp) > 100000 else resp
    )
    traj_parts.append(f"Trajectory {i+1}:\n{truncated}")
  trajectories_text = "\n\n".join(traj_parts)

  existing_rubrics_text = ""
  if existing_rubrics:
    existing_rubrics_text = (
        f"Existing Rubrics (do NOT duplicate):\n{existing_rubrics}"
    )

  user_prompt = STAGE_AWARE_RUBRIC_GENERATION_USER_TEMPLATE.format(
      question=question,
      trajectories_text=trajectories_text,
      existing_rubrics_text=existing_rubrics_text,
  )

  client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

  # Retry with backoff for transient API errors
  gen_delay = 3.0
  for gen_attempt in range(3):
    try:
      resp = await client.aio.models.generate_content(
          model=model_name,
          contents=user_prompt,
          config=types.GenerateContentConfig(
              system_instruction=STAGE_AWARE_RUBRIC_GEN_PROMPTS.get(
                  prompt_version, STAGE_AWARE_RUBRIC_GENERATION_PROMPT_v2
              ),
              response_mime_type="application/json",
              response_json_schema=StageAwareAdaptiveRubrics.model_json_schema(),
              temperature=0.2,
          ),
      )
      obj = StageAwareAdaptiveRubrics.model_validate_json(resp.text)
      result = obj.model_dump()

      # Log summary
      stage_names = {1: "Plan", 2: "Research", 3: "Review", 4: "Answer"}
      for k in range(1, 5):
        stage_data = result.get(f"stage_{k}", {})
        pos_count = len(stage_data.get("positive_rubrics", []))
        neg_count = len(stage_data.get("negative_rubrics", []))
        titles = [
            r["title"]
            for r in stage_data.get("positive_rubrics", []) + stage_data.get(
                "negative_rubrics", []
            )
        ]
        print(f"  S{k}({stage_names[k]}): {pos_count}+ {neg_count}- → {titles}")

      return result

    except Exception as e:  # pylint: disable=broad-exception-caught
      error_msg = str(e).lower()
      if any(k in error_msg for k in ["429", "quota", "500", "503", "timeout"]):
        jitter = random.uniform(1.0, 5.0)
        LOGGER.warning(
            "[RubricGen] Retry %d/3 for question '%s...': %s",
            gen_attempt + 1,
            question[:60],
            e,
        )
        await asyncio.sleep(gen_delay + jitter)
        gen_delay *= 2
      else:
        LOGGER.error(
            "[RubricGen] Non-retryable error for question '%s...': %s",
            question[:60],
            e,
        )
        break

  # All retries exhausted → return None (discard this question's M trajectories)
  LOGGER.error(
      "[RubricGen] All retries failed for question '%s...'. Returning None.",
      question[:80],
  )
  return None


# ── Batch generation ─────────────────────────────────────────────────


async def _generate_stage_aware_adaptive_rubrics_batch(
    responses,
    ground_truths,
    num_samples_per_prompt_rollout,
    rubric_buffer = None,
    use_full_responses = True,
    answer_length_limit_in_words = None,
    rubric_generation_prompt_version = "default",
):
  """Generate stage-aware adaptive rubrics for a batch of prompts."""
  parsed_gts = []
  for gt in ground_truths:
    if isinstance(gt, list):
      parsed_gts.append(json.loads(gt[0]) if isinstance(gt[0], str) else gt[0])
    elif isinstance(gt, str):
      parsed_gts.append(json.loads(gt))
    else:
      parsed_gts.append(gt)

  num_prompts = len(responses) // num_samples_per_prompt_rollout
  query_key = "query" if "query" in parsed_gts[0] else "Question"

  tasks = []
  num_subsampled_answers_list = []

  for i in range(num_prompts):
    start_idx = i * num_samples_per_prompt_rollout
    end_idx = start_idx + num_samples_per_prompt_rollout
    question = parsed_gts[start_idx][query_key]

    if rubric_buffer is not None and question in rubric_buffer:
      existing = rubric_buffer[question].get("active_rubrics", [])
      persistent = rubric_buffer[question].get("persistent_rubrics", [])
      existing_rubrics_str = _format_existing_rubrics_by_stage(
          persistent + existing
      )
    else:
      existing_rubrics_str = _format_existing_rubrics_by_stage(
          parsed_gts[start_idx].get("rubrics", [])
      )

    response_list = responses[start_idx:end_idx]

    # ---------------- Core modification start ----------------
    if use_full_responses:
      num_subsampled_answers_list.append(len(response_list))
      task = generate_stage_aware_adaptive_rubrics(
          question,
          response_list,
          existing_rubrics_str,
          prompt_version=rubric_generation_prompt_version,
      )
    else:
      # 1. Extract answer (used only for word count to prevent token overflow)
      # Note: ensure extract_answer_context_citations is imported at file top
      answer_list = [
          extract_answer_context_citations(resp)[1] for resp in response_list
      ]

      # 2. Pair full responses with extracted answers
      # Stage-aware scoring needs the model output stage tags
      valid_pairs = [
          (resp, ans)
          for resp, ans in zip(response_list, answer_list)
          if ans is not None
      ]

      # 3. Subsample based on word count limit
      if answer_length_limit_in_words is not None:
        random.shuffle(valid_pairs)
        selected_pairs = []
        total_words = 0
        for resp, ans in valid_pairs:
          word_count = len(ans.split())
          # Ensure at least 2 rollouts for comparison, otherwise accumulate
          # until limit
          if (
              len(selected_pairs) < 2
              or total_words + word_count <= answer_length_limit_in_words
          ):
            selected_pairs.append((resp, ans))
            total_words += word_count
          else:
            break
        valid_pairs = selected_pairs if selected_pairs else valid_pairs[:2]

      # 4. Unpack to get retained full responses with stage tags
      final_input_list = (
          [resp for resp, ans in valid_pairs] or response_list[:2]
      )

      # (Optional): Uncomment the line below to pass only final answers instead
      # of full trajectories:
      # final_input_list = [ans for resp, ans in valid_pairs]

      num_subsampled_answers_list.append(len(final_input_list))
      task = generate_stage_aware_adaptive_rubrics(
          question,
          final_input_list,
          existing_rubrics_str,
          prompt_version=rubric_generation_prompt_version,
      )
    # ---------------- Core modification end ----------------
    tasks.append(task)

  results = await asyncio.gather(*tasks, return_exceptions=True)

  # Convert exceptions to None (same as total generation failure → discard)
  clean_results = []
  for i, r in enumerate(results):
    if isinstance(r, Exception):
      LOGGER.error(
          "[RubricGenBatch] Question %d raised %s: %s",
          i,
          type(r).__name__,
          r,
      )
      clean_results.append(None)
    else:
      clean_results.append(r)

  return clean_results, num_subsampled_answers_list


# ── Flatten stage-keyed rubrics into ground_truth format ─────────────


def _format_existing_rubrics_by_stage(
    flat_rubrics,
):
  """Convert flat rubric list into stage-grouped format matching output schema.

  Formats existing rubrics so the judge sees them in the same structure it must
  produce.

  Input:  [{"title": ..., "description": ..., "weight": 2.0, "stage": 1}, ...]
  Output: JSON string like {"stage_1": {"positive_rubrics": [...],
  "negative_rubrics": [...]}, ...}

  Stages with no rubrics are omitted.

  Args:
    flat_rubrics: List of flat rubric dicts.

  Returns:
    JSON string with rubrics grouped by stage.
  """
  grouped = {}
  for r in flat_rubrics:
    stage = r.get("stage", 4)
    key = f"stage_{stage}"
    if key not in grouped:
      grouped[key] = {"positive_rubrics": [], "negative_rubrics": []}

    weight = r.get("weight", 1.0)
    item = {
        "title": r.get("title", ""),
        "description": r.get("description", ""),
        "weight": max(1, int(round(abs(weight)))) if weight != 0 else 1,
    }
    if weight < 0:
      grouped[key]["negative_rubrics"].append(item)
    else:
      grouped[key]["positive_rubrics"].append(item)

  return json.dumps(grouped, indent=2)


def _flatten_stage_rubrics(
    adaptive_result,
):
  """Convert stage-keyed schema to flat list with stage field.

  Input:  {"stage_1": {"positive_rubrics": [...], "negative_rubrics": [...]},
  ...}
  Output: [{"title": ..., "description": ..., "weight": 2.0, "stage": 1}, ...]

  Args:
    adaptive_result: Stage-keyed adaptive rubrics dictionary.

  Returns:
    Flat list of rubric dicts with a 'stage' field.
  """
  flat = []
  for k in range(1, 5):
    stage_data = adaptive_result.get(f"stage_{k}", {})
    for r in stage_data.get("positive_rubrics", []):
      w = min(3, max(1, r.get("weight", 2)))
      flat.append({
          "title": r["title"],
          "description": r["description"],
          "weight": float(w),
          "stage": k,
      })
    for r in stage_data.get("negative_rubrics", []):
      w = min(3, max(1, r.get("weight", 2)))
      flat.append({
          "title": r["title"],
          "description": r["description"],
          "weight": float(-w),  # negative sign from category
          "stage": k,
      })
  return flat


def update_ground_truths_with_stage_aware_adaptive_rubrics(
    ground_truths,
    all_adaptive_rubrics,
    num_samples_per_prompt_rollout,
    rubric_buffer = None,
):
  """Update ground truths and rubric buffer with newly generated rubrics."""
  valid_rate = 0.0
  num_gt_list, num_adaptive_list, num_buffer_list = [], [], []

  expanded = []
  for rubric_set in all_adaptive_rubrics:
    for _ in range(num_samples_per_prompt_rollout):
      expanded.append(rubric_set)

  processed_queries = set()
  skipped_count = 0

  for i, (gt, adaptive) in enumerate(zip(ground_truths, expanded)):
    is_list = isinstance(gt, list)
    gt_str = gt[0] if is_list else gt
    gt_obj = json.loads(gt_str)
    query = gt_obj.get("query", gt_obj.get("Question", ""))

    if adaptive is None:
      skipped_count += 1
      if rubric_buffer is not None and query in rubric_buffer:
        persistent = rubric_buffer[query].get("persistent_rubrics", [])
        active = rubric_buffer[query].get("active_rubrics", [])
        for r in persistent:
          if "stage" not in r:
            r["stage"] = 4
        gt_obj["rubrics"] = persistent + active
        gt_obj["rubrics_types"] = ["persistent"] * len(persistent) + [
            "adaptive"
        ] * len(active)
        updated_str = json.dumps(gt_obj)
        ground_truths[i] = [updated_str] if is_list else updated_str
      continue

    # Flatten the stage-keyed schema
    new_adaptive_rubrics = _flatten_stage_rubrics(adaptive)

    # Ensure existing rubrics have stage field
    for r in gt_obj.get("rubrics", []):
      if "stage" not in r:
        r["stage"] = 4

    num_gt_list.append(len(gt_obj.get("rubrics", [])))
    num_adaptive_list.append(len(new_adaptive_rubrics))

    # Update rubric buffer (once per prompt)
    if (
        rubric_buffer is not None
        and query in rubric_buffer
        and query not in processed_queries
    ):
      rubric_buffer[query]["active_rubrics"].extend(new_adaptive_rubrics)
      num_buffer_list.append(len(rubric_buffer[query]["active_rubrics"]))
      processed_queries.add(query)

    # Compose final rubrics list
    if rubric_buffer is not None and query in rubric_buffer:
      persistent = rubric_buffer[query].get("persistent_rubrics", [])
      active = rubric_buffer[query].get("active_rubrics", [])
      for r in persistent:
        if "stage" not in r:
          r["stage"] = 4
      gt_obj["rubrics"] = persistent + active
      gt_obj["rubrics_types"] = ["persistent"] * len(persistent) + [
          "adaptive"
      ] * len(active)
    else:
      original = gt_obj.get("rubrics", [])
      gt_obj["rubrics"] = original + new_adaptive_rubrics
      gt_obj["rubrics_types"] = ["persistent"] * len(original) + [
          "adaptive"
      ] * len(new_adaptive_rubrics)

    updated_str = json.dumps(gt_obj)
    ground_truths[i] = [updated_str] if is_list else updated_str
    valid_rate += 1.0

  if skipped_count > 0:
    print(
        f"⚠️  Skipped {skipped_count}/{len(expanded)} adaptive rubrics (None —"
        " API failed)"
    )

  total = len(ground_truths) if ground_truths else 1
  return (
      ground_truths,
      valid_rate / total,
      float(np.mean(num_gt_list)) if num_gt_list else 0.0,
      float(np.mean(num_adaptive_list)) if num_adaptive_list else 0.0,
      float(np.mean(num_buffer_list)) if num_buffer_list else 0.0,
      rubric_buffer,
      skipped_count,
  )


# =====================================================================
# 3. Stage-Aware Rubric SCORING
# =====================================================================

STAGE_AWARE_SCORING_SYSTEM_PROMPT = """You are an expert evaluator. You will receive an AI research agent's full trajectory and a set of rubrics grouped by stage. Score each rubric strictly based on the trajectory evidence.

## Trajectory Structure
Each trajectory follows this expected 4-stage structure:

**Stage 1 (Plan)** — start through end of `</structured_plan>`:
- `<think>`: Brainstorming
- `<structured_plan>` with `<deep_analysis>`, `<rubric>` (agent's self-generated grading criteria), `<research_plan>`

**Stage 2 (Research)** — after `</structured_plan>` through all tool-call loops:
- `<call_tool>` → `<tool_output>` → `<think>` → `<state_evaluation>` → optional plan update → next `<call_tool>`

**Stage 3 (Review)** — last `<think>` before `<review>` through `</review>`:
- `<rubric_review>`: Maps evidence to task needs and acknowledges gaps
- `<writing_plan>`: Answer architecture with citation mapping

**Stage 4 (Answer)** — `<answer>` through end of text:
- The final answer itself

## Scoring Rules
**[POSITIVE (DESIRED)] rubrics:**
  2 = FULLY exhibits this quality — clear, strong evidence in the trajectory
  1 = PARTIALLY exhibits it — present but incomplete or weak
  0 = Completely ABSENT or fails

**[NEGATIVE (FLAW)] rubrics:**
  2 = Fully EXHIBITS the flaw — clear, unambiguous evidence of this problem
  1 = PARTIALLY exhibits the flaw — mild or occasional instances
  0 = Successfully AVOIDS the flaw — no evidence of this problem

## Instructions
- Score each rubric independently based on what you observe in the trajectory.
- Use the rubric description as the sole scoring criterion — do not invent additional standards.
- Provide a brief justification grounded in specific trajectory evidence (quote or reference concrete content when possible).
- Return the exact rubric index with each score.
"""

STAGE_AWARE_SCORING_USER_TEMPLATE = """<question>
{question}
</question>

<full_trajectory>
{trajectory}
</full_trajectory>

<rubrics_to_evaluate>
{rubrics_text}
</rubrics_to_evaluate>
"""

TRAJECTORY_CHAR_LIMIT = 2000000


async def _score_all_stages_with_rubrics_async(
    response,
    ground_truth,
    use_general_rubric=False,
    use_likert_rubric=False,
    verification_reward=10.0,
):
  """Score all stages of a response using stage-grouped rubrics.

  Args:
    response: The trajectory string to score.
    ground_truth: Dictionary containing question and rubrics.
    use_general_rubric: Whether to use general rubric scoring.
    use_likert_rubric: Whether to use Likert rubric scoring.
    verification_reward: Verification reward weight (unused).

  Returns:
    Tuple of (stage_scores, stage_justifications, scoring_failed,
    rubric_details).
  """
  del verification_reward
  from open_instruct.search_rewards.utils.rubric_utils import _call_gemini_score_async  # pylint: disable=g-import-not-at-top,g-importing-member

  rubrics = ground_truth.get("rubrics", [])
  rubrics_types = ground_truth.get(
      "rubrics_types", ["persistent"] * len(rubrics)
  )
  question = ground_truth.get("query", ground_truth.get("Question", ""))

  if not rubrics:
    return [0.0] * 4, ["no rubrics"] * 4, True, []

  if use_likert_rubric or use_general_rubric:
    from open_instruct.search_rewards.utils.rubric_utils import _score_weighted_rubric  # pylint: disable=g-import-not-at-top,g-importing-member

    scores, weights, justs, evaluated = await _score_weighted_rubric(
        response,
        ground_truth,
        use_general_rubric=use_general_rubric,
        use_likert_rubric=use_likert_rubric,
    )
    total_w = sum(weights) if weights else 1.0
    gs = (
        sum(s * w for s, w in zip(scores, weights)) / max(total_w, 1e-8)
        if scores
        else 0.0
    )
    return (
        [gs] * 4,
        ["; ".join(justs[:2])] * 4,
        not all(evaluated) if evaluated else True,
        [],
    )

  # ── Build rubric text grouped by stage ────────────────────────
  stage_names = {1: "Plan", 2: "Research", 3: "Review", 4: "Answer"}
  stage_boundaries = {
      1: "start → end of </structured_plan>",
      2: "after </structured_plan> → last <think> before <review>",
      3: "last <think> before <review> → end of </review>",
      4: "<answer> → end of text",
  }

  # Group rubrics by stage with local indices, and build a mapping back to
  # global index: stage_local_to_global[(stage, local_idx)] = global_idx
  stage_to_rubric_lines = {}
  stage_local_to_global = {}
  stage_local_counters = {}
  for idx, r in enumerate(rubrics):
    stage = r.get("stage", 4)
    weight = r.get("weight", 1.0)
    abs_w = abs(weight)
    direction = "NEGATIVE (FLAW)" if weight < 0 else "POSITIVE (DESIRED)"
    importance = {1: "minor", 2: "important", 3: "critical"}.get(
        int(min(3, max(1, abs_w))), "important"
    )
    desc = r.get("description", r.get("criterion", r.get("desc", "")))
    local_idx = stage_local_counters.get(stage, 0)
    stage_local_counters[stage] = local_idx + 1
    stage_local_to_global[(stage, local_idx)] = idx
    line = (
        f"[{local_idx}] [{direction}] [Importance: {importance}"
        f" (w={abs_w:.0f})] {desc}"
    )
    stage_to_rubric_lines.setdefault(stage, []).append(line)

  # Build final text with stage headers
  rubrics_text_parts = []
  for stage in sorted(stage_to_rubric_lines.keys()):
    label = stage_names.get(stage, f"Stage {stage}")
    boundary = stage_boundaries.get(stage, "")
    rubrics_text_parts.append(
        f"## Stage {stage} — {label} (evaluate: {boundary})"
    )
    rubrics_text_parts.extend(stage_to_rubric_lines[stage])
    rubrics_text_parts.append("")  # blank line between stages

  trajectory = response[:TRAJECTORY_CHAR_LIMIT]
  if len(response) > TRAJECTORY_CHAR_LIMIT:
    trajectory += "\n...[truncated]..."

  user_prompt = STAGE_AWARE_SCORING_USER_TEMPLATE.format(
      question=question,
      trajectory=trajectory,
      rubrics_text="\n".join(rubrics_text_parts),
  )

  model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gemini-3-flash-preview")

  # Retry the scoring API call with backoff
  resp_text = None
  scoring_delay = 3.0
  for scoring_attempt in range(3):
    try:
      resp_text = await _call_gemini_score_async(
          STAGE_AWARE_SCORING_SYSTEM_PROMPT,
          user_prompt,
          model_name,
          StageGroupedScoringResult.model_json_schema(),
      )
      break
    except Exception as e:  # pylint: disable=broad-exception-caught
      error_msg = str(e).lower()
      if any(k in error_msg for k in ["429", "quota", "500", "503", "timeout"]):
        jitter = random.uniform(1.0, 5.0)
        LOGGER.warning(
            "[Scoring] Retry %d/3, waiting %.1fs: %s",
            scoring_attempt + 1,
            scoring_delay + jitter,
            e,
        )
        await asyncio.sleep(scoring_delay + jitter)
        scoring_delay *= 2
      else:
        LOGGER.error("[Scoring] Non-retryable error: %s", e)
        break

  # ── Parse stage-grouped response ────────────────────────────
  per_rubric_scores = [0.0] * len(rubrics)
  per_rubric_evaluated = [False] * len(rubrics)
  per_rubric_justifications = ["not evaluated"] * len(rubrics)

  if resp_text:
    try:
      obj = StageGroupedScoringResult.model_validate_json(resp_text)
      for stage in range(1, 5):
        block = getattr(obj, f"stage_{stage}", None)
        if block is None:
          continue
        for ev in block.evaluations:
          global_idx = stage_local_to_global.get((stage, ev.rubric_index))
          if global_idx is not None and 0 <= global_idx < len(rubrics):
            per_rubric_scores[global_idx] = max(0.0, min(1.0, ev.score / 2.0))
            per_rubric_evaluated[global_idx] = True
            per_rubric_justifications[global_idx] = ev.justification
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Full parsing failed — try to salvage individual stages from raw JSON
      LOGGER.warning(
          "Error parsing StageGroupedScoringResult: %s. Attempting partial"
          " recovery...",
          e,
      )
      try:
        raw = json.loads(resp_text)
        for stage in range(1, 5):
          stage_data = raw.get(f"stage_{stage}")
          if not isinstance(stage_data, dict):
            continue
          evals = stage_data.get("evaluations", [])
          for ev_dict in evals:
            try:
              ev = StageAwareRubricEvaluation.model_validate(ev_dict)
              global_idx = stage_local_to_global.get((stage, ev.rubric_index))
              if global_idx is not None and 0 <= global_idx < len(rubrics):
                per_rubric_scores[global_idx] = max(
                    0.0, min(1.0, ev.score / 2.0)
                )
                per_rubric_evaluated[global_idx] = True
                per_rubric_justifications[global_idx] = ev.justification
            except Exception:  # pylint: disable=broad-exception-caught
              continue  # skip individual malformed evaluation
      except Exception as e2:  # pylint: disable=broad-exception-caught
        LOGGER.warning("Partial recovery also failed: %s", e2)

  # ── Build rubric_details ─────────────────────────────────────
  rubric_details = []
  for idx, r in enumerate(rubrics):
    stage = max(1, min(4, r.get("stage", 4)))
    raw_weight = r.get("weight", 1.0)
    is_negative = raw_weight < 0
    score = per_rubric_scores[idx]
    inverted_score = (1.0 - score) if is_negative else score

    rubric_details.append({
        "rubric_index": idx,
        "stage": stage,
        "title": r.get("title", f"rubric_{idx}"),
        "description": r.get("description", r.get("criterion", "")),
        "weight": raw_weight,
        "abs_weight": abs(raw_weight),
        "direction": "negative" if is_negative else "positive",
        "type": rubrics_types[idx] if idx < len(rubrics_types) else "unknown",
        "raw_score": round(score, 4),
        "effective_score": round(inverted_score, 4),  # after negation inversion
        "justification": per_rubric_justifications[idx],
        "evaluated": per_rubric_evaluated[idx],
    })

  # ── Aggregate per stage ──────────────────────────────────────
  # Positive rubrics: contribute (effective_score * abs_weight) to numerator
  #   and abs_weight to denominator.
  # Negative rubrics: subtract (raw_score * abs_weight) from numerator
  #   but do NOT contribute to denominator.
  # This keeps max possible score = 1.0 (when all positives perfect,
  # all flaws absent) and allows flaws to push score below 0 (clamped).
  stage_pos_num = {k: 0.0 for k in range(1, 5)}  # positive contribution
  stage_neg_sub = {k: 0.0 for k in range(1, 5)}  # negative subtraction
  stage_pos_denom = {k: 0.0 for k in range(1, 5)}  # positive weight sum only
  stage_just_accum = {k: [] for k in range(1, 5)}
  stage_has_rubrics = {k: False for k in range(1, 5)}
  scoring_failed = False

  for rd in rubric_details:
    stage = rd["stage"]
    stage_has_rubrics[stage] = True
    stage_just_accum[stage].append(rd["justification"])
    if not rd["evaluated"]:
      scoring_failed = True
    if rd["direction"] == "positive":
      stage_pos_num[stage] += rd["effective_score"] * rd["abs_weight"]
      stage_pos_denom[stage] += rd["abs_weight"]
    else:
      # Negative: raw_score=1 means flaw present → subtract
      stage_neg_sub[stage] += rd["raw_score"] * rd["abs_weight"]

  stage_scores = [0.0] * 4
  stage_justifications = [""] * 4
  scored_stages = []

  for k in range(1, 5):
    j_list = stage_just_accum[k]
    if stage_has_rubrics[k]:
      denom = stage_pos_denom[k]
      if denom > 1e-8:
        raw = (stage_pos_num[k] - stage_neg_sub[k]) / denom
        stage_scores[k - 1] = max(0.0, min(1.0, raw))
      else:
        # Only negative rubrics for this stage — score based on flaw absence
        # If no flaws present, score = 1.0; if all flaws present, score = 0.0
        neg_total = sum(
            rd["abs_weight"]
            for rd in rubric_details
            if rd["stage"] == k and rd["direction"] == "negative"
        )
        if neg_total > 1e-8:
          stage_scores[k - 1] = max(0.0, 1.0 - stage_neg_sub[k] / neg_total)
        else:
          stage_scores[k - 1] = 0.0
      stage_justifications[k - 1] = "; ".join(j_list[:3])
      scored_stages.append(k)

  if scored_stages and len(scored_stages) < 4:
    fallback = np.mean([stage_scores[k - 1] for k in scored_stages])
    for k in range(1, 5):
      if k not in scored_stages:
        stage_scores[k - 1] = fallback
        stage_justifications[k - 1] = f"no rubrics — fallback={fallback:.3f}"

  return stage_scores, stage_justifications, scoring_failed, rubric_details


# =====================================================================
# 4. Legacy fallback (unchanged from v3)
# =====================================================================

PROCESS_STAGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the PROCESS quality of an AI research agent's trajectory.

- Stage 1 (PLAN): <think> → <structured_plan> with <deep_analysis>, <rubric>, <research_plan> → first <call_tool>
- Stage 2 (RESEARCH): Loops of <think> → <state_evaluation> → <call_tool>
- Stage 3 (REVIEW): <review> with <rubric_review> and <writing_plan>

Score 0-2 per stage. 0=poor, 1=adequate, 2=excellent.

Stage 1: True complexity analysis? Specific rubric? Conditional plan?
Stage 2: Specific queries? Accurate state evaluation? Good pivots?
Stage 3: Faithful rubric review? Gap honesty? Citation-mapped writing plan?
"""

PROCESS_STAGE_USER_TEMPLATE = """<question>{question}</question>
<stage_1_plan>{stage1_text}</stage_1_plan>
<stage_2_research>{stage2_text}</stage_2_research>
<stage_3_review>{stage3_text}</stage_3_review>
Evaluate each stage."""

STAGE_TEXT_CHAR_LIMIT = 6000


async def _score_process_stages_async(question, stage_texts):
  """Score process stages (Plan, Research, Review) using fixed process judge."""
  from open_instruct.search_rewards.utils.rubric_utils import _call_gemini_score_async  # pylint: disable=g-import-not-at-top,g-importing-member

  trunc = (
      lambda t: (t[:STAGE_TEXT_CHAR_LIMIT] + "\n...[truncated]...")
      if len(t) > STAGE_TEXT_CHAR_LIMIT
      else t
  )
  user_prompt = PROCESS_STAGE_USER_TEMPLATE.format(
      question=question,
      stage1_text=trunc(stage_texts.get(1, "")) or "(not present)",
      stage2_text=trunc(stage_texts.get(2, "")) or "(not present)",
      stage3_text=trunc(stage_texts.get(3, "")) or "(not present)",
  )
  model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gemini-3-flash-preview")
  resp_text = await _call_gemini_score_async(
      PROCESS_STAGE_SYSTEM_PROMPT,
      user_prompt,
      model_name,
      ProcessStagesEvaluation.model_json_schema(),
  )
  results = {
      1: (0.0, "API failed"),
      2: (0.0, "API failed"),
      3: (0.0, "API failed"),
  }
  if resp_text:
    try:
      obj = ProcessStagesEvaluation.model_validate_json(resp_text)
      for ev in obj.evaluations:
        if ev.stage in (1, 2, 3):
          results[ev.stage] = (
              max(0.0, min(1.0, ev.score / 2.0)),
              ev.justification,
          )
    except Exception as e:  # pylint: disable=broad-exception-caught
      LOGGER.warning("Error parsing ProcessStagesEvaluation: %s", e)
  return results


async def _score_all_stages_async(
    response,
    question,
    stage_texts,
    ground_truth,
    use_general_rubric=False,
    use_likert_rubric=False,
    verification_reward=10.0,
):
  """Legacy: fixed process judge for S1-3, rubric for S4."""
  del verification_reward
  from open_instruct.search_rewards.utils.rubric_utils import _score_weighted_rubric  # pylint: disable=g-import-not-at-top,g-importing-member

  process_task = _score_process_stages_async(question, stage_texts)
  answer_task = _score_weighted_rubric(
      response,
      ground_truth,
      use_general_rubric=use_general_rubric,
      use_likert_rubric=use_likert_rubric,
  )
  process_results, answer_results = await asyncio.gather(
      process_task, answer_task
  )

  stage_scores, stage_justifications = [0.0] * 4, [""] * 4
  scoring_failed = False
  for k in (1, 2, 3):
    s, j = process_results.get(k, (0.0, "missing"))
    stage_scores[k - 1], stage_justifications[k - 1] = s, j
    if "API failed" in j:
      scoring_failed = True

  ans_s, ans_w, ans_j, ans_e = answer_results
  if ans_s and ans_w:
    tw = sum(ans_w)
    stage_scores[3] = (
        sum(s * w for s, w in zip(ans_s, ans_w)) / max(tw, 1e-8)
        if tw > 0
        else 0.0
    )
    stage_justifications[3] = "; ".join(ans_j[:3])
    if not all(ans_e):
      scoring_failed = True
  else:
    scoring_failed = True
  return stage_scores, stage_justifications, scoring_failed


# =====================================================================
# 5. Helpers
# =====================================================================


def extract_agent_rubric(trajectory):
  """Extract the agent's self-generated <rubric> block from a trajectory."""
  match = re.search(r"<rubric>(.*?)</rubric>", trajectory, re.DOTALL)
  if match:
    t = match.group(1).strip()
    return t[:3000] + "\n...[truncated]..." if len(t) > 3000 else t
  return ""


def rubrics_have_stage_field(ground_truth):
  """Return True if any rubric in ground_truth has a 'stage' field."""
  return any("stage" in r for r in ground_truth.get("rubrics", []))
