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

"""Rubric Bank Item Generation — policy generates reflections, judge scores them.

Pipeline per training step:
  1. Policy generates reflection candidates (via vLLM, no tool calls)
     - Input: question + 1-2 ungraded trajectories
     - Output: rubrics + takeaways in structured format
  2. Judge scores reflections (via Gemini)
     - Has access to graded trajectories (can verify accuracy)
     - Produces per-reflection scores (for GRPO training)
     - Synthesizes best reflection into a bank item
"""

import asyncio
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from open_instruct.rubric_bank import RubricBankItem
from pydantic import BaseModel
from pydantic import Field

LOGGER = logging.getLogger(__name__)


# =====================================================================
# 1. Reflection Prompt for the POLICY model
# =====================================================================

REFLECTION_SYSTEM_PROMPT_ORIGINAL = """You are a research agent reviewing your own work. You will see a question and a research trajectory you produced (including your final <answer>).

## Process

**Step 1: Examine your <answer> closely.**
Before generating any reflections, carefully re-read the final answer you produced. Ask yourself:
- Is it actually correct? Are there factual errors, unsupported claims, or missing nuance?
- Does it fully address the question, or does it dodge parts of it?
- Is the evidence strong, or are there gaps where you asserted without support?
- Are there claims that sound confident but might be wrong?

**Step 2: Trace problems back to the research process.**
For any weaknesses you identified in your answer, trace them back:
- Did you fail to search for something critical?
- Did you trust a bad source or stop searching too early?
- Did your plan miss an important angle?
- Did you have the right information but fail to synthesize it well?

**Step 3: Generate reflections.**

<reflection_rubrics>
Key criteria for answering this question well, framed as guidance for the next attempt:
- **Critical Requirements**: What a good answer must include or get right.
- **Analytical Approach**: What reasoning strategy or connections matter most.
- **Pitfalls**: Specific mistakes or traps to avoid, grounded in what actually happened in the trajectory.
</reflection_rubrics>

<reflection_takeaways>
2-4 concise insights that complement the rubrics above — things that don't fit neatly into a checklist. For example: research strategies that worked, non-obvious domain context, why a particular assumption is tempting but wrong, or how the planning phase should be structured for this type of question.
</reflection_takeaways>

## Guidelines
- Start from the answer, not the plan — the answer reveals what actually matters
- Frame rubrics as guidance for the next attempt, not a report on this one
- Be specific enough that the advice is actionable ("be more thorough" is useless)
- Takeaways should add value beyond the rubrics, not repeat them
- These reflections should help on this exact question AND similar ones
"""

REFLECTION_SYSTEM_PROMPT = """You are a strict postmortem editor for long-form deep research QA.

You will receive:
- a question
- a research trajectory ending in a final <answer>...</answer>

You may think before answering, but your visible response must contain only the required output blocks below.

Your job is not to answer the question again. Your job is to extract the most useful guidance for:
1. a stronger next attempt on this question, and
2. similar long-form research questions.

First, read the final <answer>.

Judge it on:
- correctness and calibration
- instruction-following and coverage
- research quality and verification
- synthesis and insight
- communication and structure

Then identify the 2-4 highest-leverage strengths or weaknesses, trace them back to the visible trajectory, and turn them into reusable guidance.

Rules:
- Stay grounded in the provided question and trajectory.
- Do not invent flaws just to be critical.
- Do not speculate about hidden reasoning not shown in the trajectory.
- If something seems unsupported, frame it as a verification/support gap.
- Do not rewrite the answer.
- Do not give generic advice like "be more careful" or "add more detail."
- Write specific guidance that is grounded in this case but transferable to similar questions.

Your final output must be exactly the following two blocks and nothing else:

<reflection_rubrics>
Critical Requirements:
- 2-4 bullets on what a strong future answer must explicitly include, verify, or address

Analytical Approach:
- 2-4 bullets on the research, reasoning, comparison, and synthesis strategy that matters most

Communication & Structure:
- 1-4 bullets on how the answer should be framed, organized, and written

Pitfalls:
- 0-4 bullets on concrete failure modes to avoid, grounded in this trajectory
</reflection_rubrics>

<reflection_takeaways>
- 1-4 concise portable lessons for this and similar deep-research questions
</reflection_takeaways>
"""

REFLECTION_USER_TEMPLATE = """## Question
{question}

## Research Trajectory
{trajectory}

Examine the final <answer>, trace strengths and weaknesses through the full trajectory, and produce the <reflection_rubrics> and <reflection_takeaways> blocks."""

# Simpler reflection prompt for models not trained on rubric bank format
# (e.g., DR-Tulu)
REFLECTION_SYSTEM_PROMPT_SIMPLE = """You are reviewing a research trajectory and its final answer.

Your job is to extract lessons learned — what worked, what didn't, and what to do differently next time.

Read the question, the full trajectory, and the final <answer>. Then produce exactly two blocks:

<reflection_rubrics>
- 2-5 bullet points describing what a strong answer to this question (and similar ones) must include or avoid
</reflection_rubrics>

<reflection_takeaways>
- 2-4 concise, portable lessons that would help on a future attempt at this question or similar research questions
</reflection_takeaways>

Rules:
- Be specific and grounded in what you see in the trajectory
- No generic advice like "be more thorough"
- Focus on the highest-leverage improvements
"""


# =====================================================================
# 2. Judge Prompt for scoring reflections + synthesizing bank items
# =====================================================================

JUDGE_REFLECTION_SYSTEM_PROMPT_LEGACY = """You are an expert judge with full visibility into a research agent's performance. You have access to:
- The original question
- The agent's trajectory and final answer
- The **rubrics** used to evaluate the trajectory (with per-rubric scores and justifications)
- The agent's own **reflection candidates** (self-diagnosis of what went right/wrong)

Your job is to use your privileged position — you know the ground truth scores AND the rubrics AND the justifications — to evaluate whether the agent's self-reflections are accurate, and to synthesize the best possible bank item for future use.

## Your Task

### 1. Score each reflection candidate (0.0 to 1.0) on:
- **Actionability** (0.25 weight): Would these rubrics/reflections actually help the agent plan and answer better next time? Are they specific enough to act on?
- **Accuracy** (0.5 weight): Do they correctly identify what went wrong or right? Compare the agent's self-diagnosis against the actual rubric scores and justifications — did the agent catch the real issues, or did it miss them / fabricate problems?
- **Transferability** (0.25 weight): Would they help with similar questions, not just this exact one?

### 2. Synthesize the final bank item:
You have MORE information than the agent (rubric scores, justifications). Use this to produce a bank item that is BETTER than any individual candidate:
- **rubrics**: Key criteria for answering this question well — what matters most, informed by the actual evaluation. Include: Critical Requirements (what must be right), Analytical Approach (what reasoning works), and Pitfalls (what to avoid).
- **reflections**: 2-4 concise, high-impact lessons. Incorporate insights from the rubric justifications that the agent may have missed in its self-reflection.

## Quality Standards
- The synthesized item should capture what the agent got right AND what it missed
- Prefer fewer, sharper points over many generic ones
- The final item must be self-contained — readable without the original trajectory
- Rubrics should be specific enough to be testable but general enough to transfer
- If the agent's reflections missed a critical failure identified in the rubric justifications, ADD it to the synthesized item
"""

JUDGE_REFLECTION_USER_TEMPLATE_LEGACY = """## Question
{question}

## Reflection Candidates (agent's self-diagnosis)
{candidates_text}

## Graded Trajectory with Rubric Evaluation
{graded_trajectories_text}

Score each reflection candidate against the actual rubric evaluation above, then synthesize the best elements (from both the candidates AND the rubric justifications) into a final bank item."""


JUDGE_REFLECTION_SYSTEM_PROMPT = """You are an expert judge evaluating reflection candidates produced by a research agent.

You have access to:
- the original question
- the agent's graded trajectory, including per-rubric scores and evaluator justifications
- multiple reflection candidates written as self-diagnosis

Your task is to score each reflection candidate on how useful it would be for a future attempt on this question and similar questions.

## Scoring Dimensions (each 0.0 to 1.0)

Diagnostic Accuracy (weight 0.4)
- Does it correctly identify the main strengths and weaknesses?
- Is it aligned with the evaluator justifications and visible trajectory?
Penalize:
- missing a major failure clearly flagged by the evaluator
- inventing problems not supported by the trajectory
- misidentifying the root cause
- focusing on minor issues while missing the highest-leverage ones

Specificity (weight 0.3)
- Is the guidance concrete, actionable, and tailored to this kind of question?
Penalize:
- generic advice ("be more thorough", "verify more")
- checklist items that could apply to almost any task
- merely restating evaluator language without turning it into usable guidance

Scope & Balance (weight 0.3)
- Does it cover the right areas: requirements, approach, communication, and pitfalls?
- Are the takeaways complementary rather than redundant?
- Is it grounded in this case while still useful for similar questions?
Penalize:
- missing an important category
- repeating the same point across sections
- being too case-specific or too generic

## Rules
- Score based on the reflection's quality as future guidance, not on the trajectory's quality
- A reflection on a perfect trajectory can score low if it's generic
- A reflection on a failed trajectory can score high if it precisely identifies root causes
- Prefer reflections that identify the highest-leverage lessons.
"""

JUDGE_REFLECTION_USER_TEMPLATE = """## Question
{question}

## Reflection Candidates (agent's self-diagnosis)
{candidates_text}

## Graded Trajectory with Rubric Evaluation
{graded_trajectories_text}

Score each reflection candidate against the evaluator feedback and visible trajectory.
"""


# =====================================================================
# 3. Pydantic Schemas
# =====================================================================


class CandidateScore(BaseModel):
  """Score for a single reflection candidate."""

  candidate_index: int = Field(description="0-based index of the candidate")
  diagnostic_accuracy: float = Field(
      description=(
          "0.0-1.0: correctly identifies actual strengths/weaknesses vs rubric"
          " evaluation?"
      )
  )
  specificity: float = Field(
      description="0.0-1.0: concrete and question-specific, not generic?"
  )
  scope_and_balance: float = Field(
      description=(
          "0.0-1.0: covers the right areas with complementary, non-redundant"
          " guidance?"
      )
  )
  overall_score: float = Field(
      description=(
          "Weighted average: 0.4*diagnostic_accuracy + 0.3*specificity +"
          " 0.3*scope_and_balance"
      )
  )


class JudgeReflectionResult(BaseModel):
  """Judge output: scores for all candidates + optional synthesized bank item."""

  candidate_scores: List[CandidateScore] = Field(
      description="Scores for each reflection candidate"
  )
  synthesized_rubrics: Optional[str] = Field(
      default=None,
      description=(
          "Key criteria for answering this question well, synthesized from"
          " candidates and rubric justifications. Only produced when synthesis"
          " is requested."
      ),
  )
  synthesized_reflections: Optional[str] = Field(
      default=None,
      description=(
          "2-4 concise, high-impact lessons learned, synthesized from"
          " candidates and rubric justifications. Only produced when synthesis"
          " is requested."
      ),
  )


# =====================================================================
# 4. Reflection Prompt Preparation
# =====================================================================


def sample_reflection_indices(
    num_questions,
    num_samples_per_prompt_rollout,
    num_trajectories_per_question = 1,
):
  """Roll the dice: pick random trajectories per question for reflection.

  Called BEFORE generation or scoring so that the priority grading queue
  can be set up without waiting for any results. Multiple reflection
  candidates per trajectory are generated via vLLM n>1 sampling.

  Args:
      num_questions: Number of unique questions
      num_samples_per_prompt_rollout: M rollouts per question
      num_trajectories_per_question: How many trajectories to sample per
        question

  Returns:
      question_indices: One entry per sampled trajectory
      trajectory_indices: One global trajectory index per entry
  """
  question_indices = []
  trajectory_indices = []

  for q_idx in range(num_questions):
    start = q_idx * num_samples_per_prompt_rollout
    end = start + num_samples_per_prompt_rollout
    all_traj_indices = list(range(start, end))
    k = min(num_trajectories_per_question, len(all_traj_indices))
    sampled = random.sample(all_traj_indices, k)

    for traj_idx in sampled:
      question_indices.append(q_idx)
      trajectory_indices.append(traj_idx)

  LOGGER.info(
      "[ReflectionGen] Sampled %d trajectories for %d questions (%d per"
      " question)",
      len(trajectory_indices),
      num_questions,
      num_trajectories_per_question,
  )
  return question_indices, trajectory_indices


def prepare_reflection_prompts(
    questions,
    decoded_responses,
    num_samples_per_prompt_rollout,
    tokenizer,
    num_trajectories_per_question = 1,
    presampled_question_indices = None,
    presampled_trajectory_indices = None,
):
  """Prepare tokenized reflection prompts for vLLM generation.

  Creates 1 prompt per sampled trajectory. Multiple reflection candidates
  per trajectory are generated via vLLM n>1 sampling.

  If presampled indices are provided, uses those directly instead of
  re-sampling. This supports the async pipeline where indices are rolled
  before scoring begins.

  Args:
      questions: List of unique questions (one per prompt group)
      decoded_responses: All decoded responses (num_questions *
        num_samples_per_prompt_rollout)
      num_samples_per_prompt_rollout: M rollouts per question
      tokenizer: The tokenizer for chat template encoding
      num_trajectories_per_question: How many trajectories to sample per
        question
      presampled_question_indices: Pre-rolled question indices (from
        sample_reflection_indices)
      presampled_trajectory_indices: Pre-rolled trajectory indices (from
        sample_reflection_indices)

  Returns:
      reflection_token_ids: Tokenized reflection prompts (1 per trajectory)
      question_indices: Which question index each prompt belongs to
      trajectory_indices: Which trajectory index each prompt uses
  """
  # Use presampled indices if provided, otherwise sample now
  if (
      presampled_question_indices is not None
      and presampled_trajectory_indices is not None
  ):
    question_indices = presampled_question_indices
    trajectory_indices_list = presampled_trajectory_indices
  else:
    question_indices, trajectory_indices_list = sample_reflection_indices(
        num_questions=len(questions),
        num_samples_per_prompt_rollout=num_samples_per_prompt_rollout,
        num_trajectories_per_question=num_trajectories_per_question,
    )

  reflection_token_ids = []
  for q_idx, traj_idx in zip(question_indices, trajectory_indices_list):
    trajectory_text = decoded_responses[traj_idx]
    # Truncate very long trajectories, preserving the <answer> at the end
    # ~4 chars per token, 20480 token budget minus ~500 for system+user template
    max_traj_chars = 80000  # ~20K tokens
    if len(trajectory_text) > max_traj_chars:
      half = max_traj_chars // 2
      trajectory_text = (
          trajectory_text[:half]
          + "\n\n...[middle of research trajectory truncated]...\n\n"
          + trajectory_text[-half:]
      )

    user_message = REFLECTION_USER_TEMPLATE.format(
        question=questions[q_idx],
        trajectory=trajectory_text,
    )

    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    reflection_token_ids.append(token_ids)

  LOGGER.info(
      "[ReflectionGen] Prepared %d reflection prompts for %d questions",
      len(reflection_token_ids),
      len(questions),
  )
  return reflection_token_ids, question_indices, trajectory_indices_list


# =====================================================================
# 5. Parse Policy Reflection Output
# =====================================================================


def parse_reflection_output(text):
  """Parse the policy's reflection output into (rubrics, takeaways).

  Looks for <reflection_rubrics> and <reflection_takeaways> tags.
  Falls back to splitting on heuristics if tags are missing.

  Args:
    text: Raw reflection output text from the policy.

  Returns:
    Tuple of (rubrics, takeaways) strings.
  """
  rubrics = ""
  takeaways = ""

  # Try to extract from XML tags
  rubrics_match = re.search(
      r"<reflection_rubrics>(.*?)</reflection_rubrics>", text, re.DOTALL
  )
  takeaways_match = re.search(
      r"<reflection_takeaways>(.*?)</reflection_takeaways>", text, re.DOTALL
  )

  if rubrics_match:
    rubrics = rubrics_match.group(1).strip()
  if takeaways_match:
    takeaways = takeaways_match.group(1).strip()

  # Fallback: if no tags found, use the whole text as both
  if not rubrics and not takeaways:
    # Try splitting on common headers
    parts = re.split(r"(?i)(takeaway|reflection|lesson)", text, maxsplit=1)
    if len(parts) > 1:
      rubrics = parts[0].strip()
      takeaways = "".join(parts[1:]).strip()
    else:
      rubrics = text.strip()
      takeaways = text.strip()

  return rubrics, takeaways


# =====================================================================
# 6. vLLM Reflection Generation (simplified, no tools)
# =====================================================================


def generate_reflections_with_engines(
    vllm_engines,
    reflection_token_ids,
    sampling_params,
    ray_module,
    question_indices,
    trajectory_indices,
):
  """Generate reflections using vLLM engines.

  Simple single-turn generation, no tools.

  With n>1 sampling, each prompt produces multiple outputs. This function
  expands question_indices and trajectory_indices to match the output count.

  Args:
      vllm_engines: List of LLMRayActor instances
      reflection_token_ids: Tokenized reflection prompts (1 per question)
      sampling_params: vLLM SamplingParams (n>1 for multiple candidates)
      ray_module: The ray module (passed to avoid import issues)
      question_indices: Which question each prompt belongs to (1 per prompt)
      trajectory_indices: Which trajectory each prompt uses (1 per prompt)

  Returns:
      response_ids: Generated token IDs (num_prompts * n)
      finish_reasons: Finish reasons for each generation
      expanded_question_indices: Question index for each output
      expanded_trajectory_indices: Trajectory index for each output
  """
  if not reflection_token_ids:
    return [], [], [], []

  # Distribute across engines, tracking which prompts go where
  num_engines = len(vllm_engines)
  queries_per_engine = (
      len(reflection_token_ids) + num_engines - 1
  ) // num_engines
  split_queries = [
      reflection_token_ids[i : i + queries_per_engine]
      for i in range(0, len(reflection_token_ids), queries_per_engine)
  ]
  # Track prompt indices for each split
  split_prompt_indices = [
      list(range(i, min(i + queries_per_engine, len(reflection_token_ids))))
      for i in range(0, len(reflection_token_ids), queries_per_engine)
  ]

  # Generate in parallel across engines
  futures = []
  active_split_indices = []
  for engine_idx, (engine, queries) in enumerate(
      zip(vllm_engines, split_queries)
  ):
    if queries:
      futures.append(
          engine.generate.remote(
              sampling_params=sampling_params,
              prompt_token_ids=queries,
              use_tqdm=False,
          )
      )
      active_split_indices.append(engine_idx)

  all_outputs = ray_module.get(futures)

  response_ids = []
  finish_reasons = []
  expanded_question_indices = []
  expanded_trajectory_indices = []

  for split_idx, outputs in zip(active_split_indices, all_outputs):
    prompt_indices = split_prompt_indices[split_idx]
    for local_prompt_idx, output in enumerate(outputs):
      global_prompt_idx = prompt_indices[local_prompt_idx]
      q_idx = question_indices[global_prompt_idx]
      t_idx = trajectory_indices[global_prompt_idx]
      for out in output.outputs:
        response_ids.append(list(out.token_ids))
        finish_reasons.append(out.finish_reason)
        expanded_question_indices.append(q_idx)
        expanded_trajectory_indices.append(t_idx)

  return (
      response_ids,
      finish_reasons,
      expanded_question_indices,
      expanded_trajectory_indices,
  )


# =====================================================================
# 7. Judge Scoring + Bank Item Synthesis
# =====================================================================


async def _score_reflections_for_question(
    question,
    reflection_texts,
    graded_trajectories,
    trajectory_scores,
    trajectory_rubric_details = None,
    model_name = None,
    use_agent_reflection = True,
):
  """Judge scores all reflection candidates for one question.

  Optionally synthesizes a bank item (when use_agent_reflection=False).

  Args:
      question: The original question
      reflection_texts: Reflection candidates from the policy
      graded_trajectories: The decoded trajectory texts
      trajectory_scores: Overall scores for each trajectory
      trajectory_rubric_details: Per-trajectory list of rubric dicts
      model_name: Gemini model to use
      use_agent_reflection: If True, skip synthesis (agent's best reflection is
        used). If False, judge synthesizes a bank item (legacy mode).

  Returns:
      JudgeReflectionResult or None if scoring fails.
  """
  if model_name is None:
    model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gemini-3-flash-preview")

  # Format candidates
  candidates_parts = []
  for i, text in enumerate(reflection_texts):
    rubrics, takeaways = parse_reflection_output(text)
    candidates_parts.append(
        f"### Candidate {i}\n"
        f"**Rubrics:**\n{rubrics}\n\n"
        f"**Takeaways:**\n{takeaways}"
    )
  candidates_text = "\n\n---\n\n".join(candidates_parts)

  # Format graded trajectories with rubric details
  graded_parts = []
  for i, (traj, score) in enumerate(
      zip(graded_trajectories, trajectory_scores)
  ):
    # Truncate trajectory, preserving answer at the end
    if len(traj) > 30000:
      traj_display = traj[:15000] + "\n...[truncated]...\n" + traj[-15000:]
    else:
      traj_display = traj

    part = f"### Trajectory {i} (overall score: {score:.2f})\n{traj_display}"

    # Add rubric evaluation details if available
    if trajectory_rubric_details and i < len(trajectory_rubric_details):
      rubric_details = trajectory_rubric_details[i]
      if rubric_details:
        part += "\n\n#### Rubric Evaluation:\n"
        for rd in rubric_details:
          title = rd.get("title", "Untitled")
          criterion = rd.get("description", rd.get("criterion", ""))
          r_score = rd.get(
              "score", rd.get("effective_score", rd.get("raw_score", "N/A"))
          )
          justification = rd.get("justification", "")
          weight = rd.get("weight", 1.0)
          r_type = rd.get("type", "")
          part += (
              f"- **{title}** (weight={weight}, type={r_type}): "
              f"score={r_score}\n"
              f"  Criterion: {criterion}\n"
          )
          if justification:
            part += f"  Justification: {justification}\n"

    graded_parts.append(part)
  graded_text = "\n\n".join(graded_parts)

  if use_agent_reflection:
    sys_prompt = JUDGE_REFLECTION_SYSTEM_PROMPT
    usr_template = JUDGE_REFLECTION_USER_TEMPLATE
  else:
    sys_prompt = JUDGE_REFLECTION_SYSTEM_PROMPT_LEGACY
    usr_template = JUDGE_REFLECTION_USER_TEMPLATE_LEGACY

  user_prompt = usr_template.format(
      question=question,
      candidates_text=candidates_text,
      graded_trajectories_text=graded_text,
  )

  client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
  delay = 3.0
  for attempt in range(3):
    try:
      resp = await client.aio.models.generate_content(
          model=model_name,
          contents=user_prompt,
          config=types.GenerateContentConfig(
              system_instruction=sys_prompt,
              response_mime_type="application/json",
              response_json_schema=JudgeReflectionResult.model_json_schema(),
              temperature=0.2,
          ),
      )
      return JudgeReflectionResult.model_validate_json(resp.text)
    except Exception as e:  # pylint: disable=broad-exception-caught
      error_msg = str(e).lower()
      if any(k in error_msg for k in ["429", "quota", "500", "503", "timeout"]):
        jitter = random.uniform(1.0, 5.0)
        LOGGER.warning("[JudgeReflection] Retry %d/3: %s", attempt + 1, e)
        await asyncio.sleep(delay + jitter)
        delay *= 2
      else:
        LOGGER.error("[JudgeReflection] Non-retryable error: %s", e)
        return None

  LOGGER.error(
      "[JudgeReflection] All retries failed for '%s...'", question[:60]
  )
  return None


async def score_reflections_batch(
    questions,
    decoded_reflections,
    question_indices,
    trajectory_indices,
    decoded_responses,
    trajectory_scores,
    num_samples_per_prompt_rollout,
    all_rubric_details = None,
    use_agent_reflection = True,
):
  """Score all reflections in a batch via the judge.

  Groups reflections by TRAJECTORY (not question) so that the judge only
  compares candidates generated from the same input. When multiple
  trajectories are sampled per question, the bank item from the trajectory
  with the highest average candidate score is kept.

  Args:
      questions: Unique questions
      decoded_reflections: Decoded reflection texts
      question_indices: Which question each reflection belongs to
      trajectory_indices: Which trajectory each reflection was generated from
      decoded_responses: All decoded trajectory responses
      trajectory_scores: Scores for all trajectories
      num_samples_per_prompt_rollout: M rollouts per question
      all_rubric_details: Per-trajectory rubric evaluation details
      use_agent_reflection: Whether to use the agent's reflection directly

  Returns:
      reflection_scores: Per-reflection scores (for GRPO training)
      bank_items: Per-question synthesized bank items (for bank insertion)
      reflection_score_details: Per-reflection detailed scores (actionability,
      accuracy, transferability)
  """
  del num_samples_per_prompt_rollout
  # Group reflections by trajectory (not question)
  # Each trajectory gets its own judge call with only its candidates
  traj_reflections: Dict[int, List[str]] = {}  # traj_idx -> [refl_texts]
  traj_refl_global_indices: Dict[int, List[int]] = (
      {}
  )  # traj_idx -> [global refl indices]
  traj_to_q: Dict[int, int] = {}  # traj_idx -> q_idx
  for refl_idx, (q_idx, traj_idx) in enumerate(
      zip(question_indices, trajectory_indices)
  ):
    if traj_idx not in traj_reflections:
      traj_reflections[traj_idx] = []
      traj_refl_global_indices[traj_idx] = []
      traj_to_q[traj_idx] = q_idx
    traj_reflections[traj_idx].append(decoded_reflections[refl_idx])
    traj_refl_global_indices[traj_idx].append(refl_idx)

  # Fire all judge calls concurrently (one per trajectory)
  tasks = []
  task_traj_indices = []
  for traj_idx, refls in traj_reflections.items():
    q_idx = traj_to_q[traj_idx]

    # Get rubric details for this trajectory
    traj_rubric_details = None
    if all_rubric_details is not None and traj_idx < len(all_rubric_details):
      traj_rubric_details = [all_rubric_details[traj_idx]]

    task = _score_reflections_for_question(
        question=questions[q_idx],
        reflection_texts=refls,
        graded_trajectories=[decoded_responses[traj_idx]],
        trajectory_scores=[trajectory_scores[traj_idx]],
        trajectory_rubric_details=traj_rubric_details,
        use_agent_reflection=use_agent_reflection,
    )
    tasks.append(task)
    task_traj_indices.append(traj_idx)

  results = await asyncio.gather(*tasks, return_exceptions=True)

  # Build per-reflection scores and detailed score breakdowns
  reflection_scores = [0.0] * len(decoded_reflections)
  reflection_score_details = [None] * len(
      decoded_reflections
  )  # per-reflection detailed scores
  # Track bank items per trajectory, then pick best per question
  traj_bank_items: Dict[int, RubricBankItem] = {}  # traj_idx -> bank item
  traj_avg_scores: Dict[int, float] = {}  # traj_idx -> avg candidate score

  for traj_idx, result in zip(task_traj_indices, results):
    if isinstance(result, Exception):
      LOGGER.error(
          "[JudgeReflection] Exception for trajectory %d: %s",
          traj_idx,
          result,
      )
      continue
    if result is None:
      continue

    q_idx = traj_to_q[traj_idx]
    global_indices = traj_refl_global_indices[traj_idx]

    # Map candidate scores back to global reflection indices
    candidate_scores_sum = 0.0
    num_scored_candidates = 0
    for candidate_score in result.candidate_scores:
      ci = candidate_score.candidate_index
      if 0 <= ci < len(global_indices):
        reflection_scores[global_indices[ci]] = candidate_score.overall_score
        reflection_score_details[global_indices[ci]] = {
            "diagnostic_accuracy": candidate_score.diagnostic_accuracy,
            "specificity": candidate_score.specificity,
            "scope_and_balance": candidate_score.scope_and_balance,
            "overall_score": candidate_score.overall_score,
        }
        candidate_scores_sum += candidate_score.overall_score
        num_scored_candidates += 1

    avg_score = candidate_scores_sum / max(num_scored_candidates, 1)
    traj_avg_scores[traj_idx] = avg_score

    if use_agent_reflection:
      # Pick the best candidate with proper tags as the bank item.
      # Only consider candidates that have both <reflection_rubrics> and
      # <reflection_takeaways> tags to avoid storing raw <think> blocks.
      # Among complete candidates, pick highest score; tiebreak by shortest.
      best_ci = None
      best_score = -1.0
      best_len = float("inf")
      for candidate_score in result.candidate_scores:
        ci = candidate_score.candidate_index
        if 0 <= ci < len(global_indices):
          gi = global_indices[ci]
          text = decoded_reflections[gi]
          has_rubrics = bool(
              re.search(
                  r"<reflection_rubrics>.*</reflection_rubrics>",
                  text,
                  re.DOTALL,
              )
          )
          has_takeaways = bool(
              re.search(
                  r"<reflection_takeaways>.*</reflection_takeaways>",
                  text,
                  re.DOTALL,
              )
          )
          if not has_rubrics or not has_takeaways:
            continue  # skip candidates with missing tags
          rubrics, takeaways = parse_reflection_output(text)
          text_len = len(rubrics) + len(takeaways)
          if candidate_score.overall_score > best_score or (
              candidate_score.overall_score == best_score
              and text_len < best_len
          ):
            best_score = candidate_score.overall_score
            best_ci = gi
            best_len = text_len

      if best_ci is not None:
        best_rubrics, best_takeaways = parse_reflection_output(
            decoded_reflections[best_ci]
        )
        traj_bank_items[traj_idx] = RubricBankItem(
            question=questions[q_idx],
            rubrics=best_rubrics,
            reflections=best_takeaways,
        )
      else:
        LOGGER.warning(
            "[JudgeReflection] No complete reflection candidates for"
            " trajectory %d, skipping bank item insertion",
            traj_idx,
        )
    else:
      # Use Gemini synthesis as the bank item (legacy mode)
      if result.synthesized_rubrics and result.synthesized_reflections:
        traj_bank_items[traj_idx] = RubricBankItem(
            question=questions[q_idx],
            rubrics=result.synthesized_rubrics,
            reflections=result.synthesized_reflections,
        )

  # Pick best bank item per question (highest avg candidate score)
  bank_items: List[Optional[RubricBankItem]] = [None] * len(questions)
  best_score_per_q: Dict[int, float] = {}
  for traj_idx, item in traj_bank_items.items():
    q_idx = traj_to_q[traj_idx]
    avg = traj_avg_scores[traj_idx]
    if q_idx not in best_score_per_q or avg > best_score_per_q[q_idx]:
      best_score_per_q[q_idx] = avg
      bank_items[q_idx] = item

  num_scored = sum(1 for s in reflection_scores if s > 0.0)
  num_bank = sum(1 for b in bank_items if b is not None)
  LOGGER.info(
      "[JudgeReflection] Scored %d/%d reflections across %d trajectories,"
      " synthesized %d/%d bank items",
      num_scored,
      len(reflection_scores),
      len(traj_reflections),
      num_bank,
      len(questions),
  )

  return reflection_scores, bank_items, reflection_score_details
