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

r"""Stage Segmentation Utilities for Fine-Grained Credit Assignment.

Segments agent trajectories into 4 stages:
  Stage 1 (PLAN):     Initial <think> + <structured_plan>
  Stage 2 (RESEARCH): First <call_tool> through all research turns up to
                       the last <think> before <review>
  Stage 3 (REVIEW):   Last <think> before <review> through </review>
  Stage 4 (ANSWER):   <answer>...</answer>

Trailing whitespace (\n, spaces) after a closing tag is absorbed into the
closing stage, not the opening of the next.  This ensures system-injected
newlines are attributed to the stage that just ended, not optimized as part
of the next stage's opening.
Tool outputs (<tool_output>...</tool_output>) are external observations and
are handled by the existing tool_mask — they are NOT policy tokens.

Missing-stage punishment propagation:
  - Missing answer  → trailing tokens assigned to review (stage 3)
  - Missing review  → content between research and answer assigned to stage 2
  - Missing plan    → leading tokens assigned to research (stage 2)
  Plan and answer are considered critical and should almost always be present.
"""

import collections
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import PreTrainedTokenizer

# =====================================================================
# 1. Character-level stage boundary detection
# =====================================================================


def find_stage_char_boundaries(decoded_response):
  r"""Find character-level [start, end) ranges for each of the 4 stages.

  Stage definitions (matching the system prompt):
    Stage 1 (PLAN):     start → end of </structured_plan> + trailing whitespace
    Stage 2 (RESEARCH): after stage 1 → start of stage 3
    Stage 3 (REVIEW):   last <think> before <review> → end of </review> +
    trailing whitespace
    Stage 4 (ANSWER):   <answer> → end of text

  Trailing whitespace (\n, spaces) after a closing tag is absorbed into the
  closing stage, not the opening of the next stage.  This ensures that
  system-injected newlines and formatting are attributed to the stage that
  just ended.

  Args:
    decoded_response: The decoded response text.

  Returns:
      List of 4 (start, end) tuples.  Missing stages get a zero-width
      range pinned to a neighbour boundary so that:
        - missing plan    → 0-width at start; text falls into stage 2
        - missing review  → 0-width at <answer>; text stays in stage 2
        - missing answer  → 0-width at end; trailing text pads to stage 3
  """
  total = len(decoded_response)

  # --- Stage 4: <answer> → end of text -----------------------------
  answer_open = decoded_response.find("<answer>")
  stage4_start = (
      answer_open if answer_open >= 0 else total
  )  # missing → 0-width at end
  stage4_end = total
  pre_answer = decoded_response[:stage4_start]

  # --- Stage 1: start → end of </structured_plan> + trailing ws ----
  sp_end = _find_tag_end(pre_answer, "</structured_plan>")
  if sp_end is not None:
    stage1_end = min(_skip_whitespace(decoded_response, sp_end), stage4_start)
  else:
    stage1_end = 0  # missing plan → 0-width

  # --- Stage 3: last <think> before <review> → start of stage 4 ----
  review_open = pre_answer.find("<review>", stage1_end)
  review_close = _find_tag_end(pre_answer, "</review>")
  if (
      review_open >= 0
      and review_close is not None
      and review_close > review_open
  ):
    # Find the last <think> before <review> that is AFTER stage 1
    last_think = pre_answer.rfind("<think>", stage1_end, review_open)
    stage3_start = last_think if last_think >= stage1_end else review_open
    stage3_end = stage4_start
  else:
    # No review → 0-width pinned to stage 4 start
    stage3_start = stage3_end = stage4_start

  # --- Stage 2: between stage 1 end and stage 3 start -------------
  stage2_start = stage1_end
  stage2_end = stage3_start

  # Clamp and order
  boundaries = [
      (0, stage1_end),
      (stage2_start, stage2_end),
      (stage3_start, stage3_end),
      (stage4_start, stage4_end),
  ]

  # Ensure no negative-width ranges
  sanitized = []
  for s, e in boundaries:
    s = max(0, min(s, total))
    e = max(s, min(e, total))
    sanitized.append((s, e))

  return sanitized


def extract_stage_texts(decoded_response):
  """Extract the text content for each stage (1-indexed).

  Args:
    decoded_response: The decoded response text.

  Returns:
      {1: plan_text, 2: research_text, 3: review_text, 4: answer_text}
  """
  bounds = find_stage_char_boundaries(decoded_response)
  return {k + 1: decoded_response[s:e] for k, (s, e) in enumerate(bounds)}


# =====================================================================
# 2. Token-level stage ID assignment
# =====================================================================


def get_token_stage_ids(
    decoded_response,
    response_tokens,
    tokenizer,
):
  """Assign a stage ID (1–4) to each token in ``response_tokens``.

  Strategy:
    1. Find character-level stage boundaries.
    2. Encode each stage's text separately → count tokens per stage.
    3. Distribute stage IDs proportionally across the token list.
    4. Clamp / pad to match the actual token count.

  This is an O(4·encode) approximation that avoids O(n²) incremental
  decoding.  Boundary errors of ±1-2 tokens are acceptable for
  advantage assignment.

  Args:
    decoded_response: The decoded response text.
    response_tokens: Token IDs of the response.
    tokenizer: Tokenizer used for encoding stage segments.

  Returns:
    List of stage IDs (1-4) corresponding to each token in response_tokens.
  """
  bounds = find_stage_char_boundaries(decoded_response)
  stage_token_counts = []

  for start, end in bounds:
    text = decoded_response[start:end]
    if text.strip():
      toks = tokenizer.encode(text, add_special_tokens=False)
      stage_token_counts.append(len(toks))
    else:
      stage_token_counts.append(0)

  # Build flat stage ID list
  stage_ids: List[int] = []
  for stage_idx, count in enumerate(stage_token_counts):
    stage_ids.extend([stage_idx + 1] * count)

  # Adjust to match the actual response length
  actual_len = len(response_tokens)
  if len(stage_ids) > actual_len:
    stage_ids = stage_ids[:actual_len]
  elif len(stage_ids) < actual_len:
    last_stage = stage_ids[-1] if stage_ids else 4
    stage_ids.extend([last_stage] * (actual_len - len(stage_ids)))

  return stage_ids


# =====================================================================
# 3. Stage return computation (discount matrix)
# =====================================================================

# Lower-triangular credit propagation matrix  (Eq. in formal statement)
#   Row k gives the weights for computing G_k from [r_1, r_2, r_3, r_4].
STAGE_DISCOUNT_MATRIX = np.array(
    [
        [1.0, 0.4, 0.6, 0.8],  # G_1 plan
        [0.0, 1.0, 0.5, 0.8],  # G_2 search
        [0.0, 0.0, 1.0, 0.8],  # G_3 review
        [0.0, 0.0, 0.0, 1.0],  # G_4 answer
    ],
    dtype=np.float64,
)


def compute_stage_returns(
    stage_scores,
    discount_matrix = STAGE_DISCOUNT_MATRIX,
):
  """Compute stage returns G from raw stage scores r.

  Args:
      stage_scores: shape (N, 4) — raw stage scores for N rollouts.
      discount_matrix: (4, 4) — C such that G = r @ C^T  (row k of C gives the
        weights for G_k).

  Returns:
      stage_returns: shape (N, 4).
  """
  # G[i] = C @ r[i]  ⟹  G = r @ C^T
  return stage_scores @ discount_matrix.T


def compute_stagewise_advantages(
    stage_returns,
    prompt_indices,
    normalization = "standard",
    eps = 1e-8,
):
  """Group-relative advantage, computed independently per stage.

  Args:
      stage_returns:  (N, 4)
      prompt_indices: (N,)  — which prompt each rollout belongs to
      normalization:  "standard" | "centered"
      eps:            numerical stability

  Returns:
      advantages: (N, 4) — per-stage advantages
  """
  _, num_stages = stage_returns.shape
  advantages = np.zeros_like(stage_returns)
  unique_prompts = np.unique(prompt_indices)

  for p_idx in unique_prompts:
    mask = prompt_indices == p_idx
    for k in range(num_stages):
      group = stage_returns[mask, k]
      mu = group.mean()
      sigma = group.std(ddof=0)
      if normalization == "standard":
        advantages[mask, k] = (group - mu) / (sigma + eps)
      elif normalization == "centered":
        advantages[mask, k] = group - mu
      else:
        raise ValueError(f"Unknown normalization: {normalization}")

  return advantages


def compute_stagewise_advantages_for_batch(
    stage_scores_array,  # np.ndarray (N, 4) — scaled per-stage scores
    prompt_indices,  # np.ndarray (N_kept,) — prompt index for each kept sample
    args,  # Args object
):
  """Compute per-stage advantages using discount matrix and normalization.

  Args:
    stage_scores_array: np.ndarray (N, 4) of scaled per-stage scores.
    prompt_indices: np.ndarray (N_kept,) of prompt indices for kept samples.
    args: Configuration arguments object.

  Returns:
      stage_advantages: np.ndarray (N_kept, 4) — per-stage advantages
  """

  # 1. Compute stage returns: G = r @ C^T
  stage_returns = compute_stage_returns(stage_scores_array)  # (N_kept, 4)

  # 2. Per-stage group-relative normalization
  stage_advantages = compute_stagewise_advantages(
      stage_returns,
      prompt_indices,
      normalization=args.advantage_normalization_type
      if args.advantage_normalization_type in ("standard", "centered")
      else "standard",
  )

  return stage_advantages


def pack_stagewise_advantages(
    responses,
    decoded_responses,
    stage_advantages,  # (N_traj, 4) array — only for trajectory samples
    tokenizer,
    packed_sequences,  # the PackedSequences object from pack_sequences
    num_traj_samples = None,
    # scalar advantages for non-trajectory samples (e.g., reflections)
    scalar_advantages=None,
):
  """Build packed per-token advantages using stage membership.

  Uses stage membership for trajectory samples, and scalar advantages for
  non-trajectory samples (e.g., reflections).

  Args:
      responses: All responses (trajectories + optional reflections)
      decoded_responses: All decoded responses
      stage_advantages: (N_traj, 4) per-stage advantages for trajectory samples
        only
      tokenizer: The tokenizer
      packed_sequences: PackedSequences from pack_sequences
      num_traj_samples: Number of trajectory samples (first N responses). If
        None, all responses are treated as trajectories.
      scalar_advantages: Scalar advantages for non-trajectory samples. Must have
        length len(responses) - num_traj_samples.

  Returns:
      List of torch.Tensor — one per packed sequence, same shape as
      packed_sequences.response_masks.
  """
  if num_traj_samples is None:
    num_traj_samples = len(responses)

  # 1. Pre-compute per-token advantages for trajectory responses (stage-based)
  traj_responses = responses[:num_traj_samples]
  traj_decoded = decoded_responses[:num_traj_samples]
  per_response_token_advs = build_per_token_advantages(
      traj_responses, traj_decoded, stage_advantages, tokenizer
  )

  # 2. For non-trajectory samples (reflections), each token gets the same
  # scalar advantage
  if scalar_advantages is not None and len(scalar_advantages) > 0:
    for i, adv_val in enumerate(scalar_advantages):
      resp_len = len(responses[num_traj_samples + i])
      per_response_token_advs.append(
          np.full(resp_len, float(adv_val), dtype=np.float32)
      )

  # 3. Build packed advantages by walking through response_masks
  #    response_masks[j][t] = 0 for prompt/pad, or 1-indexed response index
  packed_advantages = []

  for packed_mask in packed_sequences.response_masks:
    adv_tensor = torch.zeros(len(packed_mask), dtype=torch.float32)
    response_token_counters = collections.defaultdict(int)

    for t, mask_val in enumerate(packed_mask):
      resp_idx = (
          mask_val.item()
          if isinstance(mask_val, torch.Tensor)
          else mask_val
      )
      if resp_idx > 0:  # this is a response token
        pos = response_token_counters[resp_idx]
        resp_array = per_response_token_advs[resp_idx - 1]  # 0-indexed
        if pos < len(resp_array):
          adv_tensor[t] = float(resp_array[pos])
        response_token_counters[resp_idx] += 1

    packed_advantages.append(adv_tensor)

  return packed_advantages


def compute_stage_metrics(stage_scores_array, stage_advantages, prefix="val"):
  """Compute per-stage logging metrics.

  Args:
      stage_scores_array: (N, 4) raw scaled scores
      stage_advantages:   (N, 4) per-stage advantages
      prefix: metric name prefix

  Returns:
      dict of metric_name -> value
  """
  metrics = {}
  stage_names = ["plan", "research", "review", "answer"]

  for k in range(4):
    name = stage_names[k]
    scores_k = stage_scores_array[:, k]
    advs_k = stage_advantages[:, k]

    metrics[f"{prefix}/stage_{name}_score_mean"] = float(scores_k.mean())
    metrics[f"{prefix}/stage_{name}_score_std"] = float(scores_k.std())
    metrics[f"{prefix}/stage_{name}_advantage_mean"] = float(advs_k.mean())
    metrics[f"{prefix}/stage_{name}_advantage_std"] = float(advs_k.std())
    metrics[f"{prefix}/stage_{name}_advantage_min"] = float(advs_k.min())
    metrics[f"{prefix}/stage_{name}_advantage_max"] = float(advs_k.max())

  # Cross-stage correlations (useful for monitoring credit propagation)
  if stage_scores_array.shape[0] > 2:
    for i in range(4):
      for j in range(i + 1, 4):
        si, sj = stage_scores_array[:, i], stage_scores_array[:, j]
        # Skip if either column has zero variance (corrcoef divides by std)
        if si.std() < 1e-10 or sj.std() < 1e-10:
          corr_val = 0.0
        else:
          corr_val = float(np.corrcoef(si, sj)[0, 1])
          if np.isnan(corr_val):
            corr_val = 0.0
        metrics[f"{prefix}/stage_corr_{stage_names[i]}_{stage_names[j]}"] = (
            corr_val
        )

  return metrics


def augment_step_log_with_stage_stats(step_log, all_sample_logs):
  """Add stage-level summary statistics to the step log."""

  stage_names = ["plan", "research", "review", "answer"]
  stage_score_lists = {k: [] for k in range(4)}

  for log in all_sample_logs:
    raw_scores = log.get("stage_scores_raw", [])
    for k in range(min(4, len(raw_scores))):
      if not np.isnan(raw_scores[k]):
        stage_score_lists[k].append(raw_scores[k])

  stage_summary = {}
  for k in range(4):
    vals = stage_score_lists[k]
    if vals:
      arr = np.array(vals)
      stage_summary[stage_names[k]] = {
          "mean": round(float(arr.mean()), 4),
          "std": round(float(arr.std()), 4),
          "min": round(float(arr.min()), 4),
          "max": round(float(arr.max()), 4),
          "count": len(vals),
      }

  step_log["summary"]["stage_scores"] = stage_summary
  return step_log


# =====================================================================
# 4. Per-token advantage builder (replaces scalar lookup)
# =====================================================================


def build_per_token_advantages(
    responses,
    decoded_responses,
    stage_advantages,  # (N, 4)
    tokenizer,
):
  """Build a per-token advantage array for every response.

  Args:
    responses: Tokenized responses.
    decoded_responses: Decoded response strings.
    stage_advantages: Array of shape (N, 4) with per-stage advantages.
    tokenizer: Tokenizer instance.

  Returns:
      List of 1-D float arrays, one per response, same length as the
      response token list.  Each token gets the advantage of its stage.
  """
  per_token_advs = []
  for i, resp_tokens in enumerate(responses):
    stage_ids = get_token_stage_ids(
        decoded_responses[i], resp_tokens, tokenizer
    )
    # stage_ids are 1-indexed; stage_advantages columns are 0-indexed
    token_advs = np.array(
        [stage_advantages[i, sid - 1] for sid in stage_ids],
        dtype=np.float32,
    )
    per_token_advs.append(token_advs)
  return per_token_advs


# =====================================================================
# Helpers
# =====================================================================


def _skip_whitespace(text, pos):
  """Advance *pos* past any whitespace characters (spaces, newlines, tabs)."""
  while pos < len(text) and text[pos] in (" ", "\n", "\t", "\r"):
    pos += 1
  return pos


def _find_tag_end(text, tag, occurrence = 1):
  """Return character index right AFTER the n-th occurrence of *tag*."""
  start = 0
  for _ in range(occurrence):
    idx = text.find(tag, start)
    if idx < 0:
      return None
    start = idx + len(tag)
  return start


def validate_stage_segmentation(decoded_response):
  """Quick check whether a response has valid stage markers.

  Useful for deciding whether to apply stage-wise scoring or fallback.

  ``present_stages`` is a list of 1-indexed stage IDs that are present
  (e.g. [1,2,4] when review is missing).  ``missing_stages`` is the
  complement.

  Args:
    decoded_response: The decoded response text.

  Returns:
    Dictionary containing stage presence and format quality flags.
  """
  has_plan = (
      "<structured_plan>" in decoded_response
      and "</structured_plan>" in decoded_response
  )
  has_research = (
      "<call_tool" in decoded_response or "</call_tool>" in decoded_response
  )
  has_review = (
      "<review>" in decoded_response and "</review>" in decoded_response
  )
  has_answer = "<answer>" in decoded_response

  present = []
  if has_plan:
    present.append(1)
  if has_research:
    present.append(2)
  if has_review:
    present.append(3)
  if has_answer:
    present.append(4)
  missing = [k for k in [1, 2, 3, 4] if k not in present]

  # Sub-component checks for plan quality
  has_initial_think = decoded_response.lstrip().startswith("<think>")
  has_deep_analysis = (
      "<deep_analysis>" in decoded_response
      and "</deep_analysis>" in decoded_response
  )
  has_rubric = (
      "<rubric>" in decoded_response and "</rubric>" in decoded_response
  )
  has_research_plan = (
      "<research_plan>" in decoded_response
      and "</research_plan>" in decoded_response
  )

  plan_sub_components = {
      "deep_analysis": has_deep_analysis,
      "rubric": has_rubric,
      "research_plan": has_research_plan,
  }
  num_plan_sub_missing = sum(1 for v in plan_sub_components.values() if not v)

  return {
      "has_structured_plan": has_plan,
      "has_call_tool": has_research,
      "has_review": has_review,
      "has_answer": has_answer,
      "has_all_stages": len(missing) == 0,
      "present_stages": present,
      "missing_stages": missing,
      # Format quality checks
      "has_initial_think": has_initial_think,
      "plan_sub_components": plan_sub_components,
      "num_plan_sub_missing": num_plan_sub_missing,
  }


def compute_format_penalty(
    stage_validity,
    no_think_penalty = 0.8,
    sub_component_penalty = 0.3,
):
  """Compute per-stage score multipliers based on format quality.

  Penalty hierarchy (applied in order, first match wins for hard kills):
    1. HARD KILL — missing <structured_plan> or <answer> → all zeros
    2. SOFT PLAN — missing initial <think> or plan sub-components
       (<deep_analysis>, <rubric>, <research_plan>) reduces stage-1
       multiplier only.  Other stages are unaffected.

  Args:
      stage_validity: dict from validate_stage_segmentation()
      no_think_penalty: plan multiplier when initial <think> is missing (default
        0.8, i.e. 20% reduction)
      sub_component_penalty: plan reduction per missing sub-component (default
        0.3)

  Returns:
      (multipliers, reason)
      multipliers: list of 4 floats, one per stage (1.0 = no penalty)
      reason: human-readable string describing what triggered the penalty
  """
  # --- Hard kill: missing critical stages ---
  if not stage_validity["has_structured_plan"]:
    return [0.0, 0.0, 0.0, 0.0], "hard_kill:missing_structured_plan"
  if not stage_validity["has_answer"]:
    return [0.0, 0.0, 0.0, 0.0], "hard_kill:missing_answer"

  multipliers = [1.0, 1.0, 1.0, 1.0]
  reasons = []

  # --- Soft plan: missing initial <think> ---
  if not stage_validity.get("has_initial_think", True):
    multipliers[0] *= no_think_penalty
    reasons.append(f"no_initial_think(plan x{no_think_penalty})")

  # --- Soft plan: missing sub-components ---
  n_missing = stage_validity.get("num_plan_sub_missing", 0)
  if n_missing > 0:
    plan_factor = max(0.0, 1.0 - sub_component_penalty * n_missing)
    multipliers[0] *= plan_factor
    missing_subs = [
        k
        for k, v in stage_validity.get("plan_sub_components", {}).items()
        if not v
    ]
    reasons.append(
        f"plan_sub_missing({','.join(missing_subs)},x{plan_factor:.2f})"
    )

  reason = "|".join(reasons) if reasons else "none"
  return multipliers, reason
