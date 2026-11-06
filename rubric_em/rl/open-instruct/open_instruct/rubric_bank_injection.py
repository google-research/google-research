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

"""Rubric Bank Injection for vLLM prompt generation.

Retrieves bank items and appends them after the user question in the tokenized
prompt for vLLM generation.

Strategy: tokenize the reference block, find the end of the user message
(last <|im_end|> before <|im_start|>assistant), and splice the reference
tokens just before that boundary.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from open_instruct.rubric_bank import RubricBank
from open_instruct.rubric_bank import RubricBankItem
from transformers import PreTrainedTokenizer

LOGGER = logging.getLogger(__name__)


_SPECIAL_TOKENS_TO_STRIP = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|im_sep|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
]


def _sanitize_text(text):
  """Remove special tokens that could corrupt tokenization."""
  for token in _SPECIAL_TOKENS_TO_STRIP:
    text = text.replace(token, "")
  return text


def _format_bank_items_as_reference(
    query,
    exact_match,
    similar_items,
):
  """Format retrieved bank items as an XML reference block.

  Uses different preamble for exact-match (same question seen before)
  vs similar questions.

  Args:
    query: The user question.
    exact_match: Exact-match bank item if available.
    similar_items: List of (item, similarity_score) pairs.

  Returns:
    Formatted XML reference block string.
  """
  del query
  if exact_match is None and not similar_items:
    return ""

  parts = []

  if exact_match is not None:
    parts.append(
        "## Your Previous Attempt on This Question:\n\n"
        "### Rubrics:\n"
        f"{_sanitize_text(exact_match.rubrics)}\n\n"
        "### Takeaways:\n"
        f"{_sanitize_text(exact_match.reflections)}"
    )

  for i, (item, _) in enumerate(similar_items):
    parts.append(
        f"## Similar Question {i+1}:\n"
        f"{_sanitize_text(item.question)}\n\n"
        "### Rubrics:\n"
        f"{_sanitize_text(item.rubrics)}\n\n"
        "### Takeaways:\n"
        f"{_sanitize_text(item.reflections)}"
    )

  reference_block = "\n\n---\n\n".join(parts)

  if exact_match is not None:
    preamble = (
        "You have attempted this exact question before. Below are the rubrics "
        "and reflections from your previous attempt. Use them to improve your "
        "planning and answering — avoid past mistakes and build on what worked."
    )
  else:
    preamble = (
        "Below are rubrics and reflections from similar questions that were"
        " previously analyzed. Use them as reference to guide your planning and"
        " answering — adapt to the current question's specific needs. Do NOT"
        " copy them verbatim; extract what is relevant and adjust."
    )

  return (
      "\n\n<reference_examples>\n"
      f"{preamble}\n\n"
      f"{reference_block}\n"
      "</reference_examples>"
  )


def inject_bank_items_into_prompts(
    queries_token_ids,
    raw_user_queries,
    rubric_bank,
    tokenizer,
    top_k = 2,
    max_prompt_token_length = 4096,
    skip_exact_match = False,
):
  """Retrieve bank items for each question and append to tokenized prompts.

  Strategy:
  1. Check for exact match (same question seen before)
  2. Retrieve top_k similar items (excluding self)
  3. Format as XML reference block
  4. Find end of user message in token IDs (last <|im_end|>)
  5. Splice reference tokens just before that boundary

  Args:
      queries_token_ids: Pre-tokenized prompt token IDs.
      raw_user_queries: Raw text of each user question.
      rubric_bank: The rubric bank to retrieve from.
      tokenizer: The tokenizer for encoding the reference block.
      top_k: Number of similar items to retrieve per question.
      max_prompt_token_length: Maximum token length for the prompt.
      skip_exact_match: If True, skip exact-match retrieval and only use similar
        items. Used on the first step of a new L-window to force diversity.

  Returns:
      Tuple of:
          - Modified list of token ID lists (same length as input).
          - Retrieval log: list of dicts with query, match_type,
          injection_status, and retrieved items.
          - Injection stats: dict with counts of injected, fallback, skipped,
          total.
  """
  _empty_stats = {
      "num_injected": 0,
      "num_fallback_1shot": 0,
      "num_skipped_too_long": 0,
      "num_total": len(queries_token_ids),
  }
  if len(rubric_bank) == 0:
    return queries_token_ids, [], _empty_stats

  # Find the <|im_end|> token ID
  im_end_token_id = None
  for special_token in ["<|im_end|>", "<|eot_id|>"]:
    token_id = tokenizer.convert_tokens_to_ids(special_token)
    if token_id != tokenizer.unk_token_id:
      im_end_token_id = token_id
      break

  if im_end_token_id is None:
    LOGGER.warning(
        "[BankInjection] Could not find end-of-turn token, skipping injection"
    )
    return queries_token_ids, [], _empty_stats

  # Cache: raw_query -> (reference block token IDs or None, retrieval_info dict)
  retrieval_cache: Dict[str, Tuple[Optional[List[int]], Dict[str, Any]]] = {}

  modified_token_ids = []
  injection_count = 0
  fallback_count = 0
  skipped_count = 0
  retrieval_log = []  # per-unique-query retrieval metadata

  for token_ids, raw_query in zip(queries_token_ids, raw_user_queries):
    if raw_query not in retrieval_cache:
      # Check for exact match (skip if first step of new L-window to force
      # diversity)
      exact_match = None if skip_exact_match else rubric_bank.get(raw_query)
      # Exact match: just use the previous attempt (no similar items needed)
      # New question (or skipped exact): retrieve top_k similar items
      if exact_match is not None:
        similar_items = []
      else:
        similar_items = rubric_bank.retrieve(
            raw_query, top_k=top_k, exclude_self=True, stochastic=True
        )

      # Build retrieval metadata for tracing
      retrieval_info = {"query": raw_query}
      if exact_match is not None:
        retrieval_info["match_type"] = "exact"
        retrieval_info["rubrics"] = exact_match.rubrics
        retrieval_info["reflections"] = exact_match.reflections
      elif similar_items:
        retrieval_info["match_type"] = "similar"
        retrieval_info["similar_items"] = [
            {
                "question": it.question,
                "score": round(sc, 4),
                "rubrics": it.rubrics,
                "reflections": it.reflections,
            }
            for it, sc in similar_items
        ]
      else:
        retrieval_info["match_type"] = "none"
        retrieval_info["injection_status"] = "no_match"
        retrieval_log.append(retrieval_info)
        retrieval_cache[raw_query] = (None, retrieval_info)
        modified_token_ids.append(token_ids)
        continue

      reference_text = _format_bank_items_as_reference(
          raw_query, exact_match, similar_items
      )
      if reference_text:
        ref_token_ids = tokenizer.encode(
            reference_text, add_special_tokens=False
        )
      else:
        retrieval_info["injection_status"] = "no_match"
        retrieval_log.append(retrieval_info)
        retrieval_cache[raw_query] = (None, retrieval_info)
        modified_token_ids.append(token_ids)
        continue

      # Try injection
      insert_pos = None
      for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == im_end_token_id:
          insert_pos = i
          break

      if insert_pos is None:
        retrieval_info["injection_status"] = "skipped_no_boundary"
        retrieval_log.append(retrieval_info)
        retrieval_cache[raw_query] = (None, retrieval_info)
        modified_token_ids.append(token_ids)
        continue

      new_token_ids = (
          token_ids[:insert_pos] + ref_token_ids + token_ids[insert_pos:]
      )

      if len(new_token_ids) <= max_prompt_token_length:
        retrieval_info["injection_status"] = "injected"
        retrieval_log.append(retrieval_info)
        retrieval_cache[raw_query] = (ref_token_ids, retrieval_info)
        modified_token_ids.append(new_token_ids)
        injection_count += 1
        continue

      # Fallback: retry with only 1 similar item
      if exact_match is None and len(similar_items) > 1:
        fallback_text = _format_bank_items_as_reference(
            raw_query, None, similar_items[:1]
        )
        if fallback_text:
          fallback_tokens = tokenizer.encode(
              fallback_text, add_special_tokens=False
          )
          fallback_new = (
              token_ids[:insert_pos] + fallback_tokens + token_ids[insert_pos:]
          )
          if len(fallback_new) <= max_prompt_token_length:
            retrieval_info["injection_status"] = "fallback_1shot"
            retrieval_log.append(retrieval_info)
            retrieval_cache[raw_query] = (fallback_tokens, retrieval_info)
            modified_token_ids.append(fallback_new)
            injection_count += 1
            fallback_count += 1
            continue

      retrieval_info["injection_status"] = "skipped_too_long"
      retrieval_log.append(retrieval_info)
      retrieval_cache[raw_query] = (None, retrieval_info)
      modified_token_ids.append(token_ids)
      skipped_count += 1
      continue

    # Cache hit: reuse previous result
    ref_tokens, _ = retrieval_cache[raw_query]
    if ref_tokens is None:
      modified_token_ids.append(token_ids)
      continue

    insert_pos = None
    for i in range(len(token_ids) - 1, -1, -1):
      if token_ids[i] == im_end_token_id:
        insert_pos = i
        break

    if insert_pos is None:
      modified_token_ids.append(token_ids)
      continue

    new_token_ids = token_ids[:insert_pos] + ref_tokens + token_ids[insert_pos:]
    if len(new_token_ids) > max_prompt_token_length:
      modified_token_ids.append(token_ids)
      skipped_count += 1
    else:
      modified_token_ids.append(new_token_ids)
      injection_count += 1

  if injection_count > 0:
    num_unique = len(retrieval_cache)
    LOGGER.info(
        "[BankInjection] Injected bank items into %d/%d prompts (%d unique"
        " questions)",
        injection_count,
        len(queries_token_ids),
        num_unique,
    )

  # Injection stats for logging
  injection_stats = {
      "num_injected": injection_count,
      "num_fallback_1shot": fallback_count,
      "num_skipped_too_long": skipped_count,
      "num_total": len(queries_token_ids),
  }

  return modified_token_ids, retrieval_log, injection_stats
