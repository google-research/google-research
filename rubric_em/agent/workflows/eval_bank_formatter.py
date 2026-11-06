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

"""Eval-time rubric bank injection formatting.

Separate from the training-side rubric_bank_injection.py so we can iterate
on eval-time preambles and formatting without affecting training.
"""

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
  for token in _SPECIAL_TOKENS_TO_STRIP:
    text = text.replace(token, "")
  return text


def format_bank_items_for_eval(
    query,
    exact_match,
    similar_items,
    cross_episode_takeaways_only = False,
):
  """Format retrieved bank items as a <reference_examples> block for eval.

  Args:
      query: The current question.
      exact_match: RubricBankItem for within-episode exact match, or None.
      similar_items: List of (RubricBankItem, score) for cross-episode similar
        matches.
      cross_episode_takeaways_only: If True, similar items include only
        takeaways (not rubrics), and exact match includes only rubrics (not
        takeaways). Preambles are also adjusted to discourage over-indexing on
        domain-specific patterns.

  Returns:
      Formatted <reference_examples> string, or empty string if no items.
  """
  del query
  if exact_match is None and not similar_items:
    return ""

  parts = []

  # Within-episode (exact match): own prior attempt
  if exact_match is not None:
    if cross_episode_takeaways_only:
      parts.append(
          "## Your Previous Attempt on This Question:\n\n"
          "### Rubrics:\n"
          f"{_sanitize_text(exact_match.rubrics)}"
      )
    else:
      parts.append(
          "## Your Previous Attempt on This Question:\n\n"
          "### Rubrics:\n"
          f"{_sanitize_text(exact_match.rubrics)}\n\n"
          "### Takeaways:\n"
          f"{_sanitize_text(exact_match.reflections)}"
      )

  # Cross-episode (similar questions)
  for i, (item, _) in enumerate(similar_items):
    if cross_episode_takeaways_only:
      parts.append(
          f"## Similar Question {i+1}:\n"
          f"{_sanitize_text(item.question)}\n\n"
          "### Takeaways:\n"
          f"{_sanitize_text(item.reflections)}"
      )
    else:
      parts.append(
          f"## Similar Question {i+1}:\n"
          f"{_sanitize_text(item.question)}\n\n"
          "### Rubrics:\n"
          f"{_sanitize_text(item.rubrics)}\n\n"
          "### Takeaways:\n"
          f"{_sanitize_text(item.reflections)}"
      )

  reference_block = "\n\n---\n\n".join(parts)

  # Preambles — eval-specific, optimized to avoid over-indexing
  if exact_match is not None:
    if cross_episode_takeaways_only:
      # Within-episode preamble. The rubrics below are abstractions from
      # ONE prior attempt that happened to score well on some axes. They
      # describe one valid shape — not the only shape, and not requirements.
      # We frame them descriptively to avoid the failure mode where the
      # model treats them as must-follow templates and ends up answering
      # the principle's question instead of the patient's question.
      preamble = (
          "You have attempted this exact question before. Below are"
          " abstractions extracted from one prior attempt — they are"
          " DESCRIPTIVE, not prescriptive. They describe ONE valid shape an"
          " answer can take, not THE shape it must take.\n\nFIRST: re-read the"
          " patient's actual question and decide what they specifically need."
          " Answer THAT question — directly, in proportion to how it was asked,"
          " in the language it was asked in.\n\nTHEN: optionally consider the"
          " patterns below. If a pattern fits and helps, apply it. If a pattern"
          " does not fit the current question's actual needs (wrong scope,"
          " wrong language, wrong level of detail), IGNORE it. The patterns"
          " emerged from one specific instance and are not requirements for"
          " this attempt."
      )
    else:
      preamble = (
          "You have attempted this exact question before. Below are the rubrics"
          " and reflections from your previous attempt. Use them to improve"
          " your planning and answering — avoid past mistakes and build on what"
          " worked."
      )
  else:
    if cross_episode_takeaways_only:
      preamble = (
          "Below are takeaways from similar questions that were previously"
          " analyzed. Extract only the general research strategies and"
          " analytical patterns that transfer — do NOT adopt domain-specific"
          " rubrics, clinical protocols, or formatting styles from these"
          " examples. The current question has its own unique context and"
          " requirements that must take priority."
      )
    else:
      preamble = (
          "Below are rubrics and reflections from similar questions that were"
          " previously analyzed. Use them as reference to guide your planning"
          " and answering — adapt to the current question's specific needs. Do"
          " NOT copy them verbatim; extract what is relevant and adjust."
      )

  return (
      "\n\n<reference_examples>\n"
      f"{preamble}\n\n"
      f"{reference_block}\n"
      "</reference_examples>"
  )
