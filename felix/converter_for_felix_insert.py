# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Training data conversion for FelixInsert.

Source-target text pairs will be converted to (source) token-level edit
operations and token-level insertions.
"""

from typing import List, Optional, Sequence, Tuple

from felix import felix_constants as constants
from felix import phrase_vocabulary_optimization_utils as phrase_utils


def compute_edits_and_insertions(
    source_tokens, target_tokens,
    max_insertions_per_token, insert_after_token = True
):
  """Computes edit operations and insertions per source token.

  Note that you should add a dummy token to the beginning / end of the source
  and target token lists if you want to be able to insert before the first
  actual token (when `insert_after_token==True`) / after the last actual token
  (when `insert_after_token==False`).

  Args:
    source_tokens: List of source tokens.
    target_tokens: List of target tokens.
    max_insertions_per_token: Maximum number of tokens to insert per source
      token.
    insert_after_token: Whether to insert after the current token (the current
      behavior on Felix) or before it (the current behavior in LaserTagger and
      in the original experimental FelixInsert implementation).

  Returns:
    None if target can't be obtained with the given `max_insertions_per_token`.
    Otherwise, a tuple with:
    1. List of edit operations ("KEEP" or "DELETE"), one per source token.
    2. List of inserted tokens, one per source token.
  """
  kept_tokens = phrase_utils.compute_lcs(source_tokens, target_tokens)
  # Added token lists between the kept source tokens.
  added_phrases = _get_added_token_lists(kept_tokens, target_tokens)
  # Regardless of input, every kept token (K_i) should be surrounded by an added
  # phrase (A_j, which can also be an empty list) on both sides, e.g.:
  #   [A_0, K_0, A_1, K_1, A_2].
  # Thus the number of added phrases has to be one larger than the number of
  # kept tokens.
  assert len(added_phrases) == len(kept_tokens) + 1, (
      f'Incorrect number of added phrases: {len(added_phrases)} != '
      f'{len(kept_tokens)} + 1')

  if insert_after_token:
    return _get_edits_and_insertions(kept_tokens, source_tokens, added_phrases,
                                     max_insertions_per_token)
  else:
    # When inserting before the current token, we can simply run the same
    # algorithm as above but for reversed lists and then reverse the output.
    edits_and_insertions = _get_edits_and_insertions(
        kept_tokens[::-1], source_tokens[::-1],
        _reverse_list_of_lists(added_phrases), max_insertions_per_token)
    if edits_and_insertions is not None:
      edits, insertions = edits_and_insertions
      return edits[::-1], _reverse_list_of_lists(insertions)
    else:
      return None


def _get_added_token_lists(kept_tokens,
                           target_tokens):
  """Return a list of added tokens lists next to every kept token."""
  added_phrases = []
  # Index of the `kept_tokens` element that we are currently looking for.
  kept_idx = 0
  phrase = []
  for token in target_tokens:
    if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
      kept_idx += 1
      added_phrases.append(phrase)
      phrase = []
    else:
      phrase.append(token)
  added_phrases.append(phrase)
  return added_phrases


def _get_edits_and_insertions(
    kept_tokens, source_tokens,
    added_phrases, max_insertions_per_token
):
  """Returns edit operations and insertions per source token."""
  edit_operations = []
  insertions = []
  kept_idx = 0
  current_added_phrase = list(added_phrases[kept_idx])
  for token in source_tokens:
    if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
      if current_added_phrase:
        # Couldn't insert all required tokens before the current kept token.
        return None
      kept_idx += 1
      current_added_phrase = list(added_phrases[kept_idx])
      edit_operations.append(constants.KEEP)
      # Insert as many tokens as possible after the current token and leave the
      # remaining to be added after next deleted tokens.
      insertions_i, current_added_phrase = (
          current_added_phrase[:max_insertions_per_token],
          current_added_phrase[max_insertions_per_token:])
      insertions.append(insertions_i)
    else:
      edit_operations.append(constants.DELETE)
      # If token i-1 is kept and token i deleted, the output will be the same
      # regardless of whether we insert new tokens after i-1 or after i.
      # However, semantically it makes more sense to insert after i since these
      # insertions typically correspond to replacing (e.g. inflecting) the ith
      # token. It also makes the tagging task easier since we need to predict
      # only one non-KEEP tag, i.e. DELETE|INSERT instead of independently
      # predicting KEEP|INSERT + DELETE.
      if (len(edit_operations) >= 2 and
          edit_operations[-2] == constants.KEEP and insertions[-1]):
        # Move the last insertion to the current token.
        insertions.append(insertions[-1])
        insertions[-2] = []
      else:
        insertions_i, current_added_phrase = (
            current_added_phrase[:max_insertions_per_token],
            current_added_phrase[max_insertions_per_token:])
        insertions.append(insertions_i)
  if current_added_phrase:
    # Tokens to be inserted remain but we've already consumed all source tokens.
    return None

  return edit_operations, insertions


def _reverse_list_of_lists(x):
  """Deep reverse of a list of lists."""
  return [sublist[::-1] for sublist in x[::-1]]
