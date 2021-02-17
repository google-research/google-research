# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Construct the input for the insertion model.

The code realizes the output of the insertion model. Realization involves
three tasks: (1) Re-arranging the source tokens; (2) Inserting
masks token; and (3) Inserting deleted tokens.

(1) Re-arranges the source tokens according to a list of indexes. source_index
is the index to the next token (first token is always included first). See
pointing_converter for more details.

Example:
source_tokens: [a, b ,c]
source_indexes: [2, 1, 0]
output_move: [a, c, b]


(2) Insert MASK token using labels. Each label indicates how many (if any) mask
tokens should be appended.

Example:

output_move: [a,b,c]
labels: [KEEP, INSERT|2, INSERT|1]

output_mask: [a, b, MASK, MASK, c, MASK]


(3) Insert deleted tokens, for which the corresponding label is DELETE.
and finding the appropriate place to insert these tokens. We try and preserve
the relative position of the deleted tokens within the output, by having them
appear next to the nearest token from the source.

In addition we bracket the deleted tokens.
source tokens: [a, b, c, d]
output_mask: [d, b, c, a]
labels: [KEEP, DELETE, DELETE, KEEP]

output: [(, b, c, ), d, a]


Note: In actuality bracket characters are not used but instead unused0 is paired
with unused1.


Within the code all three steps happen simultaneously.
In addition we record what each MASK token corresponds to.
"""

import random
from typing import List, Mapping, Optional

import tensorflow as tf

from felix import felix_constants as constants
from felix import tokenization
from felix import utils


def int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
  return f


def float_feature(values):
  f = tf.train.Feature(float_list=tf.train.FloatList(value=values))
  return f


def get_number_of_masks(label):
  """Convert a tag to the number of MASK tokens it represents."""

  if '|' not in label:
    return 0
  return int(label.split('|')[1])


class InsertionConverter(object):
  """Class for creating insertion examples."""

  def __init__(self,
               max_seq_length,
               max_predictions_per_seq,
               label_map,
               vocab_file = None,
               do_lower_case=True,
               fall_back_mode='random'):
    """Initializes an instance of InsertionConverter.

    Args:
      max_seq_length: Maximum length of source sequence.
      max_predictions_per_seq: Maximum number of MASK tokens.
      label_map: Dictionary to convert labels_ids to labels.
      vocab_file: Path to BERT vocabulary file.
      do_lower_case: text is lowercased.
      fall_back_mode: In the case no MASK tokens are generated:
                      'random': Randomly add MASK tokens.
                      'force':  Leave the output unchanged (not recommended).
                        Otherwise return None and terminate early (saving
                        computation time).
    """

    self._max_seq_length = max_seq_length
    self._max_predictions_per_seq = max_predictions_per_seq
    self._tokenizer = tokenization.FullTokenizer(
        vocab_file, do_lower_case=do_lower_case)
    self._label_map = label_map
    self._label_map_inverse = {v: k for k, v in self._label_map.items()}
    if fall_back_mode.lower() == 'random':
      self._do_random_mask = True
      self._do_lazy_generation = False
    elif fall_back_mode.lower() == 'force':
      self._do_random_mask = False
      self._do_lazy_generation = False
    else:
      self._do_random_mask = False
      self._do_lazy_generation = True

  def _create_masked_source(self, source_tokens, labels, source_indexes,
                            target_tokens):
    """Realizes source_tokens & adds deleted to source_tokens and target_tokens.

    Args:
      source_tokens: List of source tokens.
      labels: List of label IDs, which correspond to a list of labels (KEEP,
        DELETE, MASK|1, MASK|2...).
      source_indexes: List of next tokens (see pointing converter for more
        details) (ordered by source tokens)
      target_tokens: Optional list of target tokens. Only provided when
        constructing training examples.

    Returns:
      masked_tokens: The source input for the insertion model, including MASK
        tokens and bracketed deleted tokens.
      target_tokens: The target tokens for the insertion model, where mask
        tokens are replaced with the actual token, also includes bracketed
        deleted tokens.
    """

    current_index = 0
    masked_tokens = []

    kept_tokens = set([0])
    for _ in range(len(source_tokens)):
      current_index = source_indexes[current_index]
      kept_tokens.add(current_index)
      # Token is deleted.
      if current_index == 0:
        break
    current_index = 0
    for _ in range(len(source_tokens)):
      source_token = source_tokens[current_index]
      deleted_tokens = []
      # Looking forward finding all deleted tokens.
      for i in range(current_index + 1, len(source_tokens)):
        ## If not a deleted token.
        if i in kept_tokens:
          break
        deleted_tokens.append(source_tokens[i])

      # Add deleted tokens to masked_tokens and target_tokens.
      masked_tokens.append(source_token)
      # number_of_masks specifies the number MASKED tokens which
      # are added to masked_tokens.
      number_of_masks = get_number_of_masks(
          self._label_map_inverse[labels[current_index]])
      for _ in range(number_of_masks):
        masked_tokens.append(constants.MASK)
      if deleted_tokens:
        masked_tokens_length = len(masked_tokens)
        bracketed_deleted_tokens = ([constants.DELETE_SPAN_START] +
                                    deleted_tokens +
                                    [constants.DELETE_SPAN_END])
        target_tokens = (
            target_tokens[:masked_tokens_length] + bracketed_deleted_tokens +
            target_tokens[masked_tokens_length:])
        masked_tokens += bracketed_deleted_tokens

      current_index = source_indexes[current_index]
      if current_index == 0:
        break
    return masked_tokens, target_tokens

  def create_insertion_example(
      self, source_tokens, labels,
      source_indexes,
      target_tokens):
    """Creates training/test features for insertion model.

    Args:
      source_tokens: List of source tokens.
      labels: List of label IDs, which correspond to a list of labels (KEEP,
        DELETE, MASK|1, MASK|2...).
      source_indexes: List of next tokens (see pointing converter for more
        details) (ordered by source tokens).
      target_tokens: List of target tokens.

    Returns:
       A dictionary of features needed by the tensorflow insertion model.
    """

    # Reorder source sentence, add MASK tokens, adds deleted tokens
    # (to both source_tokens and target_tokens).
    masked_tokens, target_tokens = self._create_masked_source(
        source_tokens, labels, source_indexes, target_tokens)

    if target_tokens and constants.MASK not in masked_tokens:
      # Generate random MASKs.
      if self._do_random_mask:
        # Don't mask the start or end token.
        indexes = list(range(1, len(masked_tokens) - 1))
        random.shuffle(indexes)
        # Limit MASK to ~10% of the source tokens.
        indexes = indexes[:int(len(masked_tokens) * 0.1)]
        for index in indexes:
          masked_tokens[index] = constants.MASK
      elif self._do_lazy_generation:
        return None
    return utils.build_feed_dict(
        masked_tokens,
        self._tokenizer,
        target_tokens=target_tokens,
        max_seq_length=self._max_seq_length,
        max_predictions_per_seq=self._max_predictions_per_seq)
