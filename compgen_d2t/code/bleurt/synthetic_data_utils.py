# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utils for generating synthetic negatives for BLEURT."""

import copy
import random
import re
from typing import List
from typing import Tuple


class NegativeSampler:
  """Negative sampler for different kinds of negative creation."""

  def __init__(self, num_deletion_negatives, min_num_deletions,
               max_num_deletions, num_repetition_negatives,
               min_num_repetitions, max_num_repetitions,
               num_flip_negatives, num_random_negatives,
               num_digit_negatives, use_source_as_reference):
    """Initialization of negative sampler.

    Args:
      num_deletion_negatives: Number of negatives via token deletion from the
        target.
      min_num_deletions: The minimum number of tokens can be deleted.
      max_num_deletions: The maximum number of tokens can be deleted.
      num_repetition_negatives: Number of negatives via repeating tokens in the
        target.
      min_num_repetitions: The minimum number of tokens for repeating.
      max_num_repetitions: The maximum number of tokens for repeating.
      num_flip_negatives: Number of negatives via flipping tokens in target.
      num_random_negatives: Number of negatives via randomly matching source and
        target.
      num_digit_negatives: Number of negatives via corrupting digits in the
        target.
      use_source_as_reference: Whether to use the source text as reference.
    """
    self.num_deletion_negatives = num_deletion_negatives
    self.min_num_deletions = min_num_deletions
    self.max_num_deletions = max_num_deletions
    assert self.min_num_deletions <= self.max_num_deletions
    self.num_repetition_negatives = num_repetition_negatives
    self.min_num_repetitions = min_num_repetitions
    self.max_num_repetitions = max_num_repetitions
    assert self.min_num_repetitions <= self.max_num_repetitions
    self.num_flip_negatives = num_flip_negatives
    self.num_random_negatives = num_random_negatives
    self.num_digit_negatives = num_digit_negatives
    self.use_source_as_reference = use_source_as_reference

  def _tokenize(self, text):
    return re.split("\\s+", text)

  def _detokenize(self, tokens):
    return " ".join(tokens)

  def get_negatives(self, source, target):
    """Get all types of negatives."""
    negatives = []
    for _ in range(self.num_deletion_negatives):
      negatives.append(self.get_deletion_negative(source, target))
    for _ in range(self.num_repetition_negatives):
      negatives.append(self.get_repetition_negative(source, target))
    for _ in range(self.num_flip_negatives):
      negatives.append(self.get_flip_negative(source, target))
    for _ in range(self.num_digit_negatives):
      negatives.append(self.get_digit_negative(source, target))
    return [neg for neg in negatives if neg]

  def get_deletion_negative(self, source, target):
    """Get negative by deleting random phrases."""
    target_tokens = self._tokenize(target)
    trial, max_trial = 0, 10
    while trial < max_trial:
      trial += 1
      num_deletions = random.randint(self.min_num_deletions,
                                     self.max_num_deletions)
      if num_deletions >= len(target_tokens):
        continue
      span_start = random.randint(0, len(target_tokens) - num_deletions - 1)
      span_end = span_start + num_deletions
      negative_tokens = target_tokens[:span_start] + target_tokens[span_end:]
      negative_target = self._detokenize(negative_tokens)

      if source == negative_target:
        continue

      return (source, negative_target) if self.use_source_as_reference else (
          target, negative_target)

  def get_repetition_negative(self, source,
                              target):
    """Get negatives by repeating phrases."""
    target_tokens = self._tokenize(target)
    trial, max_trial = 0, 10
    while trial < max_trial:
      trial += 1
      num_repetitions = random.randint(self.min_num_repetitions,
                                       self.max_num_repetitions)
      if num_repetitions >= len(target_tokens):
        continue
      span_start = random.randint(0, len(target_tokens) - num_repetitions)
      span_end = span_start + num_repetitions
      negative_tokens = (
          target_tokens[:span_end] + target_tokens[span_start:span_end] +
          target_tokens[span_end:])
      negative_target = self._detokenize(negative_tokens)
      if source == negative_target:
        continue

      return (source, negative_target) if self.use_source_as_reference else (
          target, negative_target)

  def get_flip_negative(self, source, target):
    """Get negatives by flipping words."""
    target_tokens = self._tokenize(target)
    trial, max_trial = 0, 10
    while trial < max_trial:
      trial += 1
      index1 = random.randint(0, len(target_tokens) - 1)
      index2 = random.randint(0, len(target_tokens) - 1)
      if index1 != index2 and target_tokens[index1] != target_tokens[index2]:
        negative_tokens = copy.deepcopy(target_tokens)
        negative_tokens[index1] = target_tokens[index2]
        negative_tokens[index2] = target_tokens[index1]

        negative_target = self._detokenize(negative_tokens)
        if source == negative_target:
          continue
        return (source, negative_target) if self.use_source_as_reference else (
            target, negative_target)

  def get_identity_negative(self, source, target):
    """Pair source with itself as a negative (if use_source_as_reference=True)."""
    if self.use_source_as_reference:
      assert source != target
      return (source, source)
    else:
      return ()

  def get_digit_negative(self, source, target):
    """Generate negatives by flipping random digits."""
    digit_positions = [
        index for index, val in enumerate(target)
        if (val.isdigit() and ("0" <= val <= "9"))
    ]
    if not digit_positions:
      return []

    # Pick a random position to flip.
    pos = random.randint(0, len(digit_positions) - 1)
    target_pos = digit_positions[pos]

    x = random.randint(0, 9)
    orig_digit = int(target[target_pos])
    while x == orig_digit:
      x = random.randint(0, 9)

    negative_target = target[0:target_pos] + str(x) + target[(target_pos + 1):]
    if self.use_source_as_reference:
      assert source != negative_target
      return (source, negative_target)
    else:
      return (target, negative_target)

  def get_random_negatives(self, source_target_pairs,
                           index):
    """Generate negatives by pairing source with random targets."""
    random_negatives = []
    source = source_target_pairs[index][0]
    target = source_target_pairs[index][1]
    while len(random_negatives) < self.num_random_negatives:
      negative_index = random.randint(0, len(source_target_pairs) - 1)
      if negative_index != index:
        negative_target = source_target_pairs[negative_index][1]
        if target != negative_target:
          assert source != negative_target
          if self.use_source_as_reference:
            random_negatives.append((source, negative_target))
          else:
            random_negatives.append((target, negative_target))

    return random_negatives
