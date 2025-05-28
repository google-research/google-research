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

"""Class for enumerative sequential synthesizer."""

import copy as c
import random
from typing import Callable, Iterator, Optional, Sequence, List, TypeVar, Dict, Set, Any

from abstract_nas.model.concrete import Op
from abstract_nas.synthesis.enum_sequential import EnumerativeSequentialSynthesizer

T = TypeVar("T")


def num_in_base(num, base):
  """Converts an integer to its representation in base in big-endian.

  e.g., num_in_base(14, 2) = [1, 1, 1, 0].

  Args:
    num: the number to convert.
    base: the base of the representation.

  Returns:
    The list of digits in big-endian.
  """

  if num == 0:
    return [0]
  digits = []
  while num:
    digits.append(int(num % base))
    num //= base
  return digits[::-1]


def random_sequence_generator(gen,
                              max_len,
                              min_len = 0,
                              copy = True):
  """A random sequence generator."""
  primitives = list(gen())
  num_primitives = len(primitives)
  if copy:
    maybe_copy = c.deepcopy
  else:
    maybe_copy = lambda x: x

  for cur_len in range(min_len, max_len + 1):
    num_seqs = num_primitives ** cur_len
    all_idxs = list(range(num_seqs))
    random.shuffle(all_idxs)

    for seq_idx in all_idxs:
      seq_idxs = num_in_base(seq_idx, num_primitives)
      seq_idxs = [0] * (cur_len - len(seq_idxs)) + seq_idxs
      yield [maybe_copy(primitives[seq_idx]) for seq_idx in seq_idxs]
  return


class RandomEnumerativeSequentialSynthesizer(EnumerativeSequentialSynthesizer):
  """Synthesizer that randomly enumerates through sequential subgraphs in length order."""

  @classmethod
  def subg_enumerator(
      cls,
      prefix = None,
      max_len = -1,
      min_len = 0,
      kwarg_defaults = None,
      full = True
  ):

    return random_sequence_generator(
        lambda: cls.op_enumerator(prefix, kwarg_defaults, full), max_len,
        min_len)
