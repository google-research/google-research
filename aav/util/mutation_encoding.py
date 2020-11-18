# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for encoding sequence mutations as feature vectors.

A mutation mask is a relative sequence, given some reference base sequence.
Within a mutation mask or sequence, tokens (all single characters currently)
have the following meaning:
  "_": wild type (non-mutated) position
  "<upper case letter>": single residue substitution
  "<lower case letter>": single residue insertion

For example, given the 4-residue WT sequence, "TEST", then a single point
substitution can be encoded as a mutation mask: "A___", "_A__", "__A_", or
"___A" depending on the location of the substitution within the WT seq.

Additionally, a single prefix insertion is allowed; all other insertions occur
following the WT residue position. For example, an alanine prefix insertion
for the "TEST" (WT) sequence would be encoded as "aTEST" while a suffix
insertion of alanine would be encoded as "TESTa".
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy


# Sentinel token value for denoting "no mutation here".
_PLACEHOLDER_TOKEN = '_'
# Number of different mutation slots to use for each wildtype sequence position.
_NUM_MUTATION_SLOTS = 2
# Slot index for substitution mutations.
_SUB_INDEX = 0
# Slot index for insertion mutations.
_INS_INDEX = 1


def tokenize_mutation_seq(seq, placeholder_token='_'):
  """Converts a variable-length mutation sequence to a fixed-length sequence.

  For an N-residue reference sequence, the encoding is shape (N+1, M, A), where
  A is the alphabet size (e.g., A=20 for the canonical peptide alphabet) and M
  is the number of distinct mutation types at each position; here, M=2
  (1x sub + 1x ins at each reference sequence position).

  Args:
    seq: (str) A mutation sequence to tokenize; e.g., "__A_" or "aTEST".
    placeholder_token: (str) Sentinel value used to encode non-mutated positions
      in the mutation sequence.
  Returns:
    A length-N+1 sequence of ("<substitution_token>", "<insertion token>")
    2-tuples.
  """
  tokens = []
  i = 0
  # Consume the prefix insertion mutation if there is one.
  # A prefix insertion is denoted by a leading lower case letter on the seq.
  if seq[i].islower():
    tokens.append((placeholder_token, seq[i].upper()))
    i += 1
  else:
    tokens.append((placeholder_token, placeholder_token))

  while i < len(seq):
    if i < len(seq) - 1 and seq[i + 1].islower():
      tokens.append((seq[i], seq[i+1].upper()))
      i += 2
    else:
      tokens.append((seq[i], placeholder_token))
      i += 1
  return tokens


class DirectSequenceEncoder(object):
  """Direct sequence encoder for generating variable-length representations.

  Attributes:
    encoding_size: (int) The encoding length for a single residue.
  """

  def __init__(self, residue_encoder):
    """Constructor.

    Args:
      residue_encoder: (object) A single residue encoder.
    """
    self._residue_encoder = residue_encoder
    self.encoding_size = self._residue_encoder.encoding_size

  def encode(self, seq):
    """Encodes a sequence as a variable-length multi-dimensional array.

    Args:
      seq: (str) A residue sequence to encode; e.g., "ATEST".
    Returns:
      (numpy.ndarray(shape=(len(seq), encoding_size), dtype=float)) The encoded
      residue sequence.
    """
    return numpy.array([self._residue_encoder.encode(r) for r in seq.upper()])


class MutationSequenceEncoder(object):
  """Mutation sequence encoder for generating fixed-length representations.

  The encoding has two slots for each residue position in the ref sequence:
    1. A slot that encodes a residue substitution mutation
    2. A slot that encodes a single-residue insertion mutation

  There is also a pair of slots for any single-position prefix mutation.

  Attributes:
    encoding_size: (int) The encoding length for a single residue.
  """

  def __init__(self, residue_encoder, ref_seq):
    """Constructor.

    Args:
      residue_encoder: (object) A single residue encoder
      ref_seq: (str) The reference (non-mutated) sequence.
    """
    self._residue_encoder = residue_encoder
    self._ref_seq = ref_seq
    self.encoding_size = self._residue_encoder.encoding_size

  def encode(self, seq):
    """Encodes a mutation sequence as a fixed-length multi-dimensional array.

    Args:
      seq: (str) A mutation sequence to encode; e.g., "__A_".
    Returns:
      A numpy.ndarray(shape=(len(ref_seq)+1, 2, encoding_size), dtype=float).
    Raises:
      ValueError: if the mutation sequence references a different number of
        sequence positions than the specified ref_seq.
    """
    seq_encoding = numpy.zeros((
        len(self._ref_seq) + 1,
        _NUM_MUTATION_SLOTS,
        self.encoding_size))

    sub_ins_tokens = tokenize_mutation_seq(seq, _PLACEHOLDER_TOKEN)
    if len(sub_ins_tokens) != len(self._ref_seq) + 1:
      raise ValueError('Mutation sequence dimension mismatch: '
                       '%d mutation positions vs %d in reference sequence'
                       % (len(sub_ins_tokens), len(self._ref_seq) + 1))

    for position_i, (sub_token, ins_token) in enumerate(sub_ins_tokens):
      if sub_token != _PLACEHOLDER_TOKEN:
        seq_encoding[position_i, _SUB_INDEX, :] = self._residue_encoder.encode(
            sub_token)
      if ins_token != _PLACEHOLDER_TOKEN:
        seq_encoding[position_i, _INS_INDEX, :] = self._residue_encoder.encode(
            ins_token)
    return seq_encoding

