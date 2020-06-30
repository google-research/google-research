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

"""Utils for preprocessing sequences."""

import numpy as np
import tensorflow.compat.v1 as tf


AA_TOKENS = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
             'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
# Includes list of anomalous amino acids based on IUPAC:
# http://publications.iupac.org/pac/1984/pdf/5605x0595.pdf
AA_ANOMALOUS_TOKENS = ('B', 'O', 'U', 'X', 'Z')
AA_ALIGN_TOKENS = ('.', '-')


def pad_sequences(sequences, length=None, value=None):
  """Pads integer sequences to a certain length.

  Args:
    sequences: A list of variable-length and integer-encoded sequences.
    length: The padding length. If `None`, the maximum sequences length is used.
    value: The padding value. If `None`, the maximum of `sequences` plus one
      will be used.

  Returns:
    A `np.ndarray` with shape `(len(sequences), length)` that represents the
    encoded sequences.
  """
  if value is None:
    value = max(np.max(seq) for seq in sequences) + 1
  if length is None:
    length = max(len(seq) for seq in sequences)
  return tf.keras.preprocessing.sequence.pad_sequences(
      sequences, maxlen=length, padding='post', value=value)


def unpad_sequences(sequences, value):
  """Unpads sequences.

  Note that all occurrences of `value` in `sequences` are removed, also if
  they are not at the end of the sequence.

  Args:
    sequences: A matrix of padded integer-encoded sequences.
    value: The padding value.

  Returns:
    A list of sequences without `value`.
  """
  sequences = [np.asarray(seq) for seq in sequences]
  return [seq[seq != value] for seq in sequences]


def sequences_end_with_value(sequences, value, axis=-1):
  """Tests if `sequences` and with `value` along `axis`.

  Args:
    sequences: A matrix of integer-encoded sequences.
    value: An integer value.
    axis: Axis of `sequences` to test.

  Returns:
    A boolean `np.nadarray` that indicates for each sequences if it ends with
    `value` along `axis`.
  """
  sequences = np.asarray(sequences)
  return np.all(np.diff((sequences == value).astype(np.int8), axis=axis) >= 0,
                axis)


def strip_sequences_at_value(sequences, value):
  """Sets all values after the first occurrence of `value` to value`.

  For example:
    sequences = [[0, 1, 2], [0, 1, 0, 1]]
    value = 1
    Returns: [[0, 1, 1], [0, 1, 1, 1]]

  Args:
    sequences: An iterable of integer-encoded sequences, possibly of different
      length.
    value: An integer value.

  Returns:
    A list of sequences with all values set to `value` after the first
    occurrence of `value`.
  """
  stripped = []
  for seq in sequences:
    seq = np.array(seq)
    idx = np.where(seq == value)[0]
    if idx.size:
      seq[idx[0]:] = value
    stripped.append(seq)
  return stripped
