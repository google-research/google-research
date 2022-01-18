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

"""Common utility functions used across models."""

from typing import List

import numpy as np
import tensorflow as tf


def get_angles(position, embedding_dim):
  """Returns position based frequencies for positional encoding.

  Args:
    position: A list of token positions
    embedding_dim: Embedding size.

  Returns:
    angles: A list of angles where even elements (2i) and odd elements (2i+1)
      are pos/10000^{2i/embedding_dim}.
  """

  encoding_indices = np.arange(embedding_dim)[np.newaxis, :]
  angle_rates = 1 / np.power(
      10000, (2 * (encoding_indices // 2)) / np.float32(embedding_dim))
  return position * angle_rates


def positional_encoding(max_seq_size, embedding_dim):
  """Returns positional embedding.

  Args:
    max_seq_size: Maximum sequence size.
    embedding_dim: Dimension for positional encoding.

  Returns:
    pos_encoding: A tensor of shape [1, max_seq_size, embedding_dim], where the
      pos_encoding is the standard fixed cosine/sine of different frequencies
      for positional encodings. See Vaswani et al. "Attention Is All You Need"
      for details.
  """

  angle_rads = get_angles(np.arange(max_seq_size)[:, np.newaxis], embedding_dim)

  # Apply sin to even indices 2i in the array.
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # Apply cosine to odd 2i+1 indices in the array.
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, Ellipsis]

  return tf.cast(pos_encoding, dtype=tf.float32)
