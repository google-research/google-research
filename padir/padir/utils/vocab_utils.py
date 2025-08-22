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

"""Vocab utils."""

import seqio
from t5.data import preprocessors as t5_preprocessors


def get_mask_id(vocab):
  return t5_preprocessors.sentinel_id(vocab)  # To match t5 denoising code.


def get_mask_id_range(
    vocab, max_target_len
):
  """Returns the range of mask ids based on max_target_len.

  Args:
    vocab: the output vocabulary.
    max_target_len: max decoder length.

  Returns:
    A tuple with [first_mask_id, last_mask_id)
  """
  mask_id = get_mask_id(vocab)
  return mask_id - max_target_len, mask_id


def get_bos_id(vocab):
  """Returns the BOS id sometimes prepended to the target tokens."""
  return t5_preprocessors.sentinel_id(vocab) - 1


def get_length_id(vocab):
  """Returns the LENGTH id sometimes prepended to the source tokens."""
  return t5_preprocessors.sentinel_id(vocab) - 2
