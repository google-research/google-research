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

"""Helper class for patching and unpatching."""

import math
from typing import NamedTuple, Optional, Tuple

import tensorflow as tf

from vct.src import extract_patches


class Patched(NamedTuple):
  """Represents a patched tensor.

  Attributes:
    tensor: The patched tensor, shape (b', patch_size ** 2, d)
    num_patches: Tuple (n_h, n_w) indicating how many patches are in b'.
  """

  tensor: tf.Tensor
  num_patches: Tuple[int, int]


class Patcher:
  """Helper class for patching and unpatching."""

  def __init__(
      self,
      stride,
      pad_mode = "REFLECT",
  ):
    """Initializes the patch helper."""
    self.stride = stride
    self.pad_mode = pad_mode

  def _pad(self, x, patch_size):
    """Pads `x` such that we can do VALID patch extraction."""
    if patch_size < self.stride:
      raise ValueError("`patch_size` must be greater than `stride`!")
    # Additionally pad to handle patch_size > stride.
    missing = patch_size - self.stride
    if missing % 2 != 0:
      raise ValueError("Can only handle even missing pixels.")

    _, height, width, _ = x.shape
    (n_h, n_w), (height_padded, width_padded) = self.get_num_patches(
        height, width)

    return tf.pad(x, [
        [0, 0],
        [missing // 2, height_padded - height + missing // 2],
        [missing // 2, width_padded - width + missing // 2],
        [0, 0],
    ], self.pad_mode), n_h, n_w

  def get_num_patches(
      self, height, width):
    # Initial pad to get all strides in.
    height_padded = math.ceil(height / self.stride) * self.stride
    width_padded = math.ceil(width / self.stride) * self.stride
    # Calculate number of patches in the height and width dimensions.
    n_h = height_padded // self.stride
    n_w = width_padded // self.stride
    return (n_h, n_w), (height_padded, width_padded)

  def __call__(self, t, patch_size):
    """Pads and extracts patches, shape (b * num_patches, size ** 2, d)."""
    # First pad such that we can use `extract_patches` with padding=VALID, i.e.,
    # first patch should cover top left part.
    t_padded, n_h, n_w = self._pad(t, patch_size)
    patches = extract_patches.extract_patches(t_padded, patch_size, self.stride)
    # `extract_patches` returns (b, n_h, n_w, seq_len * d), we reshape this
    # to (..., seq_len, d).
    b, n_hp, n_wp, _ = patches.shape
    d = t_padded.shape[-1]
    assert (n_hp, n_wp) == (n_h, n_w)  # Programmer error.
    patches = tf.reshape(patches, (b * n_h * n_w, patch_size ** 2, d))
    return Patched(patches, (n_h, n_w))

  def unpatch(self, t, n_h, n_w,
              crop):
    """Goes back to (b, h, w, d)."""
    _, seq_len, d = t.shape
    assert seq_len == self.stride ** 2
    t = tf.reshape(t, (-1, n_h, n_w, self.stride, self.stride, d))
    t = tf.einsum("bijhwc->bihjwc", t)
    t = tf.reshape(t, (-1, n_h * self.stride, n_w * self.stride, d))
    if crop:
      h, w = crop
      return t[:, :h, :w, :]
    else:
      return t
