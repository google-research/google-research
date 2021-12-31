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

"""TPU friendly convolution."""

import functools
from typing import Any, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp


def split4x4(image, total_pad):
  """Pad and split img in a 4x4 grid."""
  assert total_pad % 2 == 0
  num_splits_hw = 4
  N, H, W, C = image.shape
  # pad to multiples of 4
  pad_h = (num_splits_hw - H % num_splits_hw) % num_splits_hw
  pad_w = (num_splits_hw - W % num_splits_hw) % num_splits_hw
  image = jnp.pad(image, ((0, 0), (total_pad // 2, pad_h + total_pad // 2),
                          (total_pad // 2, pad_w + total_pad // 2), (0, 0)))
  partition_size_h = (H + pad_h) // num_splits_hw
  partition_size_w = (W + pad_w) // num_splits_hw

  # find all slices
  slices = [[[] for _ in range(num_splits_hw)] for _ in range(num_splits_hw)]
  for i in range(num_splits_hw):
    for j in range(num_splits_hw):
      slices[i][j] = ([
          slice(i * (partition_size_h),
                total_pad + (i + 1) * (partition_size_h)),
          slice(j * (partition_size_w),
                total_pad + (j + 1) * (partition_size_w))
      ])
  out = jnp.stack([
      jnp.stack([image[:, s[0], s[1], :]
                 for s in row], axis=1)
      for row in slices
  ],
                  axis=1)
  # merge first 3 dimensions
  # TODO(machc): this is expensive!
  out = out.reshape((-1,) + out.shape[3:])

  return out, pad_h, pad_w


def reconstruct4x4(image, pad_h, pad_w, total_pad):
  """Reverse operation of split4x4."""
  # split first three dimensions
  # TODO(machc): this is expensive!
  reconstructed = image.reshape((-1, 4, 4) + image.shape[1:])
  # this is not needed when using conv paddings = valid.
  # reconstructed = reconstructed[..., total_pad//2:-total_pad//2, total_pad//2:-total_pad//2, :]
  # WARNING: we can probably be clever about ordering and transposes here and do this in one step
  reconstructed = jnp.concatenate(jnp.moveaxis(reconstructed, 1, 0), axis=2)
  reconstructed = jnp.concatenate(jnp.moveaxis(reconstructed, 1, 0), axis=2)
  # remove padding to mult of 4
  return reconstructed[:, :-pad_h or None, :-pad_w or None]


class SplitConvModel(nn.Module):
  """Sequence of Conv layers but splitting the input image for TPU speedup."""
  features: Sequence[int]
  kernel_size: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    num_layers = len(self.features)
    total_pad = (self.kernel_size - 1) * num_layers
    to_concat = []
    x, pad_h, pad_w = split4x4(x, total_pad=total_pad)
    for i, f in enumerate(self.features):
      x = nn.Conv(
          features=f,
          kernel_size=(self.kernel_size, self.kernel_size),
          padding='valid',
          dtype=self.dtype)(
              x)
      x = nn.relu(x)
      # undo extra padding to ensure dimensions match
      pad = (total_pad - (i + 1) * total_pad // num_layers) // 2
      to_concat.append(x[:, pad:-pad or None, pad:-pad or None, :])
    x = jnp.concatenate(to_concat, axis=-1)
    return reconstruct4x4(x, pad_h, pad_w, total_pad)
