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

"""Feature Pyramid Network layers in jax.

This is a JAX reimplementation of:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/architecture/fpn.py
"""

import functools
from typing import Dict

from flax import linen as nn
import jax.numpy as jnp

from findit import model_utils
from findit import spatial_transform_ops


Array = jnp.ndarray
LevelArray = Dict[int, Array]


class Fpn(nn.Module):
  """Feature pyramid network.

  Attributes:
    min_level: Minimum level in FPN output feature maps.
    max_level: Maximum level in FPN output feature maps.
    num_filters: Number of filters in FPN layers.
    train: Use training mode or not for batch normalization.
    use_batch_norm: Use batch_norm over output features.
    dtype: Data type of FPN.
    batch_norm_momentum: Batch normalization momentum value.
    batch_norm_epsilon: Batch normalization epsilon value.
    batch_norm_group_size: Distributed batch norm group size, which means how
      many examples within a batch will be used for batch stats computation. If
      zero, each device will use its own local data.
  """
  min_level: int = 2
  max_level: int = 6
  num_filters: int = 256
  train: bool = True
  use_batch_norm: bool = True
  dtype: jnp.dtype = jnp.float32
  batch_norm_momentum: float = 0.997
  batch_norm_epsilon: float = 1e-4
  batch_norm_group_size: int = 0

  @nn.compact
  def __call__(self,
               multilevel_feats):
    """Apply FPN block.

    Args:
      multilevel_feats: A dictionary of level features of shape [N, H, W, C].

    Returns:
      feats: A dictionary of level features of shape [N, H, W, C].
    """
    if self.train and self.batch_norm_group_size:
      axis_index_groups = model_utils.get_device_groups(
          self.batch_norm_group_size,
          multilevel_feats[self.min_level].shape[0])
    else:
      axis_index_groups = None
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not self.train,
        momentum=self.batch_norm_momentum,
        epsilon=self.batch_norm_epsilon,
        axis_name='batch' if self.batch_norm_group_size else None,
        axis_index_groups=axis_index_groups,
        dtype=self.dtype)

    input_levels = list(multilevel_feats.keys())
    backbone_max_level = min(max(input_levels), self.max_level)

    # Lateral convs
    feats_laterals = {}
    for level in range(self.min_level, backbone_max_level + 1):
      feats_laterals[level] = conv(self.num_filters, (1, 1))(
          multilevel_feats[level])

    # Adds top down path.
    feats = {backbone_max_level: feats_laterals[backbone_max_level]}
    for level in range(backbone_max_level - 1, self.min_level - 1, -1):
      feats[level] = spatial_transform_ops.nearest_upsampling(
          feats[level + 1], 2) + feats_laterals[level]

    # Adds post-hoc 3x3 convolution kernel.
    for level in range(self.min_level, backbone_max_level + 1):
      feats[level] = conv(self.num_filters, (3, 3))(feats[level])

    # Adds coarser FPN levels introduced for RetinaNet.
    for level in range(backbone_max_level + 1, self.max_level + 1):
      feats_in = feats[level - 1]
      if level > backbone_max_level + 1:
        feats_in = nn.relu(feats_in)
      feats[level] = conv(self.num_filters, (3, 3), (2, 2))(feats_in)

    if self.use_batch_norm:
      # Adds batch_norm layer.
      for level in range(self.min_level, self.max_level + 1):
        feats[level] = norm(name=f'p{level}-bn')(feats[level])

    return feats
