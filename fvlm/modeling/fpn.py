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
from typing import Dict, Any

from flax import linen as nn
import gin
import jax.numpy as jnp

from ops import spatial_transform_ops
from utils import model_utils


Array = jnp.ndarray
LevelArray = Dict[int, Array]


@gin.register
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
  min_level: int = 3
  max_level: int = 7
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


@gin.register
class SimpleFpn(nn.Module):
  """Simple feature pyramid network.

  Reference: https://arxiv.org/pdf/2203.16527.pdf. Implementation follows
  Appendix A.2 of the paper.

  Attributes:
    min_level: Minimum level in FPN output feature maps.
    max_level: Maximum level in FPN output feature maps.
    num_filters: Number of filters in FPN layers.
    dtype: Data type of FPN.
    downsample_fn: Pooling function for down-sampling.
  """
  min_level: int
  max_level: int
  num_filters: int = 256
  dtype: jnp.dtype = jnp.float32
  downsample_fn: Any = nn.max_pool

  @nn.compact
  def __call__(self, single_level_feats):
    """Apply FPN block.

    Args:
      single_level_feats: A dictionary of single level features of shape
        [N, H, W, C]. The key is an integer of the level index.

    Returns:
      feats: A dictionary of multi-level features of shape [N, H, W, C].
    """
    if len(single_level_feats.keys()) != 1:
      raise ValueError('Must accept single level features. Got:',
                       single_level_feats.keys())
    level_key = list(single_level_feats.keys())[0]
    input_feat = single_level_feats[level_key]
    conv = functools.partial(
        nn.Conv,
        features=self.num_filters,
        use_bias=False,
        dtype=self.dtype,
        padding='SAME')
    deconv = functools.partial(
        nn.ConvTranspose,
        features=self.num_filters,
        use_bias=False,
        dtype=self.dtype,
        padding='SAME')
    norm = functools.partial(nn.LayerNorm, dtype=self.dtype)

    feats = {level_key: input_feat}
    # Pooling for down-sampling.
    for level in range(level_key + 1, self.max_level + 1):
      scale_size = 2 ** abs(level - level_key)
      scale_size = (scale_size, scale_size)
      feats[level] = self.downsample_fn(input_feat, window_shape=scale_size,
                                        strides=scale_size)

    # Transpose convolution for up-sampling.
    for level in range(self.min_level, level_key):
      diff = level_key - level
      x = input_feat
      for i in range(diff):
        x = deconv(kernel_size=(2, 2), strides=(2, 2))(x)
        x = norm(name=f'simfpn-l{level}-upconv-norm{i}')(x)
      feats[level] = x

    # Add post-hoc convolutions following FPN.
    for level in range(self.min_level, self.max_level + 1):
      x = conv(kernel_size=(1, 1))(feats[level])
      x = norm(name=f'simfpn-l{level}-out-norm1')(x)
      x = conv(kernel_size=(3, 3))(x)
      x = norm(name=f'simfpn-l{level}-out-norm2')(x)
      feats[level] = x

    return feats


@gin.register
class SimpleFpnV2(nn.Module):
  """Simple feature pyramid network V2.

  Reference: https://arxiv.org/pdf/2203.16527.pdf. Implementation follows
  the ViTDet github release:
  https://github.com/facebookresearch/detectron2/blob/333efcb6d0b60d7cceb7afc91bd96315cf211b0a/detectron2/modeling/backbone/vit.py#L361
  https://github.com/facebookresearch/detectron2/blob/333efcb6d0b60d7cceb7afc91bd96315cf211b0a/configs/common/models/mask_rcnn_vitdet.py

  Attributes:
    min_level: Minimum level in FPN output feature maps.
    max_level: Maximum level in FPN output feature maps.
    num_filters: Number of filters in FPN layers.
    dtype: Data type of FPN.
    downsample_fn: Pooling function for down-sampling.
  """
  min_level: int
  max_level: int
  num_filters: int = 256
  dtype: jnp.dtype = jnp.float32
  downsample_fn: Any = nn.max_pool

  @nn.compact
  def __call__(self, single_level_feats):
    """Apply Simple FPN block.

    Args:
      single_level_feats: A dictionary of single level features of shape
        [N, H, W, C]. The key is an integer of the level index.

    Returns:
      feats: A dictionary of multi-level features of shape [N, H, W, C].
    """
    if len(single_level_feats.keys()) != 1:
      raise ValueError('Must accept single level features. Got:',
                       single_level_feats.keys())
    level_key = list(single_level_feats.keys())[0]
    if max(self.max_level - level_key, level_key - self.min_level) > 2:
      raise ValueError('Pyramid level difference must be <= 2.')

    input_feat = single_level_feats[level_key]
    conv = functools.partial(
        nn.Conv,
        features=self.num_filters,
        use_bias=False,
        dtype=self.dtype,
        padding='SAME')
    deconv = functools.partial(
        nn.ConvTranspose,
        use_bias=False,
        dtype=self.dtype,
        padding='SAME')
    norm = functools.partial(nn.LayerNorm, dtype=self.dtype)

    feats = {level_key: input_feat}
    in_feat_dim = input_feat.shape[-1]

    for level in range(self.min_level, self.max_level + 1):
      diff = level_key - level
      x = input_feat
      if diff == 2:
        x = deconv(features=in_feat_dim // 2,
                   kernel_size=(2, 2),
                   strides=(2, 2))(x)
        x = norm(name=f'simfpn-l{level}-ln-d2')(x)
        x = nn.gelu(x, approximate=False)  # torch.nn.GELU no approx by default.
        x = deconv(features=in_feat_dim // 4,
                   kernel_size=(2, 2),
                   strides=(2, 2))(x)

      elif diff == 1:
        x = deconv(features=in_feat_dim // 2,
                   kernel_size=(2, 2),
                   strides=(2, 2))(x)
      elif diff < 0:
        # diff in (-1, -2).
        x = self.downsample_fn(x, window_shape=(2, 2), strides=(2, 2))

      x = conv(kernel_size=(1, 1))(x)
      x = norm(name=f'simfpn-l{level}-ln-out-1')(x)
      x = conv(kernel_size=(3, 3))(x)
      x = norm(name=f'simfpn-l{level}-ln-out-2')(x)

      # Handles top_block following the ViTDet implementation.
      # https://github.com/facebookresearch/detectron2/blob/333efcb6d0b60d7cceb7afc91bd96315cf211b0a/detectron2/modeling/backbone/fpn.py#L188
      if diff == -2:
        x = self.downsample_fn(x, window_shape=(1, 1), strides=(2, 2))

      feats[level] = x

    return feats
