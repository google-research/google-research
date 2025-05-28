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

"""Classes to build various prediction heads in JAX detection models.

The implementation follows cloud TPU detection heads
https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/architecture/heads.py
"""

import functools
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from flax import linen as nn
from flax.linen import initializers
import gin
import jax.numpy as jnp

from ops import normalized_activation_layers
from utils import model_utils

Array = jnp.ndarray
NormalizedDense = normalized_activation_layers.NormalizedDense
NormalizedConv = normalized_activation_layers.NormalizedConv


@gin.register
class RpnHead(nn.Module):
  """Region Proposal Network head.

  Attributes:
    min_level: Number of minimum feature level.
    max_level: Number of maximum feature level.
    strides: an int tuple of stride used in convolutions.
    num_convs: Number of the intermediate
      conv layers before the prediction.
    num_filters: Number of filters of the
      intermediate conv layers.
    anchors_per_location: Number of anchors per pixel
      location.
    train: To use training mode or not in Batch Normalization.
    dtype: type of variables used.
    batch_norm_group_size: The batch norm group size.
    use_batch_norm: Use batch norm or layer norm.
  """
  min_level: int = 3
  max_level: int = 7
  strides: Sequence[int] = (1, 1)
  num_convs: int = 1
  num_filters: int = 256
  anchors_per_location: int = 3
  train: bool = True
  dtype: jnp.dtype = jnp.float32
  batch_norm_group_size: int = 0
  use_batch_norm: bool = True

  @nn.compact
  def __call__(
      self, fpn_features
      ):
    """Region Proposal Network head.

    Args:
      fpn_features: a dictionary with integers keys mapping to feature array,
        where the array has shape [batch_size, feature_height, feature_width,
        feature_dim].

    Returns:
      A 2-tuple of (score_outputs, box_outputs) where,
      score_outputs: a dictionary with integers keys mapping to score array,
        where the array has shape [batch_size, feature_height, feature_width,
        anchors_per_location].
      box_outputs: a dictionary with integers keys mapping to box output array,
        where the array has shape [batch_size, feature_height, feature_width,
        anchors_per_location].
    """
    batch_sizes = [v.shape[0] for k, v in fpn_features.items()]
    if batch_sizes.count(batch_sizes[0]) == len(batch_sizes):
      batch_size = batch_sizes[0]
    else:
      raise ValueError('Inconsistent batch size in RPN inputs!')

    # Since the TF1 RPN implementation share weights over the levels,
    # initialize the convolution functions in RPN and pass them to SubRPN.
    conv_fns = []
    for i in range(self.num_convs):
      conv_fns.append(
          nn.Conv(
              features=self.num_filters,
              kernel_size=(3, 3),
              padding='SAME',
              name='rpn' if i == 0 else f'rpn-{i}',
              strides=self.strides,
              dtype=self.dtype,
              kernel_init=initializers.normal(stddev=1e-2),
              bias_init=initializers.zeros))

    score_conv_fn = nn.Conv(
        features=self.anchors_per_location,
        kernel_size=(1, 1),
        padding='VALID',
        name='rpn-class',
        strides=self.strides,
        # Set to float32 as a temporary fix to avoid numerical instability.
        dtype=jnp.float32,
        kernel_init=initializers.normal(stddev=1e-2),
        bias_init=initializers.zeros)
    box_conv_fn = nn.Conv(
        features=4 * self.anchors_per_location,
        kernel_size=(1, 1),
        padding='VALID',
        name='rpn-box',
        strides=self.strides,
        dtype=jnp.float32,
        kernel_init=initializers.normal(stddev=1e-2),
        bias_init=initializers.zeros)

    # Set momentum and epsilon values to match TF1 implementation.
    if self.use_batch_norm:
      norm_fn = functools.partial(
          nn.BatchNorm,
          use_running_average=not self.train,
          momentum=0.997,
          epsilon=1e-4,
          axis_name='batch' if self.batch_norm_group_size else None,
          axis_index_groups=model_utils.get_device_groups(
              self.batch_norm_group_size, batch_size)
          if self.train and self.batch_norm_group_size else None,
          dtype=self.dtype)
    else:
      norm_fn = functools.partial(nn.LayerNorm, dtype=self.dtype)

    scores_outputs = {}
    box_outputs = {}
    for level in range(self.min_level, self.max_level + 1):
      scores_outputs[level], box_outputs[level] = SubRpnHead(
          norm_fn=norm_fn,
          conv_fns=conv_fns,
          score_conv_fn=score_conv_fn,
          box_conv_fn=box_conv_fn,
          level=level)(fpn_features[level])

    return scores_outputs, box_outputs


@gin.register
class SubRpnHead(nn.Module):
  """Region Proposal Network head for each level.

  Attributes:
    conv_fns: a list of convolution function handle.
    score_conv_fn: scoring output convolution function handle.
    box_conv_fn: box output convolution function handle.
    level: Number that represents the level of RPN feature map.
    norm_fn: batch normalization function handle.
    activation_fn: activation function handle.
  """
  conv_fns: Sequence[Any]
  score_conv_fn: Any
  box_conv_fn: Any
  level: int = 3
  norm_fn: Callable[[Array], Array] = nn.BatchNorm
  activation_fn: Callable[[Array], Array] = nn.relu

  @nn.compact
  def __call__(self, level_feature):
    """Per-level RPN heads.

    Args:
      level_feature: per-level feature map of shape [batch_size, feature_height,
        feature_width, feature_dim].

    Returns:
      scores: a score array of shape [batch_size, feature_height, feature_width,
        anchors_per_location].
      bboxes: a box output array of shape [batch_size, feature_height,
        feature_width, anchors_per_location].
    """
    x = level_feature
    for i, conv_fn in enumerate(self.conv_fns):
      x = conv_fn(x)
      x = self.norm_fn(name=(
          f'rpn-l{self.level}-bn') + ('' if i == 0 else f'-{i}'))(x)
      x = self.activation_fn(x)

    # Proposal classification scores
    scores = self.score_conv_fn(x)
    # Proposal bbox regression deltas
    bboxes = self.box_conv_fn(x)
    return scores, bboxes


@gin.register
class FastrcnnHead(nn.Module):
  """Fast R-CNN box head.

  Attributes:
    num_classes: an integer for the number of classes or class output
      feature dimension.
    num_convs: Number of the intermediate conv layers before the FC layers.
    num_filters: Number of filters of the intermediate conv layers.
    num_fcs: Number of FC layers before the predictions.
    fc_dims: Number of dimension of the FC layers.
    class_box_regression: Whether to use class-specific box regression or not.
    train: `bool` of using training mode or not in Batch Normalization.
    dtype: type of variables used.
    batch_norm_group_size: The batch norm group size.
    use_class_bias: Whether to use bias for class outputs prediction.
    use_batch_norm: Use batch norm or layer norm.
    use_norm_classifier: Whether to use normalized activation in classifier
      output layer. This is to normalize the weights and input features to
      reduce the scale variance for different categories.
  """
  num_classes: int = 2
  num_convs: int = 0
  num_filters: int = 256
  num_fcs: int = 2
  fc_dims: int = 1024
  class_box_regression: bool = True
  train: bool = True
  dtype: jnp.dtype = jnp.float32
  batch_norm_group_size: int = 0
  activation_fn: Callable[[Array], Array] = nn.relu
  use_class_bias: bool = True
  use_batch_norm: bool = True
  use_norm_classifier: bool = False

  @nn.compact
  def __call__(self, roi_features):
    """Box and class branches for the Faster/Mask-RCNN model.

    Args:
      roi_features: A ROI feature array of shape
        [batch_size, num_rois, height_l, width_l, feature_dim].

    Returns:
      class_outputs: an array with a shape of
        [batch_size, num_rois, num_classes], representing the class
        predictions.
      box_outputs: an array with a shape of
        [batch_size, num_rois, num_classes * 4], representing the box
        predictions, or an array of shape [batch_size, num_rois, 4] for
        class-agnostic boxes.
    """
    batch_size, num_rois, height, width, feature_dim = roi_features.shape
    # Use variance scaling kernel initializer to match TF1 implementation.
    conv_fn = functools.partial(
        nn.Conv,
        features=self.num_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        dtype=self.dtype,
        kernel_init=initializers.variance_scaling(
            scale=2, mode='fan_out', distribution='normal'),
        bias_init=initializers.zeros,
        padding='SAME')
    # Set momentum and epsilon values to match TF1 implementation.
    if self.use_batch_norm:
      norm_fn = functools.partial(
          nn.BatchNorm,
          use_running_average=not self.train,
          momentum=0.997,
          epsilon=1e-4,
          axis_name='batch' if self.batch_norm_group_size else None,
          axis_index_groups=model_utils.get_device_groups(
              self.batch_norm_group_size, batch_size)
          if self.train and self.batch_norm_group_size else None,
          dtype=self.dtype)
    else:
      norm_fn = functools.partial(nn.LayerNorm, dtype=self.dtype)
    dense_layer = NormalizedDense if self.use_norm_classifier else nn.Dense

    # reshape inputs before FC.
    x = roi_features.reshape([-1, height, width, feature_dim])
    for i in range(self.num_convs):
      x = conv_fn(name=f'conv_{i}')(x)
      x = norm_fn(name=f'conv_{i}_bn')(x)
      x = self.activation_fn(x)

    feature_dim = self.num_filters if self.num_convs > 0 else feature_dim
    x = x.reshape([batch_size, num_rois, height * width * feature_dim])
    for i in range(self.num_fcs):
      # Use Xavier uniform to initialize following TF1 implementation.
      x = nn.Dense(
          features=self.fc_dims,
          name=f'fc{i+6}',
          kernel_init=initializers.xavier_uniform(),
          dtype=self.dtype,
      )(x)
      x = norm_fn(name=f'fc{i+6}_bn')(x)
      x = self.activation_fn(x)

    num_box_outputs = self.num_classes * 4 if self.class_box_regression else 4
    box_outputs = nn.Dense(
        features=num_box_outputs,
        name='box-predict',
        kernel_init=initializers.normal(stddev=0.001),
        # Set to float32 as a temporary fix to avoid numerical instability.
        dtype=jnp.float32,
    )(x)
    class_outputs = dense_layer(
        features=self.num_classes,
        name='class-predict',
        kernel_init=initializers.normal(stddev=0.01),
        # Set to float32 as a temporary fix to avoid numerical instability.
        dtype=jnp.float32,
        use_bias=self.use_class_bias,
    )(x)

    return class_outputs, box_outputs


@gin.register
class MaskrcnnHead(nn.Module):
  """Mask R-CNN mask head.

  Attributes:
    num_classes: an integer for the number of classes.
    num_convs: Number of the intermediate conv layers before the FC layers.
    num_filters: Number of filters of the intermediate conv layers.
    mask_target_size: Number that represents the resolution of masks.
    train: `bool` of using training mode or not in Batch Normalization.
    dtype: type of variables used.
    batch_norm_group_size: The batch norm group size.
    use_batch_norm: Use batch norm or layer norm.
    use_class_agnostic: Use class agnostic mask output.
    use_norm_activation: Use normalized activation in the mask output layer.
      This is to normalize the weights and input features to reduce the scale
      variance for different categories.
  """
  num_classes: int = 2
  num_convs: int = 4
  num_filters: int = 256
  mask_target_size: int = 28
  upsample_factor: int = 2
  train: bool = True
  dtype: jnp.dtype = jnp.float32
  activation_fn: Callable[[Array], Array] = nn.relu
  batch_norm_group_size: int = 0
  use_batch_norm: bool = True
  use_class_agnostic: bool = False
  use_norm_activation: bool = False

  @nn.compact
  def __call__(
      self,
      roi_features,
      class_indices):
    """Mask branch for the Mask-RCNN model.

    Args:
      roi_features: A ROI feature array of shape
        [batch_size, num_rois, height_l, width_l, feature_dim].
      class_indices: an array of shape [batch_size, num_rois], indicating
        which class the ROI is.

    Returns:
      mask_outputs: an array with a shape of
        [batch_size, num_rois, mask_height, mask_width],
        representing the mask predictions for each ROI.
    """
    if self.use_class_agnostic:
      if self.num_classes > 1:
        raise ValueError('Number of classes must be 1 with agnostic masks.')
    batch_size, num_rois, height, width, feature_dim = roi_features.shape
    if class_indices.shape != (batch_size, num_rois):
      raise ValueError(f'Unexpected class indices shape: '
                       f'({class_indices.shape[0]}, {class_indices.shape[1]}),'
                       f' expected ({batch_size}, {num_rois})')
    if height != width:
      raise ValueError('Crop feature height and width must be equal.')

    if self.upsample_factor * height != self.mask_target_size:
      raise ValueError(
          f'Upsampled mask height: {self.upsample_factor * height} '
          f'must equal mask target size: {self.mask_target_size}!')
    x = roi_features.reshape([-1, height, width, feature_dim])
    # Use variance scaling kernel initializer to match TF1 implementation.
    conv_layer = NormalizedConv if self.use_norm_activation else nn.Conv
    conv_fn = functools.partial(
        conv_layer,
        strides=(1, 1),
        dtype=self.dtype,
        kernel_init=initializers.variance_scaling(
            scale=2, mode='fan_out', distribution='normal'),
        bias_init=initializers.zeros,
        )
    # Set momentum and epsilon values to match TF1 implementation.
    if self.use_batch_norm:
      norm_fn = functools.partial(
          nn.BatchNorm,
          use_running_average=not self.train,
          momentum=0.997,
          epsilon=1e-4,
          axis_name='batch' if self.batch_norm_group_size else None,
          axis_index_groups=model_utils.get_device_groups(
              self.batch_norm_group_size, batch_size)
          if self.train and self.batch_norm_group_size else None,
          dtype=self.dtype)
    else:
      norm_fn = functools.partial(nn.LayerNorm, dtype=self.dtype)

    for i in range(self.num_convs):
      x = conv_fn(kernel_size=(3, 3),
                  features=self.num_filters,
                  padding='SAME',
                  name=f'mask-conv-l{i}')(x)
      x = norm_fn(name=f'mask-conv-l{i}-bn')(x)
      x = self.activation_fn(x)

    # Upsample convolution layer.
    upconv_kernel_size = upconv_strides = (self.upsample_factor,
                                           self.upsample_factor)
    x = nn.ConvTranspose(
        features=self.num_filters,
        kernel_size=upconv_kernel_size,
        strides=upconv_strides,
        padding='VALID',
        kernel_init=initializers.variance_scaling(
            scale=2, mode='fan_out', distribution='normal'),
        bias_init=initializers.zeros,
        name='conv5-mask',
        dtype=self.dtype)(x)
    x = norm_fn(name='conv5-mask-bn')(x)
    x = self.activation_fn(x)
    # Output linear layer.
    x = conv_fn(
        kernel_size=(1, 1),
        features=self.num_classes,
        padding='VALID',
        # Set to float32 as a temporary fix to avoid numerical instability.
        dtype=jnp.float32,
        name='mask_fcn_logits')(x)
    x = x.reshape([
        batch_size, num_rois, self.mask_target_size, self.mask_target_size,
        self.num_classes
    ])
    if self.use_class_agnostic:
      mask_outputs = x[:, :, :, :, 0]
    else:
      # Gather the mask corresponding to the class id for each ROI.
      batch_indices = jnp.tile(jnp.arange(batch_size)[:, None], [1, num_rois])
      mask_indices = jnp.tile(jnp.arange(num_rois)[None, :], [batch_size, 1])
      mask_outputs = x[batch_indices, mask_indices, :, :, class_indices]
    return mask_outputs
