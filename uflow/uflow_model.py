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

# Lint as: python3

"""UFlow models.

This library contains the models used in UFlow. Our model is a slightly modified
version of the PWC net by Sun et al (https://arxiv.org/pdf/1709.02371.pdf).

In particular, we change the number of layers and filters in the feature
pyramid, we introduce a pyramid-level dropout, and we normalize the features
before computing a cost volume. We found these changes to improve the
performance.
"""

# Lint as: python3

import collections

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential

from uflow import uflow_utils


def normalize_features(feature_list, normalize, center, moments_across_channels,
                       moments_across_images):
  """Normalizes feature tensors (e.g., before computing the cost volume).

  Args:
    feature_list: list of tf.tensors, each with dimensions [b, h, w, c]
    normalize: bool flag, divide features by their standard deviation
    center: bool flag, subtract feature mean
    moments_across_channels: bool flag, compute mean and std across channels
    moments_across_images: bool flag, compute mean and std across images

  Returns:
    list, normalized feature_list
  """

  # Compute feature statistics.

  statistics = collections.defaultdict(list)
  axes = [-3, -2, -1] if moments_across_channels else [-3, -2]
  for feature_image in feature_list:
    mean, variance = tf.nn.moments(x=feature_image, axes=axes, keepdims=True)
    statistics['mean'].append(mean)
    statistics['var'].append(variance)

  if moments_across_images:
    statistics['mean'] = ([tf.reduce_mean(input_tensor=statistics['mean'])] *
                          len(feature_list))
    statistics['var'] = [tf.reduce_mean(input_tensor=statistics['var'])
                        ] * len(feature_list)

  statistics['std'] = [tf.sqrt(v + 1e-16) for v in statistics['var']]

  # Center and normalize features.

  if center:
    feature_list = [
        f - mean for f, mean in zip(feature_list, statistics['mean'])
    ]
  if normalize:
    feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

  return feature_list


def compute_cost_volume(features1, features2, max_displacement):
  """Compute the cost volume between features1 and features2.

  Displace features2 up to max_displacement in any direction and compute the
  per pixel cost of features1 and the displaced features2.

  Args:
    features1: tf.tensor of shape [b, h, w, c]
    features2: tf.tensor of shape [b, h, w, c]
    max_displacement: int, maximum displacement for cost volume computation.

  Returns:
    tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
    all displacements.
  """

  # Set maximum displacement and compute the number of image shifts.
  _, height, width, _ = features1.shape.as_list()
  if max_displacement <= 0 or max_displacement >= height:
    raise ValueError(f'Max displacement of {max_displacement} is too large.')

  max_disp = max_displacement
  num_shifts = 2 * max_disp + 1

  # Pad features2 and shift it while keeping features1 fixed to compute the
  # cost volume through correlation.

  # Pad features2 such that shifts do not go out of bounds.
  features2_padded = tf.pad(
      tensor=features2,
      paddings=[[0, 0], [max_disp, max_disp], [max_disp, max_disp], [0, 0]],
      mode='CONSTANT')
  cost_list = []
  for i in range(num_shifts):
    for j in range(num_shifts):
      corr = tf.reduce_mean(
          input_tensor=features1 *
          features2_padded[:, i:(height + i), j:(width + j), :],
          axis=-1,
          keepdims=True)
      cost_list.append(corr)
  cost_volume = tf.concat(cost_list, axis=-1)
  return cost_volume


class PWCFlow(Model):
  """Model for estimating flow based on the feature pyramids of two images."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               dropout_rate=0.25,
               num_channels_upsampled_context=32,
               num_levels=5,
               normalize_before_cost_volume=True,
               channel_multiplier=1.,
               use_cost_volume=True,
               use_feature_warp=True,
               accumulate_flow=True,
               use_bfloat16=False,
               shared_flow_decoder=False):

    super(PWCFlow, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'float32')
    self._leaky_relu_alpha = leaky_relu_alpha
    self._drop_out_rate = dropout_rate
    self._num_context_up_channels = num_channels_upsampled_context
    self._num_levels = num_levels
    self._normalize_before_cost_volume = normalize_before_cost_volume
    self._channel_multiplier = channel_multiplier
    self._use_cost_volume = use_cost_volume
    self._use_feature_warp = use_feature_warp
    self._accumulate_flow = accumulate_flow
    self._shared_flow_decoder = shared_flow_decoder

    self._refine_model = self._build_refinement_model()
    self._flow_layers = self._build_flow_layers()
    if not self._use_cost_volume:
      self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs(
      )
    if num_channels_upsampled_context:
      self._context_up_layers = self._build_upsample_layers(
          num_channels=int(num_channels_upsampled_context * channel_multiplier))
    if self._shared_flow_decoder:
      # pylint:disable=invalid-name
      self._1x1_shared_decoder = self._build_1x1_shared_decoder()

  def call(self, feature_pyramid1, feature_pyramid2, training=False):
    """Run the model."""
    context = None
    flow = None
    flow_up = None
    context_up = None
    flows = []

    # Go top down through the levels to the second to last one to estimate flow.
    for level, (features1, features2) in reversed(
        list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):

      # init flows with zeros for coarsest level if needed
      if self._shared_flow_decoder and flow_up is None:
        batch_size, height, width, _ = features1.shape.as_list()
        flow_up = tf.zeros(
            [batch_size, height, width, 2],
            dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)
        if self._num_context_up_channels:
          num_channels = int(self._num_context_up_channels *
                             self._channel_multiplier)
          context_up = tf.zeros(
              [batch_size, height, width, num_channels],
              dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)

      # Warp features2 with upsampled flow from higher level.
      if flow_up is None or not self._use_feature_warp:
        warped2 = features2
      else:
        warp_up = uflow_utils.flow_to_warp(flow_up)
        warped2 = uflow_utils.resample(features2, warp_up)

      # Compute cost volume by comparing features1 and warped features2.
      features1_normalized, warped2_normalized = normalize_features(
          [features1, warped2],
          normalize=self._normalize_before_cost_volume,
          center=self._normalize_before_cost_volume,
          moments_across_channels=True,
          moments_across_images=True)

      if self._use_cost_volume:
        cost_volume = compute_cost_volume(
            features1_normalized, warped2_normalized, max_displacement=4)
      else:
        concat_features = Concatenate(axis=-1)(
            [features1_normalized, warped2_normalized])
        cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

      cost_volume = LeakyReLU(
          alpha=self._leaky_relu_alpha, dtype=self._dtype_policy)(
              cost_volume)

      if self._shared_flow_decoder:
        # This will ensure to work for arbitrary feature sizes per level.
        conv_1x1 = self._1x1_shared_decoder[level]
        features1 = conv_1x1(features1)

      # Compute context and flow from previous flow, cost volume, and features1.
      if flow_up is None:
        x_in = Concatenate(axis=-1)([cost_volume, features1])
      else:
        if context_up is None:
          x_in = Concatenate(axis=-1)([flow_up, cost_volume, features1])
        else:
          x_in = Concatenate(axis=-1)(
              [context_up, flow_up, cost_volume, features1])

      # Use dense-net connections.
      x_out = None
      if self._shared_flow_decoder:
        # reuse the same flow decoder on all levels
        flow_layers = self._flow_layers
      else:
        flow_layers = self._flow_layers[level]
      for layer in flow_layers[:-1]:
        x_out = layer(x_in)
        x_in = Concatenate(axis=-1)([x_in, x_out])
      context = x_out

      flow = flow_layers[-1](context)

      if (training and self._drop_out_rate):
        maybe_dropout = tf.cast(
            tf.math.greater(tf.random.uniform([]), self._drop_out_rate),
            tf.bfloat16 if self._use_bfloat16 else tf.float32)
        context *= maybe_dropout
        flow *= maybe_dropout

      if flow_up is not None and self._accumulate_flow:
        flow += flow_up

      # Upsample flow for the next lower level.
      flow_up = uflow_utils.upsample(flow, is_flow=True)
      if self._num_context_up_channels:
        context_up = self._context_up_layers[level](context)

      # Append results to list.
      flows.insert(0, flow)

    # Refine flow at level 1.
    refinement = self._refine_model([context, flow])
    if (training and self._drop_out_rate):
      refinement *= tf.cast(
          tf.math.greater(tf.random.uniform([]), self._drop_out_rate),
          tf.bfloat16 if self._use_bfloat16 else tf.float32)
    refined_flow = flow + refinement
    flows[0] = refined_flow
    return [tf.cast(flow, tf.float32) for flow in flows]

  def _build_cost_volume_surrogate_convs(self):
    layers = []
    for _ in range(self._num_levels):
      layers.append(
          Conv2D(
              int(64 * self._channel_multiplier),
              kernel_size=(4, 4),
              padding='same',
              dtype=self._dtype_policy))
    return layers

  def _build_upsample_layers(self, num_channels):
    """Build layers for upsampling via deconvolution."""
    layers = []
    for unused_level in range(self._num_levels):
      layers.append(
          Conv2DTranspose(
              num_channels,
              kernel_size=(4, 4),
              strides=2,
              padding='same',
              dtype=self._dtype_policy))
    return layers

  def _build_flow_layers(self):
    """Build layers for flow estimation."""
    # Empty list of layers level 0 because flow is only estimated at levels > 0.
    result = [[]]
    for _ in range(1, self._num_levels):
      layers = []
      for c in [128, 128, 96, 64, 32]:
        layers.append(
            Sequential([
                Conv2D(
                    int(c * self._channel_multiplier),
                    kernel_size=(3, 3),
                    strides=1,
                    padding='same',
                    dtype=self._dtype_policy),
                LeakyReLU(
                    alpha=self._leaky_relu_alpha, dtype=self._dtype_policy)
            ]))
      layers.append(
          Conv2D(
              2,
              kernel_size=(3, 3),
              strides=1,
              padding='same',
              dtype=self._dtype_policy))
      if self._shared_flow_decoder:
        return layers
      result.append(layers)
    return result

  def _build_refinement_model(self):
    """Build model for flow refinement using dilated convolutions."""
    layers = []
    layers.append(Concatenate(axis=-1))
    for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
      layers.append(
          Conv2D(
              int(c * self._channel_multiplier),
              kernel_size=(3, 3),
              strides=1,
              padding='same',
              dilation_rate=d,
              dtype=self._dtype_policy))
      layers.append(
          LeakyReLU(alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
    layers.append(
        Conv2D(
            2,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            dtype=self._dtype_policy))
    return Sequential(layers)

  def _build_1x1_shared_decoder(self):
    """Build layers for flow estimation."""
    # Empty list of layers level 0 because flow is only estimated at levels > 0.
    result = [[]]
    for _ in range(1, self._num_levels):
      result.append(
          Conv2D(
              32,
              kernel_size=(1, 1),
              strides=1,
              padding='same',
              dtype=self._dtype_policy))
    return result


class PWCFeaturePyramid(Model):
  """Model for computing a feature pyramid from an image."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               filters=None,
               level1_num_layers=3,
               level1_num_filters=16,
               level1_num_1x1=0,
               original_layer_sizes=False,
               num_levels=5,
               channel_multiplier=1.,
               pyramid_resolution='half',
               use_bfloat16=False):
    """Constructor.

    Args:
      leaky_relu_alpha: Float. Alpha for leaky ReLU.
      filters: Tuple of tuples. Used to construct feature pyramid. Each tuple is
        of form (num_convs_per_group, num_filters_per_conv).
      level1_num_layers: How many layers and filters to use on the first
        pyramid. Only relevant if filters is None and original_layer_sizes
        is False.
      level1_num_filters: int, how many filters to include on pyramid layer 1.
        Only relevant if filters is None and original_layer_sizes if False.
      level1_num_1x1: How many 1x1 convolutions to use on the first pyramid
        level.
      original_layer_sizes: bool, if True, use the original PWC net number
        of layers and filters.
      num_levels: int, How many feature pyramid levels to construct.
      channel_multiplier: float, used to scale up or down the amount of
        computation by increasing or decreasing the number of channels
        by this factor.
      pyramid_resolution: str, specifies the resolution of the lowest (closest
        to input pyramid resolution)
      use_bfloat16: bool, whether or not to run in bfloat16 mode.
    """

    super(PWCFeaturePyramid, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'float32')
    self._channel_multiplier = channel_multiplier
    if num_levels > 6:
      raise NotImplementedError('Max number of pyramid levels is 6')
    if filters is None:
      if original_layer_sizes:
        # Orig - last layer
        filters = ((3, 16), (3, 32), (3, 64), (3, 96), (3, 128),
                   (3, 196))[:num_levels]
      else:
        filters = ((level1_num_layers, level1_num_filters), (3, 32), (3, 32),
                   (3, 32), (3, 32), (3, 32))[:num_levels]
    assert filters
    assert all(len(t) == 2 for t in filters)
    assert all(t[0] > 0 for t in filters)

    self._leaky_relu_alpha = leaky_relu_alpha
    self._convs = []
    self._level1_num_1x1 = level1_num_1x1

    for level, (num_layers, num_filters) in enumerate(filters):
      group = []
      for i in range(num_layers):
        stride = 1
        if i == 0 or (i == 1 and level == 0 and
                      pyramid_resolution == 'quarter'):
          stride = 2
        conv = Conv2D(
            int(num_filters * self._channel_multiplier),
            kernel_size=(3,
                         3) if level > 0 or i < num_layers - level1_num_1x1 else
            (1, 1),
            strides=stride,
            padding='valid',
            dtype=self._dtype_policy)
        group.append(conv)
      self._convs.append(group)

  def call(self, x, split_features_by_sample=False):
    if self._use_bfloat16:
      x = tf.cast(x, tf.bfloat16)
    x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
    features = []
    for level, conv_tuple in enumerate(self._convs):
      for i, conv in enumerate(conv_tuple):
        if level > 0 or i < len(conv_tuple) - self._level1_num_1x1:
          x = tf.pad(
              tensor=x,
              paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
              mode='CONSTANT')
        x = conv(x)
        x = LeakyReLU(alpha=self._leaky_relu_alpha, dtype=self._dtype_policy)(x)
      features.append(x)

    if split_features_by_sample:

      # Split the list of features per level (for all samples) into a nested
      # list that can be indexed by [sample][level].

      n = len(features[0])
      features = [[f[i:i + 1] for f in features] for i in range(n)]  # pylint: disable=g-complex-comprehension

    return features
