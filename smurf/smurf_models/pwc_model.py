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

"""PWC model."""
# pylint:skip-file
import collections

import gin
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU

from smurf import smurf_utils


def upsample(img, is_flow):
  """Double resolution of an image or flow field.

  Args:
    img: tf.tensor, image or flow field to be resized
    is_flow: bool, flag for scaling flow accordingly

  Returns:
    Resized and potentially scaled image or flow field.
  """
  _, height, width, _ = img.shape.as_list()
  orig_dtype = img.dtype
  if orig_dtype != tf.float32:
    img = tf.cast(img, tf.float32)
  img_resized = tf.compat.v2.image.resize(img,
                                          (int(height * 2), int(width * 2)))
  if is_flow:
    # Scale flow values to be consistent with the new image size.
    img_resized *= 2
  if img_resized.dtype != orig_dtype:
    return tf.cast(img_resized, orig_dtype)
  return img_resized



class ConvLeaky(tf.keras.layers.Layer):
  """Simple layer which composes Conv and LeakyRelu."""

  def __init__(self,
               filters,
               leaky_relu_alpha,
               kernel_size=(3, 3),
               strides=1,
               padding='same'):
    """Composes an conv and leaky relu."""
    super(ConvLeaky, self).__init__()
    self._conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)
    self._leaky = LeakyReLU(alpha=leaky_relu_alpha)

  def call(self, inputs, training=False):
    out = self._conv(inputs)
    return self._leaky(out)


class RefinementModel(tf.keras.Model):
  """Model for refining flow predictions and producing final output."""

  def __init__(self, channel_multiplier, leaky_relu_alpha):
    """Initialize the model."""
    super(RefinementModel, self).__init__()
    layers = []
    layers.append(Concatenate(axis=-1))
    for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
      layers.append(
          Conv2D(int(c * channel_multiplier),
                 kernel_size=(3, 3),
                 strides=1,
                 padding='same',
                 dilation_rate=d))
      layers.append(LeakyReLU(alpha=leaky_relu_alpha))
      layers.append(
          Conv2D(
              2,
              kernel_size=(3, 3),
              strides=1,
              padding='same'))
    self._refine_layers = layers

  def call(self, x, y, training=False):
    out = self._refine_layers[0]([x, y])
    for layer in self._refine_layers[1:]:
      out = layer(out)
    return out


def normalize_features(feature_list, normalize, center, moments_across_channels,
                       moments_across_images):
  """Normalize a list of feature images (e.g.

  before computing the cost volume).

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

  max_disp = max_displacement
  num_shifts = 2 * max_disp + 1

  # Pad features2 and shift it while keeping features1 fixed to compute the
  # cost volume through correlation.

  _, height, width, _ = features1.shape.as_list()
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

  def __init__(
      self,
      leaky_relu_alpha=0.1,
      dropout_rate=0.25,
      num_channels_upsampled_context=32,
      num_levels=5,
      output_flow_at_level=1,
      normalize_before_cost_volume=True,
      channel_multiplier=1.,
      use_cost_volume=True,
      use_feature_warp=True,
      accumulate_flow=True,
      shared_flow_decoder=False
  ):

    super(PWCFlow, self).__init__()
    self._leaky_relu_alpha = leaky_relu_alpha
    self._drop_out_rate = dropout_rate
    self._num_context_up_channels = num_channels_upsampled_context
    self._num_levels = num_levels
    self._output_flow_at_level = output_flow_at_level
    self._normalize_before_cost_volume = normalize_before_cost_volume
    self._channel_multiplier = channel_multiplier
    self._use_cost_volume = use_cost_volume
    self._use_feature_warp = use_feature_warp
    self._accumulate_flow = accumulate_flow
    self._shared_flow_decoder = shared_flow_decoder

    self._refine_model = self._build_refinement_model()
    self._flow_layers = self._build_flow_layers()
    if not self._use_cost_volume:
      self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs()
    if num_channels_upsampled_context:
      self._context_up_layers = self._build_upsample_layers(
          num_channels=int(num_channels_upsampled_context *
                           channel_multiplier))
    if self._shared_flow_decoder:
      self._1x1_shared_decoder = self._build_1x1_shared_decoder()

  def call(self, feature_dict, training=False, backward=False):
    """Run the model."""
    context = None
    flow = None
    flow_up = None
    context_up = None
    flows = []

    if backward:
      feature_pyramid1 = feature_dict['features2']
      feature_pyramid2 = feature_dict['features1']
    else:
      feature_pyramid1 = feature_dict['features1']
      feature_pyramid2 = feature_dict['features2']

    # Go top down through the levels to the second to last one to estimate flow.
    for level, (features1, features2) in reversed(
        list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[self._output_flow_at_level:]):

      # init flows with zeros for coarsest level if needed
      if self._shared_flow_decoder and flow_up is None:
        batch_size, height, width, _ = features1.shape.as_list()
        flow_up = tf.zeros([batch_size, height, width, 2])
        if self._num_context_up_channels:
          num_channels = int(self._num_context_up_channels *
                           self._channel_multiplier)
          context_up = tf.zeros(
              [batch_size, height, width, num_channels])

      # Warp features2 with upsampled flow from higher level.
      if flow_up is None or not self._use_feature_warp:
        warped2 = features2
      else:
        warp_up = smurf_utils.flow_to_warp(flow_up)
        warped2 = smurf_utils.resample(features2, warp_up)

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

      cost_volume = LeakyReLU(alpha=self._leaky_relu_alpha)(cost_volume)

      if self._shared_flow_decoder:
        # this will ensure to work for arbitrary feature sizes per level
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
            tf.float32)
        context *= maybe_dropout
        flow *= maybe_dropout

      if flow_up is not None and self._accumulate_flow:
        flow += flow_up

      # Upsample flow for the next lower level.
      flow_up = upsample(flow, is_flow=True)
      if self._num_context_up_channels:
        context_up = self._context_up_layers[level](context)

      # Append results to list.
      flows.insert(0, flow)

    # Refine flow at level '_output_flow_at_level'.
    refinement = self._refine_model(context, flow)
    if (training and self._drop_out_rate):
      refinement *= tf.cast(tf.math.greater(
          tf.random.uniform([]), self._drop_out_rate), tf.float32)
    refined_flow = flow + refinement
    flows[0] = refined_flow

    # Upsample flow to the highest available feature resolution.
    for _ in range(self._output_flow_at_level):
      upsampled_flow = upsample(flows[0], is_flow=True)
      flows.insert(0, upsampled_flow)

    # Upsample flow to the original input resolution.
    upsampled_flow = upsample(flows[0], is_flow=True)
    flows.insert(0, upsampled_flow)
    return [tf.cast(flow, tf.float32) for flow in flows]

  def _build_cost_volume_surrogate_convs(self):
    layers = []
    for _ in range(self._num_levels):
      layers.append(
          Conv2D(
              int(64 * self._channel_multiplier),
              kernel_size=(4, 4),
              padding='same'))
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
              padding='same'))
    return layers

  def _build_flow_layers(self):
    """Build layers for flow estimation."""
    # Empty list of layers level 0 because flow is only estimated at levels > 0.
    result = []
    for _ in range(self._output_flow_at_level):
      result.append([])
    for _ in range(self._output_flow_at_level, self._num_levels):
      layers = []
      for c in [128, 128, 96, 64, 32]:
        layers.append(ConvLeaky(filters=c * self._channel_multiplier,
                                leaky_relu_alpha=self._leaky_relu_alpha))
      layers.append(
          Conv2D(
              2,
              kernel_size=(3, 3),
              strides=1,
              padding='same'))
      if self._shared_flow_decoder:
        return layers
      result.append(layers)
    return result

  def _build_refinement_model(self):
    """Build model for flow refinement using dilated convolutions."""
    return RefinementModel(self._channel_multiplier,
                           self._leaky_relu_alpha)

  def _build_1x1_shared_decoder(self):
    """Build layers for flow estimation."""
    result = []
    for _ in range(self._output_flow_at_level):
      result.append([])
    for _ in range(self._output_flow_at_level, self._num_levels):
      result.append(
          Conv2D(
              32,
              kernel_size=(1, 1),
              strides=1,
              padding='same'))
    return result


class PWCFeaturePyramid(Model):
  """Model for computing a feature pyramid from an image."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               filters=None,
               level1_num_layers=3,
               level1_num_filters=32,
               level1_num_1x1=0,
               original_layer_sizes=False,
               num_levels=5,
               channel_multiplier=1.,
               pyramid_resolution='half'):
    """Constructor.

    Args:
      leaky_relu_alpha: Float. Alpha for leaky ReLU.
      filters: Tuple of tuples. Used to construct feature pyramid. Each tuple is
        of form (num_convs_per_group, num_filters_per_conv).
        level1_num_layers: int, the number of layers in pyramid `level 1`
        level1_num_filters: int, the number of filters in pyramid `level 1`
    """

    super(PWCFeaturePyramid, self).__init__()
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
            padding='valid')
        group.append(conv)
      self._convs.append(group)

  def call(self, x, split_features_by_sample=False):
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
        x = LeakyReLU(alpha=self._leaky_relu_alpha)(x)
      features.append(x)

    if split_features_by_sample:

      # Split the list of features per level (for all samples) into a nested
      # list that can be indexed by [sample][level].
      n = len(features[0])
      features = [[f[i:i + 1] for f in features] for i in range(n)]

    return features


class PWCFeatureSiamese(PWCFeaturePyramid):
  """Model for computing feature pyramids from an image pair.

  Wraps the PWC-Net feature model to compute features for an image pair.
  """

  def call(self,
           image1,
           image2,
           training=False,
           bidirectional=False):
    """Runs the model.

    Args:
      image1: First/reference image batch [b, h, w, c].
      image2: Second image batch [b, h, w, c].
      training: Flag indicating if the model is being trained or not.
      bidirectional: Flag indicating if features should also be computed for
        the reversed image order of the pair (this doesn't have and effect for
        this model).

    Returns:
      Dictionary holding the feature pyramids for both images.
    """
    features1 = super(PWCFeatureSiamese, self).call(image1)
    features2 = super(PWCFeatureSiamese, self).call(image2)
    return {'features1': features1, 'features2': features2}

