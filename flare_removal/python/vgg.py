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

"""Wrappers and extensions for a pre-trained VGG-19 network."""

import collections
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


class Vgg19(tf.keras.Model):
  """A pre-trained VGG-19 network with configurable tap-outs.

  Supported layers and their output shapes are:
  - block1_conv1 .. 2: [B,    H,    W,  64]
  - block1_pool:       [B,  H/2,  W/2,  64]
  - block2_conv1 .. 2: [B,  H/2,  W/2, 128]
  - block2_pool:       [B,  H/4,  W/4, 128]
  - block3_conv1 .. 4: [B,  H/4,  W/4, 256]
  - block3_pool:       [B,  H/8,  W/8, 256]
  - block4_conv1 .. 4: [B,  H/8,  W/8, 512]
  - block4_pool:       [B, H/16, W/16, 512]
  - block5_conv1 .. 4: [B, H/16, W/16, 512]
  - block5_pool:       [B, H/32, W/32, 512]
  where [B, H, W, 3] is the batched input image tensor.

  Attributes:
    tap_out_layers: A list of names of the layers configured for tap-out.
  """

  def __init__(self,
               tap_out_layers,
               trainable = False,
               weights = 'imagenet'):
    """Initializes a pre-trained VGG-19 network.

    Args:
      tap_out_layers: Names of the layers used as tap-out points. The output
        tensors of these layers will be returned when model is called. Must be a
        subset of the supported layers listed above.
      trainable: Whether the network's weights are frozen.
      weights: Source of the pre-trained weights. Use None if the network is to
        be initialized randomly. See `tf.keras.applications.VGG19` for details.

    Raises:
      ValueError: If `tap_out_layers` has duplicate or invalid entries.
    """
    super(Vgg19, self).__init__(name='vgg19')
    if len(set(tap_out_layers)) != len(tap_out_layers):
      raise ValueError(f'There are duplicates in the provided layers: '
                       f'{tap_out_layers}')

    # Load pre-trained weights.
    model = tf.keras.applications.VGG19(include_top=False, weights=weights)
    model.trainable = trainable
    invalid_layers = set(tap_out_layers) - set(l.name for l in model.layers)
    if invalid_layers:
      raise ValueError(f'Unrecognized layers: {invalid_layers}')
    self.tap_out_layers = tap_out_layers

    # Divide the feed-forward network into a series of segments, each of which
    # ends with a requested layer.
    # Implementation note: the default dictionary (dict) keeps insertion order
    # as of Python 3.7, making it equivalent to OrderedDict. However, we still
    # use OrderedDict here for greater compatibility and readability.
    self._ordered_segments = collections.OrderedDict()
    segment = tf.keras.Sequential()
    for layer in model.layers:
      segment.add(layer)
      if layer.name in tap_out_layers:
        self._ordered_segments[layer.name] = segment
        segment = tf.keras.Sequential()

  def call(self, images, **kwargs):
    """Invokes the model on batched images.

    Args:
      images: A [B, H, W, C]-tensor of type float32, in range [0, 1].
      **kwargs: Other arguments in the base class are ignored.

    Returns:
      Output tensors of the tap-out layers, in the same order as
      `self.tap_out_layers`.
    """
    features = {}
    # Scale from [0, 1] to [0, 255], convert to BGR channel order, and subtract
    # channel means.
    x = tf.keras.applications.vgg19.preprocess_input(images * 255.0)
    for layer, segment in self._ordered_segments.items():
      x = segment(x)
      features[layer] = x
    # Reorder according to given `tap_out_layers`.
    return [features[layer] for layer in self.tap_out_layers]


class IdentityInitializer(tf.keras.initializers.Initializer):
  """Initializes a Conv2D kernel as an identity transform.

  Specifically, the identity kernel does the following (assuming M input
  channels and N output channels):
  - If M >= N, the first N channels of the input are copied over to the output.
  - If M < N, the input is copied to the first M channels of the output, and the
    rest of the output is zero.

  The kernel weight matrix is assumed to have 4 dimensions: [H, W, M, N], where
  (H, W) are the size of each 2-D kernel, and (M, N) are the number of
  input/output channels.

  Note that this differs from the `tf.keras.initializers.Identity` initializer,
  which works on 2-D weight matrices.
  """

  def __call__(self,
               shape,
               dtype = tf.float32,
               **kwargs):
    array = np.zeros(shape, dtype=dtype.as_numpy_dtype)
    kernel_height, kernel_width, in_channels, out_channels = shape
    cy, cx = kernel_height // 2, kernel_width // 2
    for i in range(np.minimum(in_channels, out_channels)):
      array[cy, cx, i, i] = 1
    return tf.constant(array)


class _CanBlock(tf.keras.layers.Layer):
  """A convolutional block in the context aggregation network."""

  def __init__(self, channels, size, rate, **kwargs):
    """Initializes a convolutional block.

    Args:
      channels: Number of output channels.
      size: Side length of the square kernel.
      rate: Dilation rate.
      **kwargs: Other args passed into `Layer`.
    """
    super(_CanBlock, self).__init__(**kwargs)
    self.channels = channels
    self.size = size
    self.rate = rate

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv2D(
        filters=self.channels,
        kernel_size=self.size,
        dilation_rate=self.rate,
        padding='same',
        use_bias=False,
        kernel_initializer=IdentityInitializer(),
        input_shape=input_shape)
    # Trainable weights for normalization.
    self.w0 = self.add_weight(
        'w0',
        dtype=tf.float32,
        initializer=tf.keras.initializers.Constant(1.0),
        trainable=True)
    self.w1 = self.add_weight(
        'w1',
        dtype=tf.float32,
        initializer=tf.keras.initializers.Constant(0.0),
        trainable=True)
    self.batch_norm = tf.keras.layers.BatchNormalization(scale=False)
    self.activation = tf.keras.layers.LeakyReLU(0.2)

  def call(self, inputs):
    convolved = self.conv(inputs)
    normalized = self.w0 * convolved + self.w1 * self.batch_norm(convolved)
    outputs = self.activation(normalized)
    return outputs


def build_can(input_shape = (512, 512, 3),
              conv_channels=64,
              out_channels=3,
              name='can'):
  """A context aggregation network based on the pre-trained VGG-19 network.

  Reference:
  X. Zhang, R. Ng, and Q. Chen. Single image reflection removal with perceptual
  loss. CVPR, 2018.

  Args:
    input_shape: Shape of the input tensor, without the batch dimension. For a
      typical RGB image, this should be [height, width, 3].
    conv_channels: Number of channels in the intermediate convolution blocks.
    out_channels: Number of output channels.
    name: Name of this model. Will also be added as a prefix to the weight
      variable names.

  Returns:
    A Keras Model object.
  """
  input_layer = tf.keras.Input(shape=input_shape, name='input')

  vgg = Vgg19(
      tap_out_layers=[f'block{i}_conv2' for i in range(1, 6)], trainable=False)
  features = vgg(input_layer)
  features = [tf.image.resize(f, input_shape[:2]) / 255.0 for f in features]

  x = tf.concat([input_layer] + features, axis=-1)

  x = _CanBlock(conv_channels, size=1, rate=1, name=f'{name}_g_conv0')(x)

  for i, rate in enumerate([1, 2, 4, 8, 16, 32, 64, 1]):
    x = _CanBlock(
        conv_channels, size=3, rate=rate, name=f'{name}_g_conv{i + 1}')(
            x)

  output_layer = tf.keras.layers.Conv2D(
      out_channels,
      kernel_size=1,
      dilation_rate=1,
      padding='same',
      use_bias=False,
      name=f'{name}_g_conv_last')(
          x)

  return tf.keras.Model(input_layer, output_layer, name=name)
