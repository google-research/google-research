# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Keras-based ConvNet with additional conditioning inputs.

Currently the network is conditioned on extra input vector with something like
conditional batch normalization, but without batch normalization.
For each hidden layer its output h_i is element-wise multiplied
by a vector m_i and element-wise summed with a vector a_i, both m_i and a_i
predicted with MLPs from the conditioning input.
"""

import gin
import tensorflow.compat.v1 as tf
from yoto.architectures import utils


def broadcast_and_concat(x, inputs_extra):
  inputs_extra_expanded = tf.expand_dims(tf.expand_dims(inputs_extra, -2), -2)
  tile_multipliers = [1] + x.get_shape().as_list()[1:3] + [1]
  inputs_extra_broadcast = tf.tile(inputs_extra_expanded, tile_multipliers)
  return tf.concat([x, inputs_extra_broadcast], axis=3)


@gin.configurable("ConditionalConvnet")
class ConditionalConvnet(tf.keras.Model):
  """A simple VGG-style convolutional net conditioned on an additional vector.

  The network consists of several blocks of convolutional layers with 3x3
  kernels and with either stride or transposed stride inbetween. The net is
  conditioned on the extra inputs by channel-wise mutiplicative and additive
  modulation by vectors produced by an MLP.
  """

  def __init__(self, num_blocks=3, layers_per_block=2, base_num_channels=16,
               upconv=False, fc_layer_sizes=None, upconv_reshape_size=None,
               conditioning_layer_sizes=None, channels_out=3, alpha=0.3,
               conditioning_postprocessing=None,
               final_sigmoid=False, conditioning_type="mult_and_add",
               kernel_initializer_mode="fan_in"):
    """Initializes a ConditionalConvnet object.

    Args:
      num_blocks: Int, number of convolutional blocks
      layers_per_block: Int, number of conv layers per block
      base_num_channels: Int, number of channels in the highest resolution
        block. For other blocks the number of channels is doubled every time the
        resolution is reduced by 2x.
      upconv: Bool. If False, the network is a usual convolutional net (with
        downsampling), if True, it is an upconvolutional (a.k.a.
        deconvolutional) net, with upsampling.
      fc_layer_sizes: List[Int]. Sizes of the hidden layers of a fully connected
        net, which is put on top of a convnet or before an upconvnet. If None,
        there is no FC network, so the convnet is fully convolutional.
      upconv_reshape_size: List[Int], of len 3. Shape of the first convolutional
        feature map to be fed to the upconv net (should be supplied only if
        upconv=True and fc_layer_sizes is not None)
      conditioning_layer_sizes: List of Ints, sizes of hidden layers of the
        conditioning MLP.
      channels_out: Int, number of channels in the output (only relevant if
        upconv == True)
      alpha: Float. negative slope for Leaky ReLUs.
      conditioning_postprocessing: a class describing how
        to post-process the conditioning scales and shifts
      final_sigmoid: Bool. Whether to apply a sigmoid non-linearity at the end
        of the model.
      conditioning_type: Str, one of ["mult_and_add", "concat", "input"].
        Specifies in which way to condition the model on the extra inputs.
      kernel_initializer_mode: Str. Specifies the mode of the VarianceScaling
        initializer used for weight initialization
    """

    super(ConditionalConvnet, self).__init__()
    self._num_blocks = num_blocks
    self._layers_per_block = layers_per_block
    self._base_num_channels = base_num_channels
    self._channels_out = channels_out
    self._upconv = upconv
    self._fc_layer_sizes = fc_layer_sizes
    self._upconv_reshape_size = upconv_reshape_size
    self._final_sigmoid = final_sigmoid
    if upconv_reshape_size is not None and ((not upconv) or
                                            (fc_layer_sizes is None)):
      raise ValueError("upconv_reshape_size should be supplied only if "
                       "upconv=True and fc_layer_sizes is not None.")
    self._conditioning_layer_sizes = conditioning_layer_sizes
    self._nonlinearity = lambda x: tf.nn.leaky_relu(x, alpha)
    if conditioning_postprocessing is not None:
      self._conditioning_postprocessing = conditioning_postprocessing()
    else:
      self._conditioning_postprocessing = None
    if conditioning_type not in ["mult_and_add", "concat", "input"]:
      raise ValueError("Unknown conditioning_type {}".format(conditioning_type))
    self._conditioning_type = conditioning_type
    scale_factor = 2. / (1. + alpha**2)
    self._kernel_initializer = tf.keras.initializers.VarianceScaling(
        mode=kernel_initializer_mode, scale=scale_factor)

  def _call_upconv(self, inputs, inputs_extra):
    endpoints = {}
    x = inputs
    if self._fc_layer_sizes is not None:
      x = tf.keras.layers.Flatten()(x)
      x = utils.mlp(x, self._fc_layer_sizes)
      x = tf.reshape(x, [-1] + self._upconv_reshape_size)
    for nblock in range(self._num_blocks-1, -1, -1):
      num_channels = self._base_num_channels * (2**nblock)
      if (inputs_extra is not None and
          (self._conditioning_type == "concat"
           or (self._conditioning_type == "input" and nblock == 0))):
        x = broadcast_and_concat(x, inputs_extra)
      with tf.variable_scope("upconv_block_{}".format(nblock)):
        x = self._upconv_block(x, self._layers_per_block, num_channels)
        if (inputs_extra is not None and
            self._conditioning_type == "mult_and_add"):
          endpoints["upconv_block_before_cond_{}".format(nblock)] = x
          with tf.variable_scope("conditioning"):
            x = self._condition_conv(x, inputs_extra,
                                     self._conditioning_layer_sizes)
        endpoints["upconv_block_before_nonlin_{}".format(nblock)] = x
        x = self._nonlinearity(x)
      endpoints["upconv_block_{}".format(nblock)] = x
    # final layer that outputs the required number of channels
    x = tf.layers.conv2d(inputs=x, filters=self._channels_out,
                         kernel_size=3, strides=(1, 1), padding="same",
                         kernel_initializer=self._kernel_initializer)
    return x, endpoints

  def _call_conv(self, inputs, inputs_extra):
    endpoints = {}
    x = inputs
    for nblock in range(self._num_blocks):
      if (inputs_extra is not None and
          (self._conditioning_type == "concat"
           or (self._conditioning_type == "input" and nblock == 0))):
        x = broadcast_and_concat(x, inputs_extra)
      num_channels = self._base_num_channels * (2**nblock)
      with tf.variable_scope("conv_block_{}".format(nblock)):
        x = self._conv_block(x, self._layers_per_block, num_channels)
        if (inputs_extra is not None and
            self._conditioning_type == "mult_and_add"):
          endpoints["conv_block_before_cond_{}".format(nblock)] = x
          with tf.variable_scope("conditioning"):
            x = self._condition_conv(x, inputs_extra,
                                     self._conditioning_layer_sizes)
        endpoints["conv_block_before_nonlin_{}".format(nblock)] = x
        if nblock < self._num_blocks - 1:
          x = self._nonlinearity(x)
      endpoints["conv_block_{}".format(nblock)] = x
    if self._fc_layer_sizes is not None:
      x = tf.keras.layers.Flatten()(x)
      x = utils.mlp(x, self._fc_layer_sizes)
    return x, endpoints

  def call(self, inputs, inputs_extra=None, training=None):
    del training  # No batchnorm here
    if self._upconv:
      with tf.variable_scope("upconvnet"):
        outputs, endpoints = self._call_upconv(inputs, inputs_extra)
    else:
      with tf.variable_scope("convnet"):
        outputs, endpoints = self._call_conv(inputs, inputs_extra)
    if self._final_sigmoid:
      outputs = tf.keras.activations.sigmoid(outputs)
    return outputs, endpoints

  def _conv_block(self, inputs, num_layers, num_channels):
    x = inputs
    for n in range(num_layers):
      strides = (2, 2) if n == 0 else (1, 1)
      x = tf.layers.conv2d(inputs=x, filters=num_channels, kernel_size=3,
                           strides=strides, padding="same",
                           kernel_initializer=self._kernel_initializer)
      if n < num_layers - 1:
        x = self._nonlinearity(x)
    return x

  def _upconv_block(self, inputs, num_layers, num_channels):
    x = inputs
    for n in range(num_layers):
      strides = (2, 2) if n == num_layers-1 else (1, 1)
      x = tf.layers.conv2d_transpose(
          inputs=x, filters=num_channels, kernel_size=3, strides=strides,
          padding="same", kernel_initializer=self._kernel_initializer)
      if n < num_layers - 1:
        x = self._nonlinearity(x)
    return x

  def _condition_conv(self, inputs, inputs_extra, layer_sizes):
    channels = inputs.get_shape().as_list()[-1]
    inputs_extra_flat = tf.keras.layers.Flatten()(inputs_extra)
    scales = utils.mlp(inputs_extra_flat, layer_sizes + [channels])
    shifts = utils.mlp(inputs_extra_flat, layer_sizes + [channels])
    scales = tf.expand_dims(tf.expand_dims(scales, -2), -2)
    shifts = tf.expand_dims(tf.expand_dims(shifts, -2), -2)
    if self._conditioning_postprocessing is not None:
      scales, shifts = self._conditioning_postprocessing(scales, shifts)
    return scales * inputs + shifts

