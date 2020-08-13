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

"""Depth-prediction networks, based on the Struct2Depth code.

https://github.com/tensorflow/models/blob/master/research/struct2depth/nets.py
"""

import abc

import numpy as np
import tensorflow.compat.v1 as tf

from depth_and_motion_learning import maybe_summary
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

layers = contrib_layers
arg_scope = contrib_framework.arg_scope

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def encoder_resnet(target_image, weight_reg, is_training, normalizer_fn=None):
  """Defines a ResNet18-based encoding architecture.

  This implementation follows Juyong Kim's implementation of ResNet18 on GitHub:
  https://github.com/dalgu90/resnet-18-tensorflow

  Args:
    target_image: Input tensor with shape [B, h, w, 3] to encode.
    weight_reg: Parameter ignored.
    is_training: Whether the model is being trained or not.
    normalizer_fn: Normalization function, defaults to batch normalization (_bn)
      below.

  Returns:
    Tuple of tensors, with the first being the bottleneck layer as tensor of
    size [B, h_hid, w_hid, c_hid], and others being intermediate layers
    for building skip-connections.
  """
  del weight_reg
  normalizer_fn = normalizer_fn or _bn
  encoder_filters = [64, 64, 128, 256, 512]
  stride = 2

  # conv1
  with tf.variable_scope('conv1'):
    x = s_conv(target_image, 7, encoder_filters[0], stride)
    x = normalizer_fn(x, is_train=is_training)
    econv1 = s_relu(x)
    x = tf.nn.max_pool(econv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

  # conv2_x
  x = s_residual_block(
      x, is_training, name='conv2_1', normalizer_fn=normalizer_fn)
  econv2 = s_residual_block(
      x, is_training, name='conv2_2', normalizer_fn=normalizer_fn)

  # conv3_x
  x = s_residual_block_first(
      econv2,
      is_training,
      encoder_filters[2],
      stride,
      name='conv3_1',
      normalizer_fn=normalizer_fn)
  econv3 = s_residual_block(
      x, is_training, name='conv3_2', normalizer_fn=normalizer_fn)

  # conv4_x
  x = s_residual_block_first(
      econv3,
      is_training,
      encoder_filters[3],
      stride,
      name='conv4_1',
      normalizer_fn=normalizer_fn)
  econv4 = s_residual_block(
      x, is_training, name='conv4_2', normalizer_fn=normalizer_fn)

  # conv5_x
  x = s_residual_block_first(
      econv4,
      is_training,
      encoder_filters[4],
      stride,
      name='conv5_1',
      normalizer_fn=normalizer_fn)
  econv5 = s_residual_block(
      x, is_training, name='conv5_2', normalizer_fn=normalizer_fn)
  return econv5, (econv4, econv3, econv2, econv1)


class GenericDepthPredictor(object):
  """An abstract class for a depth predictor."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, mode, params=None):
    """Creates an instance.

    Args:
      mode: One of tf.estimator.ModeKeys: TRAIN, PREDICT or EVAL.
      params: A dictionary containing relevant parameters.
    """
    allowed_attrs = ['TRAIN', 'PREDICT', 'EVAL']
    allowed_values = [
        getattr(tf.estimator.ModeKeys, attr) for attr in allowed_attrs
    ]
    if mode not in allowed_values:
      raise ValueError('\'mode\' must be one of tf.estimator.ModeKeys.(%s)' %
                       ', '.join(allowed_attrs))
    self._mode = mode
    self._params = self._default_params
    self._params.update(params or {})

  @property
  def _defalut_params(self):
    return {}

  @abc.abstractmethod
  def predict_depth(self, rgb, sensor_depth):
    """An interface for predicting depth.

    Args:
      rgb: A batch of RGB images, of shape [B, H, W, 3].
      sensor_depth: Optional, batch of depth sensor images of shape [B, H, W],
        to be fused into the prediction.
    """
    pass


class ResNet18DepthPredictor(GenericDepthPredictor):
  """A depth predictor based on ResNet18 with randomized layer normalization."""

  @property
  def _default_params(self):
    return {
        # Number of training steps over which the noise in randomized layer
        # normalization ramps up.
        'layer_norm_noise_rampup_steps': 10000,

        # Weight decay regularization of the network base.
        'weight_decay': 0.01,

        # If true, a learned scale factor will multiply the network's depth
        # prediction. This is useful when direct depth supervision exists.
        'learn_scale': False,

        # A boolean, if True, deconvolutions will be padded in 'REFLECT' mode,
        # otherwise in 'CONSTANT' mode (the former is not supported on TPU)
        'reflect_padding': False
    }

  def predict_depth(self, rgb, sensor_depth=None):
    del sensor_depth  # unused
    with tf.variable_scope('depth_prediction', reuse=tf.AUTO_REUSE):
      if self._mode == tf.estimator.ModeKeys.TRAIN:
        noise_stddev = 0.5
        global_step = tf.train.get_global_step()
        rampup_steps = self._params['layer_norm_noise_rampup_steps']
        if global_step is not None and rampup_steps > 0:
          # If global_step is available, ramp up the noise.
          noise_stddev *= tf.square(
              tf.minimum(tf.to_float(global_step) / float(rampup_steps), 1.0))
      else:
        noise_stddev = 0.0

      def _normalizer_fn(x, is_train, name='bn'):
        return randomized_layer_norm(
            x, is_train=is_train, name=name, stddev=noise_stddev)

      if self._params['learn_scale']:
        depth_scale = tf.get_variable('depth_scale', initializer=1.0)
        maybe_summary.scalar('depth_scale', depth_scale)
      else:
        depth_scale = 1.0

      return depth_scale * depth_prediction_resnet18unet(
          2 * rgb - 1.0,
          self._mode == tf.estimator.ModeKeys.TRAIN,
          self._params['weight_decay'],
          _normalizer_fn,
          reflect_padding=self._params['reflect_padding'])


def depth_prediction_resnet18unet(images, is_training, decoder_weight_reg=0.0,
                                  normalizer_fn=None, reflect_padding=True):
  """A depth prediciton network based on a ResNet18 UNet architecture.

  This network is identical to disp_net in struct2depth.nets with
  architecture='resnet', with the following differences:

  1. We use a softplus activation to generate positive depths. This eliminates
     the need for the hyperparameters DISP_SCALING and MIN_DISP defined in
     struct2depth.nets. The predicted depth is no longer bounded.

  2. The network predicts depth rather than disparity, and at a single scale.

  Args:
    images: A tf.Tensor of shape [B, H, W, C] representing images.
    is_training: A boolean, True if in training mode.
    decoder_weight_reg: A scalar, strength of L2 weight regularization to be
      used in the decoder.
    normalizer_fn: Normalizer function to use for convolutions. Defaults to
      batch normalization.
    reflect_padding: A boolean, if True, deconvolutions will be padded in
      'REFLECT' mode, otherwise in 'CONSTANT' mode (the former is not supported
      on  TPU)

  Returns:
    A tf.Tensor of shape [B, H, W, 1] containing depths maps.
  """

  # The struct2depth resnet encoder does not use the weight_reg argument, hence
  # we're passing None.
  bottleneck, skip_connections = encoder_resnet(
      images,
      weight_reg=None,
      is_training=is_training,
      normalizer_fn=normalizer_fn)

  (econv4, econv3, econv2, econv1) = skip_connections
  decoder_filters = [16, 32, 64, 128, 256]
  reg = layers.l2_regularizer(decoder_weight_reg)
  padding_mode = 'REFLECT' if reflect_padding else 'CONSTANT'
  with arg_scope([layers.conv2d, layers.conv2d_transpose],
                 normalizer_fn=None,
                 normalizer_params=None,
                 activation_fn=tf.nn.relu,
                 weights_regularizer=reg):
    upconv5 = layers.conv2d_transpose(
        bottleneck, decoder_filters[4], [3, 3], stride=2, scope='upconv5')
    iconv5 = layers.conv2d(
        _concat_and_pad(upconv5, econv4, padding_mode),
        decoder_filters[4], [3, 3],
        stride=1,
        scope='iconv5',
        padding='VALID')
    upconv4 = layers.conv2d_transpose(
        iconv5, decoder_filters[3], [3, 3], stride=2, scope='upconv4')
    iconv4 = layers.conv2d(
        _concat_and_pad(upconv4, econv3, padding_mode),
        decoder_filters[3], [3, 3],
        stride=1,
        scope='iconv4',
        padding='VALID')
    upconv3 = layers.conv2d_transpose(
        iconv4, decoder_filters[2], [3, 3], stride=2, scope='upconv3')
    iconv3 = layers.conv2d(
        _concat_and_pad(upconv3, econv2, padding_mode),
        decoder_filters[2], [3, 3],
        stride=1,
        scope='iconv3',
        padding='VALID')
    upconv2 = layers.conv2d_transpose(
        iconv3, decoder_filters[1], [3, 3], stride=2, scope='upconv2')
    iconv2 = layers.conv2d(
        _concat_and_pad(upconv2, econv1, padding_mode),
        decoder_filters[1], [3, 3],
        stride=1,
        scope='iconv2',
        padding='VALID')
    upconv1 = layers.conv2d_transpose(
        iconv2, decoder_filters[0], [3, 3], stride=2, scope='upconv1')
    upconv1 = tf.pad(
        upconv1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)
    iconv1 = layers.conv2d(
        upconv1,
        decoder_filters[0], [3, 3],
        stride=1,
        scope='iconv1',
        padding='VALID')
    depth_input = tf.pad(
        iconv1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)

    return layers.conv2d(
        depth_input,
        1, [3, 3],
        stride=1,
        activation_fn=tf.nn.softplus,
        normalizer_fn=None,
        scope='disp1',
        padding='VALID')


def _concat_and_pad(decoder_layer, encoder_layer, padding_mode):
  concat = tf.concat([decoder_layer, encoder_layer], axis=3)
  return tf.pad(concat, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)


def randomized_layer_norm(x, is_train, name='bn', stddev=0.5):
  """Applies layer normalization and applies noise on the mean and variance.

  For every item in a batch and for every layer, we calculate the mean and
  variance across the spatial dimensions, and multiply them by Gaussian noise
  with a mean equal to 1.0 (at training time only). This improved the results
  compared to batch normalization - see more in
  https://arxiv.org/abs/1904.04998.

  Args:
    x: tf.Tensor to normalize, of shape [B, H, W, C].
    is_train: A boolean, True at training mode.
    name: A string, a name scope.
    stddev: Standard deviation of the Gaussian noise. Defaults to 0.5 because
      this is the largest value where the noise is guaranteed to be a
      non-negative multiplicative factor

  Returns:
    A tf.Tensor of shape [B, H, W, C], the normalized tensor.
  """

  with tf.variable_scope(name, None, [x]):
    inputs_shape = x.shape.as_list()
    params_shape = inputs_shape[-1:]
    beta = tf.get_variable(
        'beta', shape=params_shape, initializer=tf.initializers.zeros())
    gamma = tf.get_variable(
        'gamma', shape=params_shape, initializer=tf.initializers.ones())
    mean, variance = tf.nn.moments(x, [1, 2], keep_dims=True)
    if is_train:
      mean *= 1.0 + tf.random.truncated_normal(tf.shape(mean), stddev=stddev)
      variance *= 1.0 + tf.random.truncated_normal(
          tf.shape(variance), stddev=stddev)
    outputs = tf.nn.batch_normalization(
        x,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-3)
    outputs.set_shape(x.shape)
  return outputs


def s_residual_block_first(x,
                           is_training,
                           out_channel,
                           strides,
                           name='unit',
                           normalizer_fn=None):
  """Helper function for defining ResNet architecture."""
  normalizer_fn = normalizer_fn or _bn
  in_channel = x.get_shape().as_list()[-1]
  with tf.variable_scope(name):
    # Shortcut connection
    if in_channel == out_channel:
      if strides == 1:
        shortcut = tf.identity(x)
      else:
        shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                  [1, strides, strides, 1], 'VALID')
    else:
      shortcut = s_conv(x, 1, out_channel, strides, name='shortcut')
    # Residual
    x = s_conv(x, 3, out_channel, strides, name='conv_1')
    x = normalizer_fn(x, is_train=is_training, name='bn_1')
    x = s_relu(x, name='relu_1')
    x = s_conv(x, 3, out_channel, 1, name='conv_2')
    x = normalizer_fn(x, is_train=is_training, name='bn_2')
    # Merge
    x = x + shortcut
    x = s_relu(x, name='relu_2')
  return x


def s_residual_block(x,
                     is_training,
                     input_q=None,
                     output_q=None,
                     name='unit',
                     normalizer_fn=None):
  """Helper function for defining ResNet architecture."""
  normalizer_fn = normalizer_fn or _bn
  num_channel = x.get_shape().as_list()[-1]
  with tf.variable_scope(name):
    shortcut = x  # Shortcut connection
    # Residual
    x = s_conv(
        x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
    x = normalizer_fn(x, is_train=is_training, name='bn_1')
    x = s_relu(x, name='relu_1')
    x = s_conv(
        x,
        3,
        num_channel,
        1,
        input_q=output_q,
        output_q=output_q,
        name='conv_2')
    x = normalizer_fn(x, is_train=is_training, name='bn_2')
    # Merge
    x = x + shortcut
    x = s_relu(x, name='relu_2')
  return x


def s_conv(x,
           filter_size,
           out_channel,
           stride,
           pad='SAME',
           input_q=None,
           output_q=None,
           name='conv'):
  """Helper function for defining ResNet architecture."""
  if (input_q is None) ^ (output_q is None):
    raise ValueError('Input/Output splits are not correctly given.')

  in_shape = x.get_shape()
  with tf.variable_scope(name):
    kernel = tf.get_variable(
        'kernel', [filter_size, filter_size, in_shape[3], out_channel],
        tf.float32,
        initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)))
    if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
      tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
    conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], pad)
  return conv


def _bn(x, is_train, name='bn'):
  """Helper function for defining ResNet architecture."""
  bn = tf.layers.batch_normalization(x, training=is_train, name=name)
  return bn


def s_relu(x, name=None, leakness=0.0):
  """Helper function for defining ResNet architecture."""
  if leakness > 0.0:
    name = 'lrelu' if name is None else name
    return tf.maximum(x, x * leakness, name='lrelu')
  else:
    name = 'relu' if name is None else name
    return tf.nn.relu(x, name='relu')
