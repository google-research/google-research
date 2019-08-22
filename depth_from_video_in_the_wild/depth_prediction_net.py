# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

import numpy as np
import tensorflow as tf

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

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
  econv2 = s_residual_block(x, is_training, name='conv2_2')

  # conv3_x
  x = s_residual_block_first(
      econv2,
      is_training,
      encoder_filters[2],
      stride,
      name='conv3_1',
      normalizer_fn=normalizer_fn)
  econv3 = s_residual_block(x, is_training, name='conv3_2')

  # conv4_x
  x = s_residual_block_first(
      econv3,
      is_training,
      encoder_filters[3],
      stride,
      name='conv4_1',
      normalizer_fn=normalizer_fn)
  econv4 = s_residual_block(x, is_training, name='conv4_2')

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
    # Main operation: conv2d
    with tf.device('/CPU:0'):
      kernel = tf.get_variable(
          'kernel', [filter_size, filter_size, in_shape[3], out_channel],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
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
    return tf.maximum(x, x*leakness, name='lrelu')
  else:
    name = 'relu' if name is None else name
    return tf.nn.relu(x, name='relu')
