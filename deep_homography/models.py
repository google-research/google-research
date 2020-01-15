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

# Lint as: python2, python3
"""Homography estimation neural network models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import range
import tensorflow.compat.v1 as tf

from deep_homography import hmg_util
from tensorflow.contrib import slim as contrib_slim
from tensorflow.contrib.slim.nets import vgg as contrib_slim_nets_vgg

slim = contrib_slim

VGG_MEANS = [123.68, 116.779, 103.939]


def homography_arg_scope(weight_decay=0.0005, activation_fn=tf.nn.relu):
  """Defines the homography network arg scope.

  Args:
    weight_decay: the l2 regularization coefficient
    activation_fn: activation functions for convolutional layers
  Returns:
    an arg_scope
  """
  batch_norm_var_collection = 'moving_vars'
  batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'fused': None,
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }
  with slim.arg_scope([slim.conv2d, slim.conv3d, slim.conv2d_transpose,
                       slim.conv3d_transpose], activation_fn=activation_fn,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer(),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params,
                      padding='SAME') as arg_sc:
    return arg_sc


def hier_homography_fmask_estimator(color_inputs, num_param=8, num_layer=7,
                                    num_level=3, dropout_keep_prob=0.8,
                                    reuse=None, is_training=True,
                                    trainable=True,
                                    scope='hier_hmg'):
  """A hierarchical neural network with mask for homograhy estimation.

  Args:
    color_inputs: batch of input image pairs of data type float32 and of shape
      [batch_size, height, width, 6]
    num_param: the number of parameters for homography (default 8)
    num_layer: the number of convolutional layers in the motion feature network
    num_level: the number of hierarchical levels
    dropout_keep_prob: the percentage of activation values that are kept
    reuse: whether to reuse this network weights
    is_training: whether used for training or testing
    trainable: whether this network is to be trained or not
    scope: the scope of variables in this function

  Returns:
    a list of homographies at each level and motion feature maps if
    final_endpoint='mfeature'; otherwise a list of images warped by the list of
    corresponding homographies
  """
  _, h_input, w_input = color_inputs.get_shape().as_list()[0 : 3]
  vgg_inputs = (color_inputs[Ellipsis, 3 : 6] * 256 + 128)- VGG_MEANS

  with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=False):
      with slim.arg_scope([slim.conv2d], normalizer_fn=None):
        with slim.arg_scope(contrib_slim_nets_vgg.vgg_arg_scope()):
          sfeature, _ = contrib_slim_nets_vgg.vgg_16(
              vgg_inputs,
              1000,
              predictions_fn=slim.softmax,
              global_pool=False,
              is_training=False,
              reuse=reuse,
              spatial_squeeze=True,
              final_endpoint='pool5',
              scope='vgg_16')

  gray_image1 = tf.image.rgb_to_grayscale(color_inputs[Ellipsis, 0 : 3])
  gray_image2 = tf.image.rgb_to_grayscale(color_inputs[Ellipsis, 3 : 6])
  inputs = tf.concat([gray_image1, gray_image2], 3)

  hmgs_list = []
  warped_list = []
  with tf.variable_scope(scope, [inputs], reuse=reuse):
    for level_index in range(num_level):
      scale = 2 ** (num_level - 1 - level_index)
      h = tf.to_float(tf.floordiv(h_input, scale))
      w = tf.to_float(tf.floordiv(w_input, scale))
      inputs_il = tf.image.resize_images(inputs, tf.to_int32([h, w]))
      if level_index == 0:
        mfeature = hier_base_layers(inputs_il,
                                    num_layer + 1 - num_level + level_index,
                                    level_index, is_training=is_training,
                                    trainable=trainable)
        hmgs_il = homography_regression(mfeature, num_param, level_index,
                                        dropout_keep_prob=dropout_keep_prob,
                                        is_training=is_training,
                                        trainable=trainable)
        hmgs_list.append(hmgs_il)
      else:
        warped, _ = hmg_util.homography_scale_warp_per_batch(
            inputs_il[:, :, :, 0], w / 2, h / 2, hmgs_list[level_index - 1])
        pre_warped_inputs_il = tf.stack([warped, inputs_il[:, :, :, 1]], -1)
        warped_list.append(pre_warped_inputs_il)
        mfeature = hier_base_layers(pre_warped_inputs_il,
                                    num_layer + 1 - num_level + level_index,
                                    level_index, is_training=is_training,
                                    trainable=trainable)
        if level_index == num_level - 1:
          mfeature = fmask_layers_semantic(mfeature, sfeature, level_index,
                                           is_training=is_training,
                                           trainable=trainable)
        hmgs_il = homography_regression(mfeature, num_param, level_index,
                                        dropout_keep_prob=dropout_keep_prob,
                                        is_training=is_training,
                                        trainable=trainable)
        new_hmgs_il = hmg_util.homography_shift_mult_batch(
            hmgs_list[level_index - 1], w / 2, h / 2, hmgs_il, w, h, w, h)
        hmgs_list.append(new_hmgs_il)
  return hmgs_list, warped_list


def hier_homography_estimator(inputs, num_param=8, num_layer=7, num_level=3,
                              dropout_keep_prob=0.8, reuse=None,
                              is_training=True, trainable=True,
                              final_endpoint=None, scope='hier_hmg'):
  """A hierarchical VGG-style neural network for homograhy estimation.

  Args:
    inputs: batch of input image pairs of data type float32 and of shape
      [batch_size, height, width, 2]
    num_param: the number of parameters for homography (default 8)
    num_layer: the number of convolutional layers in the motion feature network
    num_level: the number of hierarchical levels
    dropout_keep_prob: the percentage of activation values that are kept
    reuse: whether to reuse this network weights
    is_training: whether used for training or testing
    trainable: whether this network is to be trained or not
    final_endpoint: specifies the endpoint to construct the network up to
    scope: the scope of variables in this function

  Returns:
    a list of homographies at each level and motion feature maps if
    final_endpoint='mfeature'; otherwise a list of images warped by the list of
    corresponding homographies
  """
  _, h_input, w_input = inputs.get_shape().as_list()[0:3]
  hmgs_list = []
  warped_list = []
  with tf.variable_scope(scope, [inputs], reuse=reuse):
    for level_index in range(num_level):
      scale = 2 ** (num_level - 1 - level_index)
      h = tf.to_float(tf.floordiv(h_input, scale))
      w = tf.to_float(tf.floordiv(w_input, scale))
      inputs_il = tf.image.resize_images(inputs, tf.to_int32([h, w]))
      if level_index == 0:
        mfeature = hier_base_layers(inputs_il,
                                    num_layer + 1 - num_level + level_index,
                                    level_index, is_training=is_training,
                                    trainable=trainable)
        hmgs_il = homography_regression(mfeature, num_param, level_index,
                                        dropout_keep_prob=dropout_keep_prob,
                                        is_training=is_training,
                                        trainable=trainable)
        hmgs_list.append(hmgs_il)
      else:
        warped, _ = hmg_util.homography_scale_warp_per_batch(
            inputs_il[:, :, :, 0], w / 2, h / 2, hmgs_list[level_index - 1])
        pre_warped_inputs_il = tf.stack([warped, inputs_il[:, :, :, 1]], -1)
        warped_list.append(pre_warped_inputs_il)
        if level_index == num_level - 1 and final_endpoint == 'mfeature':
          mfeature = hier_base_layers(pre_warped_inputs_il,
                                      num_layer - num_level + level_index,
                                      level_index, is_training=is_training,
                                      trainable=trainable)
          return hmgs_list, mfeature
        else:
          mfeature = hier_base_layers(pre_warped_inputs_il,
                                      num_layer + 1 - num_level + level_index,
                                      level_index, is_training=is_training,
                                      trainable=trainable)
        hmgs_il = homography_regression(mfeature, num_param, level_index,
                                        dropout_keep_prob=dropout_keep_prob,
                                        is_training=is_training,
                                        trainable=trainable)
        new_hmgs_il = hmg_util.homography_shift_mult_batch(
            hmgs_list[level_index - 1], w / 2, h / 2, hmgs_il, w, h, w, h)
        hmgs_list.append(new_hmgs_il)
  return hmgs_list, warped_list


def hier_base_layers(images, num_layer, level, is_training=True,
                     trainable=True):
  """Base sub-convolutional network to compute motion features.

  Args:
    images: input images of data type float32 and of shape
      [batch_size, height, width, channel]
    num_layer: the number of layers
    level: the hierachical level
    is_training: whether used for training or testing
    trainable: whether the model parameters are trainable
  Returns:
    motion features of data type float32 and of shape
      [batch_size, feature_height, feature_width, feature_channel]
  """
  with tf.variable_scope('level%d' % level):
    with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=trainable):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        mfeature = slim.conv2d(images, 32, [3, 3], scope='conv1_0')
        mfeature = slim.conv2d(mfeature, 32, [3, 3], scope='conv1_1')
        mfeature = slim.max_pool2d(mfeature, [2, 2], padding='SAME',
                                   scope='pool1')
        for layer_index in range(1, num_layer):
          scale = 2 ** math.floor((layer_index + 1) / 2)
          num_channel = scale * 32
          mfeature = slim.conv2d(mfeature, num_channel, [3, 3],
                                 scope='conv%d_0' % (layer_index + 1))
          mfeature = slim.conv2d(mfeature, num_channel, [3, 3],
                                 scope='conv%d_1' % (layer_index + 1))
          if layer_index < num_layer - 1:
            mfeature = slim.max_pool2d(mfeature, [2, 2], padding='SAME',
                                       scope='pool%d' % (layer_index + 1))
  return mfeature


def homography_regression(mfeature, num_param, level=1, dropout_keep_prob=0.8,
                          is_training=True, trainable=True):
  """Regresses homographies from the given features.

  Args:
    mfeature: features to estimate homographies from of data type float32 and
      of shape [batch_size, height, width, channel]
    num_param: the number of parameters to represent a homography
    level: the hierachical level
    dropout_keep_prob: the percentage of activation values that are kept
    is_training: whether used for training or testing
    trainable: whether the model parameters are trainable
  Returns:
    homographies of data type float32 and of shape [batch_size, num_param]
  """
  with tf.variable_scope('level%d' % level):
    with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=trainable):
      with slim.arg_scope([slim.batch_norm, slim.dropout],
                          is_training=is_training):
        kernel_size = mfeature.get_shape().as_list()[1:3]
        net = slim.avg_pool2d(mfeature, kernel_size, padding='VALID',
                              scope='avgpool_{}x{}'.format(*kernel_size))
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout')
        logits = slim.conv2d(net, num_param, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='final_conv')
        logits = tf.squeeze(logits, [1, 2], name='spatial_squeeze')
  return logits


def fmask_layers_semantic(mfeature, sfeature, level, is_training=True,
                          trainable=True):
  """Base sub-convolutional network to compute motion features.

  Args:
    mfeature: motion feature maps of data type float32 and of shape
      [batch_size, height, width, channel]
    sfeature: appearance feature maps of data type float32 and of shape
      [batch_size, height, width, channel]
    level: the hierachical level
    is_training: whether used for training or testing
    trainable: whether the model parameters are trainable
  Returns:
    motion features of data type float32 and of shape
      [batch_size, feature_height, feature_width, feature_channel]
  """
  with tf.variable_scope('mask_level%d' % level):
    with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=trainable):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        nchan = mfeature.get_shape().as_list()[3]
        sfeature = slim.conv2d(sfeature, 256, [3, 3], scope='sconv')
        mnet = tf.concat([mfeature, sfeature], 3)
        mnet = slim.conv2d(mnet, 2 * nchan, [3, 3], scope='conv1')
        mnet = slim.conv2d(mnet, 2 * nchan, [3, 3], scope='conv2')
        mnet = slim.conv2d(mnet, 2 * nchan, [3, 3], scope='conv3')
        kernel_size = mnet.get_shape().as_list()[1 : 3]
        mnet = slim.avg_pool2d(mnet, kernel_size, padding='VALID',
                               scope='avgpool_{}x{}'.format(*kernel_size))
        mnet = slim.conv2d(mnet, nchan, [1, 1], activation_fn=None,
                           normalizer_fn=None, scope='conv4')
        mnet = tf.nn.softmax(mnet)
        out_mfeature = tf.multiply(mfeature, mnet)
        out_mfeature = slim.conv2d(out_mfeature, 256, [3, 3], scope='conv5')
        out_mfeature = slim.conv2d(out_mfeature, 256, [3, 3], scope='conv6')
  return out_mfeature
