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

"""Residual cell DNA to create and serialize models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from evanet import inception_cell_spec_pb2 as search_proto

from evanet import tgm_layer
from tensorflow.contrib import slim as contrib_slim
from tensorflow.contrib.slim import initializers as contrib_slim_initializers

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last'):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  del data_format
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  var = {
      'beta': None,
      'gamma': None,
      'moving_mean': ['moving_vars'],
      'moving_variance': ['moving_vars'],
  }

  inputs = contrib_slim.batch_norm(
      inputs=inputs,
      decay=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      is_training=is_training,
      fused=False,
      variables_collections=var,
      param_initializers={'gamma': gamma_initializer})

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def conv3d(inputs,
           filters,
           kernel_size,
           strides=1,
           scope=None,
           dilation=1,
           data_format=''):
  """Returns Conv3D wrapped with default values."""

  del data_format
  del dilation
  init = contrib_slim_initializers.variance_scaling_initializer

  return contrib_slim.conv3d(
      inputs,
      filters,
      kernel_size=kernel_size,
      stride=strides,
      padding='SAME',
      activation_fn=None,
      biases_initializer=None,
      normalizer_fn=None,
      scope=scope,
      weights_initializer=init(factor=2.0, mode='FAN_IN', uniform=False))


def conv21d(inputs,
            filters,
            kernel_size,
            strides=1,
            is_training=False,
            scope=None,
            dilation=1,
            data_format=''):
  """Returns conv(2+1)D with default values."""

  del data_format
  del dilation
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size, kernel_size, kernel_size]
  if isinstance(strides, int):
    strides = [strides, strides, strides]

  init = contrib_slim_initializers.variance_scaling_initializer
  inputs = contrib_slim.conv3d(
      inputs,
      filters // 2,
      kernel_size=[kernel_size[0], 1, 1],
      stride=[strides[0], 1, 1],
      padding='SAME',
      activation_fn=None,
      biases_initializer=None,
      normalizer_fn=None,
      scope=scope,
      weights_initializer=init(factor=2.0, mode='FAN_IN', uniform=False))
  inputs = batch_norm_relu(inputs, is_training, relu=True)

  inputs = contrib_slim.conv3d(
      inputs,
      filters,
      kernel_size=[1, kernel_size[1], kernel_size[2]],
      stride=[1, strides[1], strides[2]],
      padding='SAME',
      activation_fn=None,
      biases_initializer=None,
      normalizer_fn=None,
      scope=scope + 's',
      weights_initializer=init(factor=2.0, mode='FAN_IN', uniform=False))
  return inputs


def tgm_conv3d(inputs,
               filters,
               kernel_size,
               strides=1,
               num_gaussian=4,
               scope=None,
               dilation=1,
               data_format=''):
  """Returns TGM conv with default values."""

  del data_format
  del dilation
  init = contrib_slim_initializers.variance_scaling_initializer
  return tgm_layer.tgm_3d_conv(
      inputs,
      filters,
      kernel_size=kernel_size,
      stride=strides,
      padding='SAME',
      activation_fn=None,
      num=num_gaussian,
      normalizer_fn=None,
      scope=scope,
      weights_initializer=init(factor=2.0, mode='FAN_IN', uniform=False))


class ModelDNA(object):
  """Constructs a residual model from DNA for the video CNN.

    TYPE can be 1: 3D conv
                2: TGM conv
                3: (2+1)D conv
    Temporal size can be 1,3,5,7,9,11 and determines the temporal
    size of the conv kernel
  """

  def __init__(self, serialized_dna, dropout=0.5, num_classes=51):
    self.data_format = 'channels_last'
    self.spec = search_proto.DNASpec()
    self.spec.ParseFromString(serialized_dna)
    self.data_dir = None
    self.layers = 0
    self.dropout_keep_prob = dropout
    self.num_classes = num_classes

  def serialize(self):
    return self.spec.SerializeToString()

  def block(self,
            net,
            total_filters,
            layers,
            scope,
            data_format,
            use_projection=False,
            use_time=True):
    """Block function."""

    axis = 4
    is_training = self.is_training
    filters = total_filters // len(layers)
    total_filters = filters * len(layers)

    branches = []
    with tf.variable_scope(scope):
      shortcut = net
      if use_projection:
        shortcut = conv3d(net, total_filters, [1, 1, 1], scope='shortcut')
        shortcut = batch_norm_relu(
            shortcut, is_training, relu=False, data_format=data_format)

      for i, layer in enumerate(layers):
        with tf.variable_scope('Branch_' + str(i)):
          if layer.layer_type == search_proto.Layer.SINGLE_DEFAULT:
            branch = conv3d(
                net,
                filters, [1, 1, 1],
                scope='Conv2d_0a_1x1',
                data_format=data_format)
            branch = batch_norm_relu(
                branch, is_training, relu=False, data_format=data_format)
          elif layer.layer_type == search_proto.Layer.CONV:
            branch = conv3d(
                net,
                filters // 2, [1, 1, 1],
                scope='Conv2d_0a_1x1',
                data_format=data_format)
            branch = batch_norm_relu(
                branch, is_training, relu=True, data_format=data_format)
            conv_fn = self.get_layer_type(layer.conv_type)
            branch = conv_fn(
                branch,
                filters, [layer.time if use_time else 1, 3, 3],
                scope='Conv2d_0b_3x3',
                data_format=data_format,
                dilation=layer.dilation if use_time else 1)
            branch = batch_norm_relu(
                branch, is_training, relu=False, data_format=data_format)
          elif layer.layer_type == search_proto.Layer.CONV2:
            branch = conv3d(
                net,
                filters // 4, [1, 1, 1],
                scope='Conv2d_0a_1x1',
                data_format=data_format)
            branch = batch_norm_relu(
                branch, is_training, relu=True, data_format=data_format)
            conv_fn = self.get_layer_type(layer.conv_type)
            branch = conv_fn(
                branch,
                filters // 2, [layer.time if use_time else 1, 3, 3],
                scope='Conv2d_0b_3x3',
                data_format=data_format,
                dilation=layer.dilation if use_time else 1)
            branch = batch_norm_relu(
                branch, is_training, relu=True, data_format=data_format)
            conv_fn = self.get_layer_type(layer.conv_type2)
            branch = conv_fn(
                branch,
                filters, [layer.time2 if use_time else 1, 3, 3],
                scope='Conv2d_0c_3x3',
                data_format=data_format,
                dilation=layer.dilation2 if use_time else 1)
            branch = batch_norm_relu(
                branch, is_training, relu=False, data_format=data_format)
          elif layer.layer_type == search_proto.Layer.MAXPOOLCONV:
            branch = contrib_slim.max_pool3d(
                net, [layer.time, 3, 3],
                scope='MaxPool_0a_3x3',
                stride=1,
                padding='SAME')
            branch = conv3d(
                branch,
                filters, [1, 1, 1],
                scope='Conv2d_0b_1x1',
                data_format=data_format)
            branch = batch_norm_relu(
                branch, is_training, relu=False, data_format=data_format)
        branches.append(branch)
      net = tf.concat(branches, axis=axis)

    return tf.nn.relu(net + shortcut)

  def get_layer_type(self, layer_type):
    if layer_type == search_proto.Layer.CONV3D_DEFAULT:
      conv_op = conv3d
    elif layer_type == search_proto.Layer.TGM:
      conv_op = tgm_conv3d
    elif layer_type == search_proto.Layer.CONV2P1:
      conv_op = conv21d
    return conv_op

  def residual_block(self, net, filters, layers, scope, data_format, block):
    """Residual block: repeats a grouping of layers several times."""
    with tf.variable_scope(scope):
      with tf.variable_scope('BaseBlock'):
        net = self.block(
            net,
            filters,
            layers,
            scope=scope,
            data_format=data_format,
            use_projection=True)
      for i in range(1, block.repeats):
        # disable T for every other
        with tf.variable_scope('Block_' + str(i)):
          net = self.block(
              net,
              filters,
              layers,
              scope=scope,
              data_format=data_format,
              use_time=(i % 2) == 0)

    return net

  def model(self,
            video,
            mode,
            only_endpoints=False,
            final_endpoint=''):
    """Create the model graph.

    Args:
      video: a BxTxHxWxC video tensor
      mode: string,  train or eval
      only_endpoints: Whether to return only the endpoints.
      final_endpoint: Specifies the endpoint to construct the network up to.
          If not specified, the entire network is constructed and returned.
          Only used if only_endpoints is True.

    Returns:
      loss, accuracy and logits, or endpoints
    """
    self.is_training = (mode == 'train')
    is_training = self.is_training
    data_format = self.data_format

    endpoints = {}

    def add_and_check_endpoint(net, endpoint):
      endpoints[endpoint] = net
      return only_endpoints and final_endpoint == endpoint

    with contrib_slim.arg_scope([contrib_slim.conv2d], padding='SAME'):
      with tf.variable_scope('VidIncRes', 'VidIncRes', [video]):
        with contrib_slim.arg_scope(
            [contrib_slim.batch_norm, contrib_slim.dropout],
            is_training=is_training):
          net = video

          conv_op = self.get_layer_type(self.spec.convop1)
          net = conv_op(
              net,
              64, [self.spec.time1, 7, 7],
              strides=[2, 2, 2],
              scope='Conv2d_1a_7x7',
              dilation=self.spec.dilation)
          net = batch_norm_relu(
              net, is_training, relu=True, data_format=data_format)
          if add_and_check_endpoint(net, 'Conv2d_1a_7x7'):
            return endpoints

          net = contrib_slim.max_pool3d(
              net, [self.spec.max_pool1_time, 3, 3],
              stride=[2, 2, 2],
              scope='maxpool1',
              padding='SAME')
          if add_and_check_endpoint(net, 'maxpool1'):
            return endpoints

          net = self.residual_block(
              net=net,
              filters=4 * 64,
              layers=self.spec.blocks[0].layers,
              scope='res_block_2',
              data_format=data_format,
              block=self.spec.blocks[0])
          if add_and_check_endpoint(net, 'res_block_2'):
            return endpoints
          net = contrib_slim.max_pool3d(
              net, [self.spec.max_pool1_time, 2, 2],
              stride=[1, 2, 2],
              scope='maxpool2',
              padding='SAME')
          if add_and_check_endpoint(net, 'maxpool2'):
            return endpoints

          net = self.residual_block(
              net,
              4 * 128,
              self.spec.blocks[1].layers,
              scope='res_block_3',
              data_format=data_format,
              block=self.spec.blocks[1])
          if add_and_check_endpoint(net, 'res_block_3'):
            return endpoints
          net = contrib_slim.max_pool3d(
              net, [self.spec.max_pool3_time, 2, 2],
              stride=[1, 2, 2],
              scope='maxpool3',
              padding='SAME')
          if add_and_check_endpoint(net, 'maxpool3'):
            return endpoints

          net = self.residual_block(
              net,
              filters=4 * 256,
              layers=self.spec.blocks[2].layers,
              scope='res_block_4',
              data_format=data_format,
              block=self.spec.blocks[2])
          if add_and_check_endpoint(net, 'res_block_4'):
            return endpoints
          net = contrib_slim.max_pool3d(
              net, [self.spec.max_pool4_time, 2, 2],
              stride=[1, 2, 2],
              scope='maxpool4',
              padding='SAME')
          if add_and_check_endpoint(net, 'maxpool4'):
            return endpoints

          net = self.residual_block(
              net,
              4 * 512,
              self.spec.blocks[3].layers,
              scope='res_block_5',
              data_format=data_format,
              block=self.spec.blocks[3])
          if add_and_check_endpoint(net, 'res_block_5'):
            return endpoints
          # Adds one more endpoint denoting the last cell before logits.
          if add_and_check_endpoint(net, 'LastCell'):
            return endpoints

          with tf.variable_scope('Logits'):
            shape = net.get_shape().as_list()
            s = shape[3]
            pool_size = (min(
                shape[1] if data_format == 'channels_last' else shape[2], 2), s,
                         s)
            net = contrib_slim.avg_pool3d(
                inputs=net, kernel_size=pool_size, stride=1, padding='VALID')
            net = contrib_slim.dropout(
                net,
                self.dropout_keep_prob,
                scope='Dropout_0b',
                is_training=is_training)
            net = contrib_slim.conv3d(
                net,
                self.num_classes,
                kernel_size=1,
                stride=1,
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=contrib_slim_initializers
                .variance_scaling_initializer(
                    factor=2.0, mode='FAN_IN', uniform=False))
            # spatial-temporal pooling
            logits = tf.reduce_mean(
                net,
                axis=([1, 2, 3]
                      if data_format == 'channels_last' else [2, 3, 4]))
            if add_and_check_endpoint(logits, 'Logits'):
              return endpoints

    pred = tf.argmax(contrib_slim.softmax(logits), axis=1)
    if add_and_check_endpoint(pred, 'Predictions'):
      return endpoints

    if only_endpoints:
      return endpoints

    return logits
