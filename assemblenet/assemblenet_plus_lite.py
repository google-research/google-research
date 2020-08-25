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

"""This  is the `lite' version of AssembleNet++ [2], which is smaller.

The entire experiments in AssembleNet/++ papers were done with TPUs (v3), and
due to different memory optimization in TPUs and GPUs, directly fitting the
AssembleNet/++ models on GPUs with a large enough batch size (e.g., 8 per GPU)
has been challengeing. Thus, we are introducing this lite version of
AssembleNet++ which we confirmed that a single GPU (with 16GB memory) can host
a batch of 8 videos with 32 frames for the training.

Specifically, it is modified to use smaller modules as its building block; it
adopts an inverted bottleneck architecture, also used in MobileNet V2 [3] and
V3 [4], as well as X3D [5]. It also uses 3D residual modules instead of the
(2+1)D ResNet modules originally used in AssembleNet and AssembleNet+, to save
TPU/GPU memory. More details of this version are described in the supplementary
materials of [2].

Also notice that this version is without object inputs described in [2]. The one
will need to slightly modify the code to also provide object segmentation
inputs.

Requires the AssembleNet++ architecture to be specified in
FLAGS.model_structure (and optionally FLAGS.model_edge_weights). This is
identical to the form described in assemblenet.py for the AssembleNet. Please
check assemblenet.py for the detailed format of the model strings.

AssembleNet++ adds `peer-attention' to the basic AssembleNet, which allows each
conv. block connection to be conditioned differently based on another block [2].
It is a form of channel-wise attention. Note that we learn to apply attention
independently for each frame.

The `peer-attention' implementation in this file is the version that enables
one-shot differentiable search of attention connectivity (Fig. 2 in [2]).

[2] Michael S. Ryoo, AJ Piergiovanni, Juhana Kangaspunta, Anelia Angelova,
    AssembleNet++: Assembling Modality Representations via Attention
    Connections. ECCV 2020
    https://arxiv.org/abs/2008.08072

[3] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh
    Chen, MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018

[4] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing
    Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le,
    Hartwig Adam, Searching for MobileNetV3. ICCV 2019

[5] Christoph Feichtenhofer, X3D: Expanding Architectures for Efficient Video
    Recognition. CVPR 2020

It uses 3D convolutions for video representations. The main AssembleNet++lite
takes a 5-D NDHWC tensor as an input, where D is the time dimension.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math

from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from assemblenet import assemblenet as asn
from assemblenet import assemblenet_plus as asnp
from assemblenet import rep_flow_2d_layer as rf

tf.enable_v2_tensorshape()

FLAGS = flags.FLAGS


def conv1d_bn(inputs,
              is_training,
              filters,
              kernel_size,
              temporal_dilation=1,
              data_format='channels_last'):
  """Performs 1D temporal conv. followed by batch normalization, using conv3d.

  Args:
    inputs: `Tensor` of size `[batch, time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    is_training: `bool` specifying whether in training mode or not.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` kernel size to be used for `conv3d`
      operations. Should be a positive integer.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    data_format: `str`. Only supports "channels_last" as the data format.

  Returns:
     The output `Tensor` after 1D conv and batch normalization.
  """
  if FLAGS.precision == 'bfloat16':
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  if temporal_dilation == 1:
    inputs = tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=(kernel_size, 1, 1),
        strides=1,
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)
  else:
    # Dilated 1D conv was manually implemented to save computation time for TPU.
    # Using tf.layes.conv3d with dilation parameter might be better for GPU.
    assert data_format == 'channels_last'
    feature_shape = inputs.shape

    initial_filter_values = lambda: tf.random.truncated_normal(  # pylint: disable=g-long-lambda
        [kernel_size, 1, 1, feature_shape[-1], filters],
        mean=0.0,
        stddev=math.sqrt(2.0 / (kernel_size * feature_shape[-1])),
        dtype=tf.float32)
    filter_values = tf.Variable(
        initial_value=initial_filter_values, dtype=tf.float32)
    filter_values = tf.cast(filter_values, dtype)
    filter_value_list = tf.unstack(filter_values, axis=0)

    for i in range(len(filter_value_list) - 1):
      for _ in range(temporal_dilation - 1):
        zeros = tf.zeros([1, 1, feature_shape[-1], filters], dtype=dtype)
        filter_value_list.insert(i * temporal_dilation + 1, zeros)

    filter_values = tf.stack(filter_value_list, axis=0)
    inputs = tf.nn.conv3d(
        inputs, filter_values, strides=[1, 1, 1, 1, 1], padding='SAME')

  inputs = rf.batch_norm_relu(inputs, is_training,
                              bn_decay=FLAGS.bn_decay,
                              bn_epsilon=FLAGS.bn_epsilon,
                              data_format=data_format)

  return inputs


def inverted_bottleneck_3dblock(inputs,
                                filters,
                                is_training,
                                strides,
                                use_projection=False,
                                temporal_dilation=1,
                                data_format='channels_last'):
  """Inverted bottleneck residual block with a 3D conv layer.

  Inverted bottleneck block variant for 3D residual networks with BN after
  convolutions. It uses (2+1)D conv instead when striding is needed.

  Args:
    inputs: 5D `Tensor` following the data_format.
    filters: `List` of `int` number of filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
      downsample the input spatially.
    use_projection: `bool` for whether this block should use a projection
      shortcut (versus the default identity shortcut). This is usually `True`
      for the first block of a block group, which may change the number of
      filters and the resolution.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    data_format: `str` either "channels_first" for `[batch, time, channels,
      height, width]` or "channels_last for `[batch, time, height, width,
      channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.

    shortcut = asn.conv3d_same_padding(
        inputs=inputs,
        filters=filters[-1],
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = rf.batch_norm_relu(
        shortcut, is_training, relu=False,
        bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
        data_format=data_format)

  inputs = asn.conv3d_same_padding(
      inputs=inputs,
      filters=filters[0],
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = rf.batch_norm_relu(inputs, is_training,
                              bn_decay=FLAGS.bn_decay,
                              bn_epsilon=FLAGS.bn_epsilon,
                              data_format=data_format)

  if strides > 1:
    inputs = asn.conv3d_same_padding(
        inputs=inputs,
        filters=filters[1],
        kernel_size=3,
        strides=strides,
        do_2d_conv=True,
        data_format=data_format)
    inputs = rf.batch_norm_relu(inputs, is_training,
                                bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,  # pylint: disable=line-too-long
                                data_format=data_format)
    inputs = conv1d_bn(inputs,
                       is_training,
                       filters=filters[1],
                       kernel_size=3,
                       temporal_dilation=temporal_dilation,
                       data_format=data_format)
  else:
    inputs = asn.conv3d_same_padding(
        inputs=inputs,
        filters=filters[1],
        kernel_size=[3, 3, 3],
        strides=1,
        temporal_dilation=temporal_dilation,
        data_format=data_format)
    inputs = rf.batch_norm_relu(inputs, is_training,
                                bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,  # pylint: disable=line-too-long
                                data_format=data_format)

  inputs = asn.conv3d_same_padding(
      inputs=inputs,
      filters=filters[-1],
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = rf.batch_norm_relu(
      inputs, is_training, relu=False, init_zero=True,
      bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training,
                name,
                temporal_dilation=1,
                data_format='channels_last'):
  """Creates one group of blocks for the AssembleNet model.

  Args:
    inputs: `Tensor` of size `[batch, time, height, width, channels]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
      greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str` name for the Tensor output of the block layer.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    data_format: `str` either "channels_first" for `[batch, channels, time,
      height, width]` or "channels_last" for `[batch, time, height, width,
      channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      is_training,
      strides=strides,
      use_projection=True,
      temporal_dilation=temporal_dilation,
      data_format=data_format)

  for _ in range(1, blocks):
    inputs = block_fn(
        inputs,
        filters,
        is_training,
        strides=1,
        use_projection=False,
        temporal_dilation=temporal_dilation,
        data_format=data_format)

  return tf.identity(inputs, name)


def lite_conv_stem(inputs,
                   filters,
                   temporal_dilation=1,
                   is_training=False,
                   data_format='channels_last'):
  """Layers for a RGB or optical flow stem, using 2D + 1D conv layers.

  Args:
    inputs: A list of `Tensors` of size `[batch*time, channels, height, width]`.
    filters: `int` number of filters in the convolution.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    is_training: `bool` specifying whether in training mode or not.
    data_format: `str`. Only supports "channels_last" as the data format.
  Returns:
    The output `Tensor`.
  """
  assert data_format == 'channels_last'

  if temporal_dilation < 1:
    temporal_dilation = 1

  inputs = asn.conv3d_same_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=2,
      do_2d_conv=True,
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_conv')
  inputs = rf.batch_norm_relu(
      inputs, is_training,
      bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
      data_format=data_format)

  inputs = conv1d_bn(
      inputs=inputs,
      is_training=is_training,
      filters=filters,
      kernel_size=3,
      temporal_dilation=temporal_dilation,
      data_format=data_format)

  return inputs


def lite_one_stream_head(inputs,
                         num_classes,
                         is_training=False,
                         data_format='channels_last'):
  """Layers for one classification head.

  Args:
    inputs: A 4D `Tensor` following the data_format.
    num_classes: `int` number of possible classes for video classification.
    is_training: `bool` specifying whether in training mode or not.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last for `[batch*time, height, width,
      channels]`. Only works for 'channels_last' currently.
  Returns:
    The output `Tensor`.
  """
  assert data_format == 'channels_last'

  batch_size = inputs.shape[0]
  num_frames = inputs.shape[1]

  inputs = asn.conv3d_same_padding(
      inputs=inputs,
      filters=432,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = tf.identity(inputs, 'last_conv')
  inputs = rf.batch_norm_relu(
      inputs, is_training,
      bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
      data_format=data_format)

  inputs = asn.conv3d_same_padding(
      inputs=inputs,
      filters=2048,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = tf.identity(inputs, 'last_conv2')
  inputs = rf.batch_norm_relu(
      inputs, is_training,
      bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
      data_format=data_format)

  if not FLAGS.max_pool_preditions:
    pool_size = (inputs.shape[1], inputs.shape[2], inputs.shape[3])
    inputs = tf.layers.average_pooling3d(
        inputs=inputs,
        pool_size=pool_size,
        strides=1,
        padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    inputs = tf.reshape(inputs, [batch_size, -1])
  else:
    pool_size = (1, inputs.shape[2], inputs.shape[3])
    inputs = tf.layers.average_pooling3d(
        inputs=inputs,
        pool_size=pool_size,
        strides=1,
        padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    inputs = tf.reshape(inputs, [batch_size, num_frames, -1])

  if FLAGS.dropout_keep_prob:
    inputs = tf.keras.layers.Dropout(FLAGS.dropout_keep_prob)(inputs, training=is_training)  # pylint: disable=line-too-long

  outputs = tf.layers.dense(
      inputs=inputs,
      units=num_classes,
      kernel_initializer=tf.random_normal_initializer(stddev=.01))
  outputs = tf.identity(outputs, 'final_dense')

  if FLAGS.max_pool_preditions:
    pre_logits = outputs / np.sqrt(num_frames)
    acts = tf.nn.softmax(pre_logits, axis=1)
    outputs = tf.math.multiply(outputs, acts)

    outputs = tf.reduce_sum(outputs, 1)

  return outputs


def assemblenet_lite_generator(block_fn,
                               layers,
                               num_classes,
                               data_format='channels_last'):
  """Generator for AssembleNet++lite models, while using 5D BDHWC tensors.

  Args:
    block_fn: `function` for the block to use within the model. Currently only
      has `bottleneck_block_interleave as its option`.
    layers: list of 4 `int`s denoting the number of blocks to include in each of
      the 4 block groups. Each group consists of blocks that take inputs of the
      same resolution.
    num_classes: `int` number of possible classes for video classification.
    data_format: `str` either "channels_first" for `[batch, channels, time,
      height, width]` or "channels_last" for `[batch, time, height, width,
      channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the AssembleNet model.
  """

  def model(inputs, is_training):
    """Creation of the model graph."""

    tf.logging.info(FLAGS.model_structure)
    tf.logging.info(FLAGS.model_edge_weights)
    structure = json.loads(FLAGS.model_structure)

    feature_shape = inputs.shape
    batch_size = feature_shape[0]
    original_num_frames = feature_shape[1]

    grouping = {-3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(len(structure)):
      grouping[structure[i][0]].append(i)

    stem_count = len(grouping[-3]) + len(grouping[-2]) + len(grouping[-1])

    assert stem_count != 0
    stem_filters = 24

    original_inputs = inputs
    if grouping[-2]:
      # Instead of loading optical flows as inputs from data pipeline, we are
      # applying the "Representation Flow" to RGB frames so that we can compute
      # the flow within TPU/GPU on fly. It's essentially optical flow since we
      # do it with RGBs.
      # We use reshaping for Represeentation Flow computation for speed.
      flow_inputs = tf.reshape(original_inputs,
                               [batch_size * original_num_frames,
                                feature_shape[2], feature_shape[3], -1])
      flow_inputs = rf.rep_flow(
          flow_inputs,
          batch_size,
          original_num_frames,
          num_iter=20,
          is_training=is_training,
          bottleneck=1,
          scope='rep_flow')
      flow_inputs = tf.reshape(flow_inputs,
                               [batch_size, original_num_frames,
                                feature_shape[2], feature_shape[3], -1])
    streams = []

    for i in range(len(structure)):
      with tf.variable_scope('Node_' + str(i)):
        if structure[i][0] == -1:
          inputs = lite_conv_stem(original_inputs,
                                  stem_filters,
                                  structure[i][1],
                                  is_training,
                                  data_format)
          streams.append(inputs)
        elif structure[i][0] == -2:
          inputs = lite_conv_stem(flow_inputs,
                                  stem_filters,
                                  structure[i][1],
                                  is_training,
                                  data_format)
          streams.append(inputs)
        elif structure[i][0] == -3:
          object_inputs = None
          # In order to use the object inputs, you need to feed your object
          # input tensor here and modify the stem function.
          inputs = lite_conv_stem(object_inputs,
                                  stem_filters,
                                  structure[i][1],
                                  is_training,
                                  data_format)
        else:
          block_number = structure[i][0]

          combined_inputs = [streams[structure[i][1][j]]
                             for j in range(0, len(structure[i][1]))]

          nodes_below = []
          for k in range(-3, structure[i][0]):
            nodes_below = nodes_below + grouping[k]

          peers = []
          if FLAGS.attention_mode:
            lg_channel = -1
            for k in nodes_below:
              lg_channel = max(streams[k].shape[-1], lg_channel)
            for node_index in nodes_below:
              attn = tf.reduce_mean(streams[node_index], [2, 3])

              attn = tf.layers.dense(
                  inputs=attn,
                  units=lg_channel,
                  kernel_initializer=tf.random_normal_initializer(stddev=.01))
              peers.append(attn)

          combined_inputs = asnp.fusion_with_peer_attention(
              combined_inputs,
              index=i,
              attention_mode=FLAGS.attention_mode,
              attention_in=peers,
              use_5d_mode=True,
              data_format='channels_last')

          graph = block_group(
              inputs=combined_inputs,
              filters=structure[i][2],
              block_fn=block_fn,
              blocks=layers[block_number],
              strides=structure[i][4],
              is_training=is_training,
              name='block_group' + str(i),
              temporal_dilation=structure[i][3],
              data_format=data_format)

          streams.append(graph)

    outputs = lite_one_stream_head(streams[grouping[3][0]],
                                   num_classes,
                                   is_training,
                                   data_format)

    return outputs

  model.default_image_size = 224
  return model


def assemblenet_plus_lite(num_layers,
                          num_classes=339,
                          data_format='channels_last'):
  """Returns the AssembleNet++lite model for a given size and number of output classes."""

  return assemblenet_lite_generator(
      inverted_bottleneck_3dblock, num_layers, num_classes, data_format)
