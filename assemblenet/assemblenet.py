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

"""Contains definitions for the AssembleNet [1] models.

Requires the AssembleNet architecture to be specified in
FLAGS.model_structure (and optionally FLAGS.model_edge_weights).
This structure is a list corresponding to a graph representation of the
network, where a node is a convolutional block and an edge specifies a
connection from one block to another as described in [1].

Each node itself (in the structure list) is a list with the following format:
[block_level, [list_of_input_blocks], number_filter, temporal_dilation,
spatial_stride]. [list_of_input_blocks] should be the list of node indexes whose
values are less than the index of the node itself. The 'stems' of the network
directly taking raw inputs follow a different node format:
[stem_type, temporal_dilation]. The stem_type is -1 for RGB stem and is -2 for
optical flow stem.

Also note that the codes in this file could be used for one-shot differentiable
connection search by (1) giving an overly connected structure as
FLAGS.model_structure and by (2) setting FLAGS.model_edge_weights to be '[]'.
The 'agg_weights' variables will specify which connections are needed and which
are not, once trained.

[1] Michael S. Ryoo, AJ Piergiovanni, Mingxing Tan, Anelia Angelova,
    AssembleNet: Searching for Multi-Stream Neural Connectivity in Video
    Architectures. ICLR 2020
    https://arxiv.org/abs/1905.13209

It uses (2+1)D convolutions for video representations. The main AssembleNet
takes a 4-D (N*T)HWC tensor as an input (i.e., the batch dim and time dim are
mixed), and it reshapes a tensor to NT(H*W)C whenever a 1-D temporal conv. is
necessary. This is to run this on TPU efficiently.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math

from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from assemblenet import rep_flow_2d_layer as rf

tf.enable_v2_tensorshape()

FLAGS = flags.FLAGS


intermediate_channel_size = [64, 128, 256, 512]


def topological_sort(structure):
  """Does the topological sorting of the given structure.

  Args:
    structure: A 'list' of the nodes, following the format described in
      architecture_graph.py.

  Returns:
    A list of ordered indexes.
  """
  structure = copy.deepcopy(structure)

  set_l = []
  set_s = []

  for i, node in enumerate(structure):
    if node[0] < 0:
      set_s.append(i)

  while set_s:
    index = set_s[0]
    del set_s[0]
    set_l.append(index)

    for i, node in enumerate(structure):
      if node[0] < 0:
        continue
      if index in node[1]:
        node[1].remove(index)
        if not node[1]:
          set_s.append(i)

  return set_l


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
      height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def reshape_temporal_conv1d_bn(inputs,
                               is_training,
                               filters,
                               kernel_size,
                               num_frames=32,
                               temporal_dilation=1,
                               data_format='channels_last'):
  """Performs 1D temporal conv. followed by batch normalization with reshaping.

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    is_training: `bool` specifying whether in training mode or not.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    num_frames: `int` number of frames in the input tensor.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    data_format: `str`. Only supports "channels_last" as the data format.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  if FLAGS.precision == 'bfloat16':
    dtype = tf.bfloat16
  else:
    dtype = tf.float32

  assert data_format == 'channels_last'

  feature_shape = inputs.shape

  tf.logging.info(inputs.shape)
  inputs = tf.reshape(inputs, [
      feature_shape[0] // num_frames, num_frames,
      feature_shape[1] * feature_shape[2], -1
  ])
  tf.logging.info(inputs.shape)

  if temporal_dilation == 1:
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=(kernel_size, 1),
        strides=1,
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)
  else:
    initial_filter_values = lambda: tf.random.truncated_normal(  # pylint: disable=g-long-lambda
        [kernel_size, 1, feature_shape[3], filters],
        mean=0.0,
        stddev=math.sqrt(2.0 / (kernel_size * feature_shape[3])),
        dtype=tf.float32)
    filter_values = tf.Variable(
        initial_value=initial_filter_values, dtype=tf.float32)
    filter_values = tf.cast(filter_values, dtype)
    filter_value_list = tf.unstack(filter_values, axis=0)

    for i in range(len(filter_value_list) - 1):
      for _ in range(temporal_dilation - 1):
        zeros = tf.zeros([1, feature_shape[3], filters], dtype=dtype)
        filter_value_list.insert(i * temporal_dilation + 1, zeros)

    filter_values = tf.stack(filter_value_list, axis=0)
    inputs = tf.nn.conv2d(
        inputs, filter_values, strides=[1, 1, 1, 1], padding='SAME')

  inputs = tf.reshape(
      inputs, [feature_shape[0], feature_shape[1], feature_shape[2], -1])
  inputs = rf.batch_norm_relu(inputs, is_training,
                              bn_decay=FLAGS.bn_decay,
                              bn_epsilon=FLAGS.bn_epsilon,
                              data_format=data_format)

  return inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last'):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last" for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def conv3d_same_padding(inputs,
                        filters,
                        kernel_size,
                        strides,
                        temporal_dilation=1,
                        do_2d_conv=False,
                        data_format='channels_last'):
  """3D convolution layer wrapper.

  Uses conv3d function.

  Args:
    inputs: 5D `Tensor` following the data_format.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    do_2d_conv: `bool` indicating whether to do 2d conv. If false, do 3D conv.
    data_format: `str` either "channels_first" for `[batch, channels, time,
    height, width]` or "channels_last" for `[batch, time, height, width,
    channels]`.

  Returns:
    A `Tensor` of shape `[batch, time_in, height_in, width_in, channels]`.
  """
  if isinstance(kernel_size, int):
    if do_2d_conv:
      kernel_size = [1, kernel_size, kernel_size]
    else:
      kernel_size = [kernel_size, kernel_size, kernel_size]

  return tf.layers.conv3d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=[1, strides, strides],
      padding='SAME',
      dilation_rate=[temporal_dilation, 1, 1],
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def bottleneck_block_interleave(inputs,
                                filters,
                                inter_filters,
                                is_training,
                                strides,
                                use_projection=False,
                                num_frames=32,
                                temporal_dilation=1,
                                data_format='channels_last',
                                step=1):
  """Interleaves a standard 2D residual module and (2+1)D residual module.

  Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch*time, channels, height, width]`.
    filters: `int` number of filters for the first conv. layer. The last conv.
      layer will use 4 times as many filters.
    inter_filters: `int` number of filters for the second conv. layer.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
      downsample the input spatially.
    use_projection: `bool` for whether this block should use a projection
      shortcut (versus the default identity shortcut). This is usually `True`
      for the first block of a block group, which may change the number of
      filters and the resolution.
    num_frames: `int` number of frames in the input tensor.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last" for `[batch*time, height, width,
      channels]`.
    step: `int` to decide whether to put 2D module or (2+1)D module.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = rf.batch_norm_relu(
        shortcut, is_training, relu=False,
        bn_decay=FLAGS.bn_decay,
        bn_epsilon=FLAGS.bn_epsilon,
        data_format=data_format)

  if step % 2 == 1:
    k = 3

    inputs = reshape_temporal_conv1d_bn(
        inputs=inputs,
        is_training=is_training,
        filters=filters,
        kernel_size=k,
        num_frames=num_frames,
        temporal_dilation=temporal_dilation,
        data_format=data_format)
  else:
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)
    inputs = rf.batch_norm_relu(
        inputs,
        is_training,
        bn_decay=FLAGS.bn_decay,
        bn_epsilon=FLAGS.bn_epsilon,
        data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=inter_filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = rf.batch_norm_relu(inputs,
                              is_training,
                              bn_decay=FLAGS.bn_decay,
                              bn_epsilon=FLAGS.bn_epsilon,
                              data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = rf.batch_norm_relu(
      inputs,
      is_training,
      relu=False,
      init_zero=True,
      bn_decay=FLAGS.bn_decay,
      bn_epsilon=FLAGS.bn_epsilon,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training,
                name,
                block_level,
                num_frames=32,
                temporal_dilation=1,
                data_format='channels_last'):
  """Creates one group of blocks for the AssembleNett model.

  Args:
    inputs: `Tensor` of size `[batch*time, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
      greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str` name for the Tensor output of the block layer.
    block_level: `int` block level in AssembleNet.
    num_frames: `int` number of frames in the input tensor.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last" for `[batch*time, height, width,
      channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      intermediate_channel_size[block_level],
      is_training,
      strides,
      use_projection=True,
      num_frames=num_frames,
      temporal_dilation=temporal_dilation,
      data_format=data_format,
      step=0)

  for i in range(1, blocks):
    inputs = block_fn(
        inputs,
        filters,
        intermediate_channel_size[block_level],
        is_training,
        1,
        num_frames=num_frames,
        temporal_dilation=temporal_dilation,
        data_format=data_format,
        step=i)

  return tf.identity(inputs, name)


def spatial_resize_merge(input1, input2, data_format='channels_last'):
  """Concatenates two different sized tensors channel-wise.

  Args:
    input1: `Tensor` of size `[batch*time, channels, height, width]`.
    input2: `Tensor` of size `[batch*time, channels, height, width]`.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last for `[batch*time, height, width,
      channels]`. Only works for 'channels_last' currently.
  Returns:
    The output `Tensor` after concatenation.
  """
  assert data_format == 'channels_last'

  if input1.shape[1] < input2.shape[1]:
    ratio = (input2.shape[1] + 1) // input1.shape[1]
    input2 = tf.layers.max_pooling2d(
        input2, [ratio, ratio], ratio, padding='same', data_format=data_format)
  elif input1.shape[1] > input2.shape[1]:
    ratio = (input1.shape[1] + 1) // input2.shape[1]
    input1 = tf.layers.max_pooling2d(
        input1, [ratio, ratio], ratio, padding='same', data_format=data_format)

  return tf.concat([input1, input2], 3)


def spatial_resize_and_concat(inputs, data_format='channels_last'):
  """Concatenates multiple different sized tensors channel-wise.

  Args:
    inputs: A list of `Tensors` of size `[batch*time, channels, height, width]`.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last for `[batch*time, height, width,
      channels]`. Only works for 'channels_last' currently.
  Returns:
    The output `Tensor` after concatenation.
  """
  assert data_format == 'channels_last'

  # Do nothing if only 1 input
  if len(inputs) == 1:
    return inputs[0]
  if data_format != 'channels_last':
    return inputs

  # get smallest spatial size and largest channels
  sm_size = [1000, 1000]
  for inp in inputs:
    # assume batch X height x width x channels
    sm_size[0] = min(sm_size[0], inp.shape[1])
    sm_size[1] = min(sm_size[1], inp.shape[2])

  for i in range(len(inputs)):
    if inputs[i].shape[1] != sm_size[0] or inputs[i].shape[
        2] != sm_size[1]:
      ratio = (inputs[i].shape[1] + 1) // sm_size[0]
      inputs[i] = tf.layers.max_pooling2d(
          inputs[i], [ratio, ratio],
          ratio,
          padding='same',
          data_format=data_format)

  return tf.concat(inputs, 3)


def multi_connection_fusion(inputs,
                            index=None,
                            use_5d_mode=False,
                            data_format='channels_last'):
  """Do weighted summation of multiple different sized tensors.

  A weight is assigned for each connection (i.e., each input tensor), and their
  summation weights are learned. Uses spatial max pooling and 1x1 conv.
  to match their sizes.

  Args:
    inputs: A `Tensor`. Either 4D or 5D, depending of use_5d_mode.
    index: `int` index of the block within the AssembleNet architecture. Used
      for summation weight initial loading.
    use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.
    data_format: `str`. Only works for `channels_last' currently. If use_5d_mode
      is True, its shape is `[batch, time, height, width, channels]`. Otherwise
      `[batch*time, height, width, channels]`
  Returns:
    The output `Tensor` after concatenation.
  """
  assert data_format == 'channels_last'

  if FLAGS.precision == 'bfloat16':
    dtype = tf.bfloat16
  else:
    dtype = tf.float32

  if use_5d_mode:
    h_channel_loc = 2
    conv_function = conv3d_same_padding
  else:
    h_channel_loc = 1
    conv_function = conv2d_fixed_padding

  # If only 1 input.
  if len(inputs) == 1:
    return inputs[0]

  assert data_format == 'channels_last'

  # get smallest spatial size and largest channels
  sm_size = [10000, 10000]
  lg_channel = 0
  for inp in inputs:
    # assume batch X height x width x channels
    sm_size[0] = min(sm_size[0], inp.shape[h_channel_loc])
    sm_size[1] = min(sm_size[1], inp.shape[h_channel_loc+1])
    lg_channel = max(lg_channel, inp.shape[-1])

  # loads or creates weight variables to fuse multiple inputs
  weights_shape = [len(inputs)]
  if index is None or FLAGS.model_edge_weights == '[]':
    initial_weight_values = lambda: tf.random.truncated_normal(  # pylint: disable=g-long-lambda
        weights_shape,
        mean=0.0,
        stddev=0.01,
        dtype=tf.float32)
    weights = tf.Variable(
        initial_value=initial_weight_values, trainable=True,
        name='agg_weights', dtype=tf.float32)
  else:
    model_edge_weights = json.loads(FLAGS.model_edge_weights)
    initial_weights_after_sigmoid = np.asarray(
        model_edge_weights[index][0]).astype('float32')
    # Initial_weights_after_sigmoid is never 0, as the initial weights are
    # based the results of a successful connectivity search.
    initial_weights = -np.log(1. / initial_weights_after_sigmoid - 1.)

    weights = tf.Variable(
        initial_value=initial_weights, trainable=False,
        name='agg_weights', dtype=tf.float32)
  weights = tf.math.sigmoid(tf.cast(weights, dtype))

  # Compute weighted inputs. We group inputs with the same channels.
  per_channel_inps = dict({0: []})
  for i, inp in enumerate(inputs):
    if inp.shape[h_channel_loc] != sm_size[0] or inp.shape[h_channel_loc+1] != sm_size[1]:  # pylint: disable=line-too-long
      assert sm_size[0] != 0
      ratio = (inp.shape[h_channel_loc] + 1) // sm_size[0]
      if use_5d_mode:
        inp = tf.layers.max_pooling3d(
            inp, [1, ratio, ratio], [1, ratio, ratio],
            padding='same', data_format=data_format)
      else:
        inp = tf.layers.max_pooling2d(
            inp, [ratio, ratio], ratio, padding='same', data_format=data_format)

    if inp.shape[-1] in per_channel_inps:
      per_channel_inps[inp.shape[-1]].append(weights[i] * inp)
    else:
      per_channel_inps.update({inp.shape[-1]: [weights[i] * inp]})

  # Adding 1x1 conv layers (to match channel size) and fusing all inputs.
  # We add inputs with the same channels first before applying 1x1 conv to save
  # memory.
  inps = []
  for key, channel_inps in per_channel_inps.items():
    if len(channel_inps) < 1:
      continue
    if len(channel_inps) == 1:
      if key == lg_channel:
        inp = channel_inps[0]
      else:
        inp = conv_function(
            channel_inps[0],
            lg_channel,
            kernel_size=1,
            strides=1,
            data_format=data_format)
      inps.append(inp)
    else:
      if key == lg_channel:
        inp = tf.add_n(channel_inps)
      else:
        inp = conv_function(
            tf.add_n(channel_inps),
            lg_channel,
            kernel_size=1,
            strides=1,
            data_format=data_format)
      inps.append(inp)

  return tf.add_n(inps)


def rgb_conv_stem(inputs,
                  num_frames,
                  filters,
                  temporal_dilation,
                  is_training=False,
                  data_format='channels_last'):
  """Layers for a RGB stem.

  Args:
    inputs: A `Tensor` of size `[batch*time, height, width, channels]`.
    num_frames: `int` number of frames in the input tensor.
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

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=7,
      strides=2,
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_conv')
  inputs = rf.batch_norm_relu(
      inputs, is_training,
      bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
      data_format=data_format)

  inputs = reshape_temporal_conv1d_bn(
      inputs=inputs,
      is_training=is_training,
      filters=filters,
      kernel_size=5,
      num_frames=num_frames,
      temporal_dilation=temporal_dilation,
      data_format=data_format)

  inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=3,
      strides=2,
      padding='SAME',
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_max_pool')

  return inputs


def flow_conv_stem(inputs,
                   filters,
                   temporal_dilation,
                   is_training=False,
                   data_format='channels_last'):
  """Layers for an optical flow stem.

  Args:
    inputs: A `Tensor`.
    filters: `int` number of filters in the convolution.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    is_training: `bool` specifying whether in training mode or not.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last" for `[batch*time, height, width,
      channels]`.
  Returns:
    The output `Tensor`.
  """

  if temporal_dilation < 1:
    temporal_dilation = 1

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=7,
      strides=2,
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_conv')
  inputs = rf.batch_norm_relu(
      inputs, is_training,
      bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
      data_format=data_format)

  inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=2,
      strides=2,
      padding='SAME',
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_max_pool')

  return inputs


def multi_stream_heads(streams,
                       final_nodes,
                       num_frames,
                       num_classes,
                       data_format='channels_last'):
  """Layers for the classification heads.

  Args:
    streams: A list of 4D `Tensors` following the data_format.
    final_nodes: A list of `int` where classification heads will be added.
    num_frames: `int` number of frames in the input tensor.
    num_classes: `int` number of possible classes for video classification.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last for `[batch*time, height, width,
      channels]`. Only works for 'channels_last' currently.
  Returns:
    The output `Tensor`.
  """
  inputs = streams[final_nodes[0]]
  batch_size = inputs.shape[0] // num_frames

  # The activation is 7x7 so this is a global average pool.
  pool_size = (inputs.shape[1], inputs.shape[2])
  inputs = tf.layers.average_pooling2d(
      inputs=inputs,
      pool_size=pool_size,
      strides=1,
      padding='VALID',
      data_format=data_format)
  inputs = tf.identity(inputs, 'final_avg_pool0')

  inputs = tf.reshape(inputs, [batch_size, num_frames, -1])
  if not FLAGS.max_pool_preditions:
    outputs = tf.reduce_mean(inputs, 1)
  else:
    outputs = inputs

  for i in range(1, len(final_nodes)):
    inputs = streams[final_nodes[i]]

    # The activation is 7x7 so this is a global average pool.
    pool_size = (inputs.shape[1], inputs.shape[2])
    inputs = tf.layers.average_pooling2d(
        inputs=inputs,
        pool_size=pool_size,
        strides=1,
        padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool' + str(i))

    inputs = tf.reshape(inputs, [batch_size, num_frames, -1])
    if not FLAGS.max_pool_preditions:
      inputs = tf.reduce_mean(inputs, 1)

    outputs = outputs + inputs

  if len(final_nodes) > 1:
    outputs = outputs / len(final_nodes)

  outputs = tf.layers.dense(
      inputs=outputs,
      units=num_classes,
      kernel_initializer=tf.random_normal_initializer(stddev=.01))
  outputs = tf.identity(outputs, 'final_dense0')
  if FLAGS.max_pool_preditions:
    pre_logits = outputs / np.sqrt(num_frames)
    acts = tf.nn.softmax(pre_logits, axis=1)
    outputs = tf.math.multiply(outputs, acts)

    outputs = tf.reduce_sum(outputs, 1)

  return outputs


def assemblenet_v1_generator(block_fn,
                             layers,
                             num_classes,
                             combine_method='sigmoid',
                             data_format='channels_last'):
  """Generator for AssembleNet v1 models.

  Args:
    block_fn: `function` for the block to use within the model. Currently only
      has `bottleneck_block_interleave as its option`.
    layers: list of 4 `int`s denoting the number of blocks to include in each of
      the 4 block groups. Each group consists of blocks that take inputs of the
      same resolution.
    num_classes: `int` number of possible classes for video classification.
    combine_method: 'str' for the weighted summation to fuse different blocks.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last for `[batch*time, height, width,
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
    batch_size = feature_shape[0] // FLAGS.num_frames
    original_num_frames = FLAGS.num_frames

    grouping = {-3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(len(structure)):
      grouping[structure[i][0]].append(i)

    stem_count = len(grouping[-3]) + len(grouping[-2]) + len(grouping[-1])

    assert stem_count != 0
    stem_filters = 128 // stem_count

    original_inputs = inputs
    if grouping[-2]:
      # Instead of loading optical flows as inputs from data pipeline, we are
      # applying the "Representation Flow" to RGB frames so that we can compute
      # the flow within TPU/GPU on fly. It's essentially optical flow since we
      # do it with RGBs.
      flow_inputs = rf.rep_flow(
          original_inputs,
          batch_size,
          original_num_frames,
          num_iter=40,
          is_training=is_training,
          bottleneck=1,
          scope='rep_flow')
    streams = []

    for i in range(len(structure)):
      with tf.variable_scope('Node_' + str(i)):
        if structure[i][0] == -1:
          inputs = rgb_conv_stem(original_inputs, original_num_frames,
                                 stem_filters, structure[i][1], is_training,
                                 data_format)
          streams.append(inputs)
        elif structure[i][0] == -2:
          inputs = flow_conv_stem(flow_inputs,
                                  stem_filters, structure[i][1], is_training,
                                  data_format)
          streams.append(inputs)

        else:
          num_frames = original_num_frames
          block_number = structure[i][0]

          combined_inputs = []
          if combine_method == 'concat':
            combined_inputs = [streams[structure[i][1][j]]
                               for j in range(0, len(structure[i][1]))]

            combined_inputs = spatial_resize_and_concat(combined_inputs)

          else:
            combined_inputs = [streams[structure[i][1][j]]
                               for j in range(0, len(structure[i][1]))]

            combined_inputs = multi_connection_fusion(
                combined_inputs,
                index=i)

          graph = block_group(
              inputs=combined_inputs,
              filters=structure[i][2],
              block_fn=block_fn,
              blocks=layers[block_number],
              strides=structure[i][4],
              is_training=is_training,
              name='block_group' + str(i),
              block_level=structure[i][0],
              num_frames=num_frames,
              temporal_dilation=structure[i][3],
              data_format=data_format)

          streams.append(graph)

    outputs = multi_stream_heads(streams,
                                 grouping[3],
                                 original_num_frames,
                                 num_classes,
                                 data_format)

    return outputs

  model.default_image_size = 224
  return model


def assemblenet_v1(assemblenet_depth,
                   num_classes,
                   combine_method='sigmoid',
                   data_format='channels_last'):
  """Returns the AssembleNet model for a given size and number of output classes."""

  assert data_format == 'channels_last'

  model_params = {
      26: {
          'block': bottleneck_block_interleave,
          'layers': [2, 2, 2, 2]
      },
      38: {
          'block': bottleneck_block_interleave,
          'layers': [2, 4, 4, 2]
      },
      50: {
          'block': bottleneck_block_interleave,
          'layers': [3, 4, 6, 3]
      },
      68: {
          'block': bottleneck_block_interleave,
          'layers': [3, 4, 12, 3]
      },
      77: {
          'block': bottleneck_block_interleave,
          'layers': [3, 4, 15, 3]
      },
      101: {
          'block': bottleneck_block_interleave,
          'layers': [3, 4, 23, 3]
      },
  }

  if assemblenet_depth not in model_params:
    raise ValueError('Not a valid assemblenet_depth:', assemblenet_depth)

  params = model_params[assemblenet_depth]
  return assemblenet_v1_generator(
      params['block'], params['layers'], num_classes, combine_method,
      data_format)
