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

"""Contains definitions for the AssembleNet++ [2] models (without object input).

Requires the AssembleNet++ architecture to be specified in
FLAGS.model_structure (and optionally FLAGS.model_edge_weights). This is
identical to the form described in assemblenet.py for the AssembleNet. Please
check assemblenet.py for the detailed format of the model strings.

AssembleNet++ adds `peer-attention' to the basic AssembleNet, which allows each
conv. block connection to be conditioned differently based on another block [2].
It is a form of channel-wise attention. Note that we learn to apply attention
independently for each frame.

The `peer-attention' implementation in this file is the version that enables
one-shot differentiable search of attention connectivity (Fig. 2 in [2]), using
a softmax weighted summation of possible attention vectors.

[2] Michael S. Ryoo, AJ Piergiovanni, Juhana Kangaspunta, Anelia Angelova,
    AssembleNet++: Assembling Modality Representations via Attention
    Connections. ECCV 2020
    https://arxiv.org/abs/2008.08072

In order to take advantage of object inputs, one will need to set the flag
FLAGS.use_object_input as True, and provide the list of input tensors as an
input to the network, as shown in run_asn_with_object.py. This will require a
pre-processed object data stream.

It uses (2+1)D convolutions for video representations. The main AssembleNet++
takes a 4-D (N*T)HWC tensor as an input (i.e., the batch dim and time dim are
mixed), and it reshapes a tensor to NT(H*W)C whenever a 1-D temporal conv. is
necessary. This is to run this on TPU efficiently.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from assemblenet import assemblenet as asn
from assemblenet import rep_flow_2d_layer as rf

tf.enable_v2_tensorshape()

FLAGS = flags.FLAGS


def softmax_merge_peer_attentions(peers, data_format):
  """Merge multiple peer-attention vectors with softmax weighted sum.

  Summation weights are to be learned.

  Args:
    peers: A list of `Tensors` of size `[batch*time, channels]`.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last for `[batch*time, height, width,
      channels]`. Only works for "channels_last" currently.
  Returns:
    The output `Tensor` of size `[batch*time, channels].
  """
  assert data_format == 'channels_last'

  if FLAGS.precision == 'bfloat16':
    dtype = tf.bfloat16
  else:
    dtype = tf.float32

  initial_attn_weights = lambda: tf.random.truncated_normal(  # pylint: disable=g-long-lambda
      [len(peers)],
      mean=0.0,
      stddev=0.01,
      dtype=tf.float32)
  attn_weights = tf.Variable(
      initial_value=initial_attn_weights, trainable=True,
      name='peer_attention_weights', dtype=tf.float32)
  attn_weights = tf.nn.softmax(tf.cast(attn_weights, dtype))

  weighted_peers = []
  for i, peer in enumerate(peers):
    weighted_peers.append(attn_weights[i] * peer)

  return tf.add_n(weighted_peers)


def apply_attention(inputs,
                    attention_mode=None,
                    attention_in=None,
                    use_5d_mode=False,
                    data_format='channels_last'):
  """Applies peer-attention or self-attention to the input tensor.

  Depending on the attention_mode, this function either applies channel-wise
  self-attention or peer-attention. For the peer-attention, the function
  combines multiple candidate attention vectors (given as attention_in), by
  learning softmax-sum weights described in the AssembleNet++ paper. Note that
  the attention is applied individually for each frame, which showed better
  accuracies than using video-level attention.

  Args:
    inputs: A `Tensor`. Either 4D or 5D, depending of use_5d_mode.
    attention_mode: `str` specifying mode. If not `peer', does self-attention.
    attention_in: A list of `Tensors' of size [batch*time, channels].
    use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.
    data_format: `str`. Only works for "channels_last" currently. If use_5d_mode
      is True, its shape is `[batch, time, height, width, channels]`. Otherwise
      `[batch*time, height, width, channels]`.
  Returns:
    The output `Tensor` after concatenation.
  """
  assert data_format == 'channels_last'

  h_ch_loc = 2 if use_5d_mode else 1

  if attention_mode == 'peer':
    attn = softmax_merge_peer_attentions(attention_in, data_format)
  else:
    attn = tf.reduce_mean(inputs, [h_ch_loc, h_ch_loc+1])
  attn = tf.layers.dense(
      inputs=attn,
      units=inputs.shape[-1],
      kernel_initializer=tf.random_normal_initializer(stddev=.01))
  attn = tf.math.sigmoid(attn)
  channel_attn = tf.expand_dims(tf.expand_dims(attn, h_ch_loc), h_ch_loc)

  inputs = tf.multiply(inputs, channel_attn)

  return inputs


def fusion_with_peer_attention(inputs,
                               index=None,
                               attention_mode=None,
                               attention_in=None,
                               use_5d_mode=False,
                               data_format='channels_last'):
  """Weighted summation of multiple tensors, while using peer-attention.

  Summation weights are to be learned. Uses spatial max pooling and 1x1 conv.
  to match their sizes. Before the summation, each connection (i.e., each input)
  itself is scaled with channel-wise peer-attention. Notice that attention is
  applied for each connection, conditioned based on attention_in.

  Args:
    inputs: A list of `Tensors`. Either 4D or 5D, depending of use_5d_mode.
    index: `int` index of the block within the AssembleNet architecture. Used
      for summation weight initial loading.
    attention_mode: `str` specifying mode. If not `peer', does self-attention.
    attention_in: A list of `Tensors' of size [batch*time, channels].
    use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.
    data_format: `str`. Only works for "channels_last" currently. If use_5d_mode
      is True, its shape is `[batch, time, height, width, channels]`. Otherwise
      `[batch*time, height, width, channels]`.
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
    conv_function = asn.conv3d_same_padding
  else:
    h_channel_loc = 1
    conv_function = asn.conv2d_fixed_padding

  # If only 1 input. Apply peer-attention to the connection when used.
  if len(inputs) == 1:
    if attention_mode:
      inputs[0] = apply_attention(inputs[0],
                                  attention_mode,
                                  attention_in,
                                  use_5d_mode,
                                  data_format)
    return inputs[0]

  # get smallest spatial size and largest channels
  sm_size = [10000, 10000]
  lg_channel = 0
  for inp in inputs:
    # assume batch X height x width x channels
    sm_size[0] = min(sm_size[0], inp.shape[h_channel_loc])
    sm_size[1] = min(sm_size[1], inp.shape[h_channel_loc+1])
    # Note that, when using object inputs, object channel sizes are usually big.
    # Since we do not want the object channel size to increase the number of
    # parameters for every fusion, we exclude it when computing lg_channel.
    if inp.shape[-1] > lg_channel and inp.shape[-1] != FLAGS.num_object_classes:  # pylint: disable=line-too-long
      lg_channel = inp.shape[3]

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
    input_shape = inp.shape
    if input_shape[h_channel_loc] != sm_size[0] or input_shape[h_channel_loc+1] != sm_size[1]:  # pylint: disable=line-too-long
      assert sm_size[0] != 0
      ratio = (input_shape[h_channel_loc] + 1) // sm_size[0]
      if use_5d_mode:
        inp = tf.layers.max_pooling3d(
            inp, [1, ratio, ratio], [1, ratio, ratio],
            padding='same', data_format=data_format)
      else:
        inp = tf.layers.max_pooling2d(
            inp, [ratio, ratio], ratio, padding='same', data_format=data_format)

    if input_shape[-1] in per_channel_inps:
      per_channel_inps[input_shape[-1]].append(weights[i] * inp)
    else:
      per_channel_inps.update({input_shape[-1]: [weights[i] * inp]})

  # Implementation of connectivity with peer-attention
  if attention_mode:
    for key, channel_inps in per_channel_inps.items():
      for idx in range(len(channel_inps)):
        with tf.variable_scope('Connection_' + str(key) + '_' + str(idx)):
          channel_inps[idx] = apply_attention(channel_inps[idx],
                                              attention_mode,
                                              attention_in,
                                              use_5d_mode,
                                              data_format)

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


def object_conv_stem(inputs,
                     data_format='channels_last'):
  """Layers for an object input stem.

  It expects its input tensor to have a separate channel for each object class.

  Args:
    inputs: A `Tensor`.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last" for `[batch*time, height, width,
      channels]`.
  Returns:
    The output `Tensor`.
  """
  inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=4,
      strides=4,
      padding='SAME',
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_max_pool')

  return inputs


def assemblenet_plus_generator(block_fn,
                               layers,
                               num_classes,
                               data_format='channels_last'):
  """Generator for AssembleNet++ models.

  Args:
    block_fn: `function` for the block to use within the model. Currently only
      has `bottleneck_block_interleave as its option`.
    layers: list of 4 `int`s denoting the number of blocks to include in each of
      the 4 block groups. Each group consists of blocks that take inputs of the
      same resolution.
    num_classes: `int` number of possible classes for video classification.
    data_format: `str` either "channels_first" for `[batch*time, channels,
      height, width]` or "channels_last" for `[batch*time, height, width,
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

    if FLAGS.use_object_input:
      feature_shape = inputs[0].shape
      original_inputs = inputs[0]
      object_inputs = inputs[1]
    else:
      feature_shape = inputs.shape
      original_inputs = inputs
      object_inputs = None

    batch_size = feature_shape[0] // FLAGS.num_frames
    original_num_frames = FLAGS.num_frames
    num_frames = original_num_frames

    grouping = {-3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(len(structure)):
      grouping[structure[i][0]].append(i)

    stem_count = len(grouping[-3]) + len(grouping[-2]) + len(grouping[-1])

    assert stem_count != 0
    stem_filters = 128 // stem_count

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
          inputs = asn.rgb_conv_stem(original_inputs,
                                     original_num_frames,
                                     stem_filters,
                                     structure[i][1],
                                     is_training,
                                     data_format)
          streams.append(inputs)
        elif structure[i][0] == -2:
          inputs = asn.flow_conv_stem(flow_inputs,
                                      stem_filters,
                                      structure[i][1],
                                      is_training,
                                      data_format)
          streams.append(inputs)
        elif structure[i][0] == -3:
          # In order to use the object inputs, you need to feed your object
          # input tensor here.
          inputs = object_conv_stem(object_inputs,
                                    data_format)
          streams.append(inputs)
        else:
          block_number = structure[i][0]

          combined_inputs = [streams[structure[i][1][j]]
                             for j in range(0, len(structure[i][1]))]

          tf.logging.info(grouping)
          nodes_below = []
          for k in range(-3, structure[i][0]):
            nodes_below = nodes_below + grouping[k]

          peers = []
          if FLAGS.attention_mode:
            lg_channel = -1
            tf.logging.info(nodes_below)
            for k in nodes_below:
              tf.logging.info(streams[k].shape)
              lg_channel = max(streams[k].shape[3], lg_channel)

            for node_index in nodes_below:
              attn = tf.reduce_mean(streams[node_index], [1, 2])

              attn = tf.layers.dense(
                  inputs=attn,
                  units=lg_channel,
                  kernel_initializer=tf.random_normal_initializer(stddev=.01))
              peers.append(attn)

          combined_inputs = fusion_with_peer_attention(
              combined_inputs,
              index=i,
              attention_mode=FLAGS.attention_mode,
              attention_in=peers,
              use_5d_mode=False,
              data_format=data_format)

          graph = asn.block_group(
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

    outputs = asn.multi_stream_heads(streams,
                                     grouping[3],
                                     original_num_frames,
                                     num_classes,
                                     data_format)

    return outputs

  model.default_image_size = 224
  return model


def assemblenet_plus(assemblenet_depth,
                     num_classes,
                     data_format='channels_last'):
  """Returns the AssembleNet++ model for a given size and number of output classes."""

  assert data_format == 'channels_last'

  model_params = {
      26: {
          'block': asn.bottleneck_block_interleave,
          'layers': [2, 2, 2, 2]
      },
      38: {
          'block': asn.bottleneck_block_interleave,
          'layers': [2, 4, 4, 2]
      },
      50: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 6, 3]
      },
      68: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 12, 3]
      },
      77: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 15, 3]
      },
      101: {
          'block': asn.bottleneck_block_interleave,
          'layers': [3, 4, 23, 3]
      },
  }

  if assemblenet_depth not in model_params:
    raise ValueError('Not a valid assemblenet_depth:', assemblenet_depth)

  params = model_params[assemblenet_depth]
  return assemblenet_plus_generator(
      params['block'], params['layers'], num_classes, data_format)
