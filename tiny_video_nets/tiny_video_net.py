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

"""Contains definitions for TinyVideoNets.

Generates a network based on the provided string.

Example Tiny Video Net Structure:
 {img_size:224, num_frames:32, frame_stride:2, blocks:[
     {repeats:1, spatial_kernel:3, temporal_kernel:3,
      spatial_type:std,
      temporal_type:1d,
      filters:32, spatial_stride:2, temporal_stride:2, expand:6, skip:0,
      squeeze:0.25, non_local:32, context_gate:0} ]}

      spatial_type:std/depth/maxpool/avgpool
      temporal_type:1d/maxpool/avgpool
      skip:0 or 1, use a skip connection in the layer
      squeeze: in [0,1] ratio for squeeze and excite layer
      non_local: 0 to disable, int of number of filters in bottleneck if used
      context_gate: 0 (disable) or 1 (enable)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import flags

import tensorflow.compat.v1 as tf

from tiny_video_nets import layers

FLAGS = flags.FLAGS


def tiny_block(block,
               inputs,
               num_frames,
               is_training,
               data_format='channels_last',
               strides=0,
               use_temporal=True):
  """Generates a block for TinyVideoNet based on the parameters.

  Args:
    block: dict for the block options (see top for block options)
    inputs: tensor [batch*frames, height, width, channels]
    num_frames: int number of frames currently in video
    is_training: bool
    data_format: string, channels_last or channels_first
    strides: int number of strides
    use_temporal: bool, indicates whether to use the temporal conv layer or not

  Returns:
    Tensor output and int number of frames
  """
  expanded_filters = max(block['filters'] * block['expand'], block['filters'])
  res = inputs
  if 'act' not in block:
    # Default to relu.
    block['act'] = 'relu'
    block['spatial_act'] = 'relu'
    block['temporal_act'] = 'relu'
  if 'inv-bottle' not in block:
    block['inv-bottle'] = False

  # Spatial-temporal layers.
  new_frames = num_frames
  with tf.variable_scope('STConvs'):
    if block['inv-bottle'] and block['expand'] > 0:
      # Expand channels.
      with tf.variable_scope('Expand'):
        inputs = layers.conv2d(
            inputs,
            1,
            expanded_filters,
            1,
            is_training,
            data_format=data_format,
            use_relu=True,
            init_zero=False)

    filters = expanded_filters if block['inv-bottle'] else block['filters']
    sp_strides = int(block['spatial_stride']) if strides == 0 else strides
    inputs = layers.spatial_conv(
        inputs,
        block['spatial_type'],
        block['spatial_kernel'],
        filters,
        sp_strides,
        is_training,
        activation_fn=block['spatial_act'],
        data_format=data_format)

    # Temporal layer, if chosen.
    if use_temporal and num_frames > 1:
      t_strides = block['temporal_stride'] if strides == 0 else strides
      inputs, new_frames = layers.temporal_conv(
          inputs,
          block['temporal_type'],
          block['temporal_kernel'],
          filters,
          t_strides,
          is_training,
          num_frames,
          activation_fn=block['temporal_act'],
          data_format=data_format)

  use_skip = (block['skip'] != 0 and inputs.shape[0] == res.shape[0])

  # Expand channels for inverted bottleneck.
  if not block['inv-bottle'] and block['expand'] > 0:
    with tf.variable_scope('Expand'):
      inputs = layers.conv2d(
          inputs,
          1,
          expanded_filters,
          1,
          is_training,
          data_format=data_format,
          use_relu=False,
          init_zero=False)

  if block['squeeze'] > 0:
    with tf.variable_scope('SqueezeExcite'):
      inputs = layers.squeeze_and_excite(inputs, expanded_filters,
                                         block['squeeze'], new_frames)

  if block['context_gate'] != 0:
    with tf.variable_scope('ContextGate'):
      inputs = layers.context_gate(inputs, expanded_filters, new_frames)

  if block['inv-bottle'] and block['expand'] > 0:
    inputs = layers.conv2d(
        inputs,
        1,
        block['filters'],
        1,
        is_training,
        data_format=data_format,
        use_relu=False,
        init_zero=False)

  if use_skip:
    with tf.variable_scope('Residual'):
      sp_strides = sp_strides if inputs.shape[2] != res.shape[2] else 1
      res = layers.conv2d(
          res,
          1,
          inputs.shape[3],
          sp_strides,
          is_training,
          data_format=data_format,
          use_relu=False)

    inputs = inputs + res

  if block['act'] == 'relu':
    act = tf.nn.relu
  elif block['act'] == 'hswish':
    act = layers.hard_swish
  else:
    raise ValueError('%s is not implemented' % block['act'])
  return act(inputs), new_frames


def tiny_video_net(model_string,
                   num_classes,
                   num_frames,
                   data_format='channels_last',
                   dropout_keep_prob=0.5,
                   get_representation=False,
                   max_pool_predictions=False):
  """Builds TinyVideoNet based on model string.

  Args:
    model_string: string defining the tiny video model (see top for example
      model string)
    num_classes: int number of classes to classify
    num_frames: int, number of frames in clip
    data_format: string, either channels_last or channels_first
    dropout_keep_prob: float, dropout keep probability
    get_representation: bool, True to return the representation.
    max_pool_predictions: bool, if True, will max pool the predictions over the
      temporal dimension. If False, will average pool. Max pooling is useful for
      long videos where the action only happens over a short sub-sequence of the
      whole video, such as in the Charades dataset.

  Returns:
    model function (inputs, is_training)
  """
  if dropout_keep_prob is None:
    dropout_keep_prob = 1.0

  model = json.loads(model_string)

  def model_fn(inputs, is_training):
    """Creation of the model graph."""

    input_shape = inputs.shape

    if 'input_streams' in model:
      batch_size = input_shape[0]
      inputs, blocks = multistream_tvn(model, inputs, is_training, num_frames)
    else:
      batch_size = input_shape[0] // num_frames
      inputs, blocks = tvn(model, inputs, is_training)

    feature_shape = inputs.shape
    current_frames = feature_shape[0] // batch_size

    if get_representation:
      representation = inputs

    if max_pool_predictions:
      batch_size = int(feature_shape[0].value // current_frames)
      inputs = tf.reshape(inputs,
                          [batch_size, current_frames, -1, inputs.shape[3]])
      # Spatial average pooling.
      inputs = tf.reduce_mean(inputs, axis=2)

      if is_training:
        inputs = tf.nn.dropout(inputs, dropout_keep_prob)

      # Per-frame predictions.
      inputs = tf.layers.dense(
          inputs=inputs,
          units=num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=.01))
      # Per-frame max-pooling. The max pooling is done in a softmax manner for
      # more stable training.
      pre_logits = inputs / tf.sqrt(tf.cast(current_frames, inputs.dtype))
      acts = tf.nn.softmax(pre_logits, axis=1)
      inputs = tf.math.multiply(inputs, acts)
      inputs = tf.reduce_sum(inputs, axis=1)
    else:
      # Global-average-pool.
      inputs = tf.reshape(inputs, [
          int(feature_shape[0].value // current_frames),
          current_frames * feature_shape[1], feature_shape[2], -1
      ])
      inputs = tf.reduce_mean(inputs, axis=[1, 2])
      if is_training:
        inputs = tf.nn.dropout(inputs, dropout_keep_prob)
      inputs = tf.layers.dense(
          inputs=inputs,
          units=num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=.01))

    if get_representation:
      return {
          'feats-emb': representation,
          'blocks': blocks,
          'frames': current_frames,
          'predictions': inputs,
          'logits': inputs
      }
    return {'logits': inputs}

  def tvn(model, inputs, is_training):
    """Standard single-stream TVN."""
    current_frames = num_frames
    b = 0
    blocks = {}

    for block in model['blocks']:
      with tf.variable_scope('Block-%d-0' % b):
        inputs, new_frames = tiny_block(block, inputs, current_frames,
                                        is_training, data_format)
      current_frames = new_frames
      # Repeat block with no stride and alternating use of temporal conv1.
      use_temporal = False
      for i in range(block['repeats'] - 1):
        with tf.variable_scope('Block-%d-%d' % (b, i + 1)):
          inputs, _ = tiny_block(
              block,
              inputs,
              current_frames,
              is_training,
              data_format,
              strides=1,
              use_temporal=use_temporal)
          use_temporal = not use_temporal
      blocks['block-' + str(b)] = inputs
      b += 1
    return inputs, blocks

  def multistream_tvn(model, inputs, is_training, input_num_frames):
    """Multi-stream (assemblenet-like) TVN."""
    input_shape = inputs.shape
    is_4d = False
    if len(input_shape) == 4:
      # Handle 4D input tensor [batch*time, height, width, channels].
      batch_size = input_shape[0] // input_num_frames
      is_4d = True
    else:
      # Handle 5D input tensor.
      batch_size = input_shape[0]
    b = 0
    blocks = {}

    # Get input streams.
    input_streams = []
    dtype = inputs.dtype
    for stream in model['input_streams']:
      img_size = stream['image_size']
      num_frames = stream['num_frames']
      height = inputs.shape[2]
      if is_4d:
        # Maintain 4D tensor always
        strm = tf.reshape(inputs, [batch_size, input_num_frames,
                                   height*height, 3])
      else:
        strm = inputs
      strm = strm[:, :num_frames]
      strm = tf.reshape(strm, [batch_size * strm.shape[1], height, height, 3])
      if height != img_size:
        strm = tf.image.resize(strm, (img_size, img_size))
      if strm.dtype != dtype:
        strm = tf.cast(strm, dtype)
      input_streams.append(tf.stop_gradient(strm))

    for block in model['blocks']:
      with tf.variable_scope('Block-%d-0' % b):
        # Get block input.
        inputs = input_streams[block['inputs'][0]]
        inputs, current_frames = tiny_block(block, inputs, current_frames,
                                            is_training, data_format)
      # Repeat block with no stride and alternating use of temporal conv1.
      use_temporal = False
      for i in range(block['repeats'] - 1):
        with tf.variable_scope('Block-%d-%d' % (b, i + 1)):
          inputs, _ = tiny_block(
              block,
              inputs,
              current_frames,
              is_training,
              data_format,
              strides=1,
              use_temporal=use_temporal)
          use_temporal = not use_temporal
      blocks['block-' + str(b)] = inputs
      b += 1
      input_streams.append(inputs)
    return input_streams[-1], blocks

  return model_fn
