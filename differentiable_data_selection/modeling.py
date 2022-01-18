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

# pylint: disable=logging-format-interpolation
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=protected-access

r"""Models."""


from absl import logging

import tensorflow.compat.v1 as tf
from differentiable_data_selection import modeling_utils as ops


class Wrn28k(object):
  """WideResNet."""

  def __init__(self, params, k=2):
    self.params = params
    self.name = f'wrn-28-{k}'
    self.k = k
    logging.info(f'Build `wrn-28-{k}` under scope `{self.name}`')

  def __call__(self, x, training, return_scores=False):
    if training:
      logging.info(f'Call {self.name} for `training`')
    else:
      logging.info(f'Call {self.name} for `eval`')

    params = self.params
    k = self.k
    if params.use_bfloat16:
      ops.use_bfloat16()

    s = [16, 135, 135*2, 135*4] if k == 135 else [16*k, 16*k, 32*k, 64*k]

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      with tf.variable_scope('stem'):
        x = ops.conv2d(x, 3, s[0], 1)
        ops.log_tensor(x, True)

      x = ops.wrn_block(x, params, s[1], 1, training, 'block_1')
      x = ops.wrn_block(x, params, s[1], 1, training, 'block_2')
      x = ops.wrn_block(x, params, s[1], 1, training, 'block_3')
      x = ops.wrn_block(x, params, s[1], 1, training, 'block_4')

      x = ops.wrn_block(x, params, s[2], 2, training, 'block_5')
      x = ops.wrn_block(x, params, s[2], 1, training, 'block_6')
      x = ops.wrn_block(x, params, s[2], 1, training, 'block_7')
      x = ops.wrn_block(x, params, s[2], 1, training, 'block_8')

      x = ops.wrn_block(x, params, s[3], 2, training, 'block_9')
      x = ops.wrn_block(x, params, s[3], 1, training, 'block_10')
      x = ops.wrn_block(x, params, s[3], 1, training, 'block_11')
      x = ops.wrn_block(x, params, s[3], 1, training, 'block_12')

      with tf.variable_scope('head'):
        x = ops.batch_norm(x, params, training)
        x = ops.relu(x)
        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
        ops.log_tensor(x, True)

        x = ops.dropout(x, params.dense_dropout_rate, training)
        if return_scores:
          x = ops.dense(x, 1, use_bias=False)
          x = params.scorer_clip * tf.tanh(x, name='scores')
        else:
          x = ops.dense(x, params.num_classes)
        x = tf.cast(x, dtype=tf.float32, name='logits')
        ops.log_tensor(x, True)

    return x


class ResNet50(object):
  """Bottleneck ResNet."""

  def __init__(self, params):
    self.params = params
    self.name = 'resnet-50'
    logging.info(f'Build `resnet-50` under scope `{self.name}`')

  def __call__(self, x, training):
    if training:
      logging.info(f'Call {self.name} for `training`')
    else:
      logging.info(f'Call {self.name} for `eval`')

    params = self.params
    if params.use_bfloat16:
      ops.use_bfloat16()

    def _block_fn(inputs, num_out_filters, stride, name):
      return ops.resnet_block(inputs,
                              params=params,
                              num_out_filters=num_out_filters,
                              stride=stride,
                              training=training,
                              name=name)

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      with tf.variable_scope('stem'):
        x = ops.conv2d(x, 7, 64, 2)
        x = ops.batch_norm(x, params, training)
        x = ops.relu(x, leaky=0.)
        ops.log_tensor(x, True)

        x = ops.max_pool(x, 3, 2)
        ops.log_tensor(x, True)

      x = _block_fn(x, 256, 1, name='block_1')
      x = _block_fn(x, 256, 1, name='block_2')
      x = _block_fn(x, 256, 1, name='block_3')

      x = _block_fn(x, 512, 2, name='block_4')
      x = _block_fn(x, 512, 1, name='block_5')
      x = _block_fn(x, 512, 1, name='block_6')
      x = _block_fn(x, 512, 1, name='block_7')

      x = _block_fn(x, 1024, 2, name='block_8')
      x = _block_fn(x, 1024, 1, name='block_9')
      x = _block_fn(x, 1024, 1, name='block_10')
      x = _block_fn(x, 1024, 1, name='block_11')
      x = _block_fn(x, 1024, 1, name='block_12')
      x = _block_fn(x, 1024, 1, name='block_13')

      x = _block_fn(x, 2048, 2, name='block_14')
      x = _block_fn(x, 2048, 1, name='block_15')
      x = _block_fn(x, 2048, 1, name='block_16')

      with tf.variable_scope('head'):
        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
        ops.log_tensor(x, True)

        x = ops.dropout(x, params.dense_dropout_rate, training)
        x = ops.dense(x, params.num_classes)
        x = tf.cast(x, dtype=tf.float32, name='logits')
        ops.log_tensor(x, True)

    return x
