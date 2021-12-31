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

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=protected-access

r"""Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
from meta_pseudo_labels import modeling_utils as ops


class Wrn28k(object):
  """WideResNet."""

  def __init__(self, params, k=2):
    self.params = params
    self.name = f'wrn-28-{k}'
    self.k = k
    logging.info(f'Build `wrn-28-{k}` under scope `{self.name}`')

  def __call__(self, x, training, start_core_index=0, final_core_index=1):
    if training:
      logging.info(f'Call {self.name} for `training`')
    else:
      logging.info(f'Call {self.name} for `eval`')

    params = self.params
    k = self.k
    if params.use_bfloat16:
      ops.use_bfloat16()
    if params.use_xla_sharding:
      ops.set_xla_sharding(params.num_cores_per_replica)

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

  def __call__(self, x, training, start_core_index=0, final_core_index=1):
    if training:
      logging.info(f'Call {self.name} for `training`')
    else:
      logging.info(f'Call {self.name} for `eval`')

    params = self.params
    if params.use_bfloat16:
      ops.use_bfloat16()
    if params.use_xla_sharding:
      ops.set_xla_sharding(params.num_cores_per_replica)

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


class EfficientNet(object):
  """EfficientNet."""

  def __init__(self, params):
    self.params = params
    self.name = params.model_type
    logging.info(f'Build `{self.name}` under scope `{self.name}`')

    self.width_mul, self.depth_mul, self.eval_image_size, _ = {
        # (width_mul, depth_mul, resolution, dropout)
        'efficientnet-bs': (0.5, 1.0, 224, 0.2),
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 600, 0.5),
        'efficientnet-b5-mpl': (5.0, 2.2, 800, 0.4),
        'efficientnet-b6-mpl': (5.0, 2.6, 800, 0.5),
    }[params.model_type]

  def _is_small_net(self):
    """Returns whether we are dealing with CIFAR / TinyImages images."""
    dataset_name = self.params.dataset_name.lower()
    key_words = ['cifar', 'tinyimages', 'svhn']
    return any([k in dataset_name for k in key_words])

  def _fil(self, inp_filters, divisor=8):
    """Round number of filters based on width coefficient."""
    inp_filters *= self.width_mul
    out_filters = int(inp_filters + divisor//2) // divisor * divisor
    if out_filters < divisor:
      out_filters = divisor
    if out_filters < 0.9 * inp_filters:
      out_filters += divisor
    return int(out_filters)

  def _rep(self, num_repeats):
    """Round number of repeats based on depth multiplier."""
    return int(np.ceil(self.depth_mul * num_repeats))

  def __call__(self, x, training, start_core_index=0, final_core_index=1):
    if training:
      logging.info(f'Call {self.name} for `training`')
    else:
      logging.info(f'Call {self.name} for `eval`')

    params = self.params
    if params.use_bfloat16:
      ops.use_bfloat16()
    if params.use_xla_sharding:
      ops.set_xla_sharding(params.num_cores_per_replica)

    num_repeats = [self._rep(rp) for rp in [1, 2, 2, 3, 3, 4, 1]]
    num_blocks = sum(num_repeats)

    start = 0
    block_start = []
    for r in num_repeats:
      block_start.append(start)
      start += r

    s = 1 if self._is_small_net() else 2  # smaller strides for CIFAR

    def stack_fn(inputs, repeats, filter_size, num_out_filters, stride,
                 expand_ratio, block_start):
      """Build a stack of multiple `mb_conv_block`."""
      for i in range(self._rep(repeats)):
        inputs = ops.mb_conv_block(
            x=inputs,
            params=params,
            filter_size=filter_size,
            num_out_filters=self._fil(num_out_filters),
            stride=stride if i == 0 else 1,  # only first block uses `stride`
            training=training,
            stochastic_depth_drop_rate=(
                params.stochastic_depth_drop_rate *
                float(block_start+i) / num_blocks),
            expand_ratio=expand_ratio,
            use_se=True,
            se_ratio=0.25,
            name=f'block_{block_start+i}')
      return inputs

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      with tf.variable_scope('stem'):
        x = ops.conv2d(x, 3, self._fil(32), s)
        x = ops.batch_norm(x, params, training=training)
        x = ops.swish(x)

      x = stack_fn(x, 1, 3, 16, 1, 1, block_start[0])
      x = stack_fn(x, 2, 3, 24, s, 6, block_start[1])
      x = stack_fn(x, 2, 5, 40, s, 6, block_start[2])
      x = stack_fn(x, 3, 3, 80, 2, 6, block_start[3])
      x = stack_fn(x, 3, 5, 112, 1, 6, block_start[4])
      x = stack_fn(x, 4, 5, 192, 2, 6, block_start[5])
      x = stack_fn(x, 1, 3, 320, 1, 6, block_start[6])

      with tf.variable_scope('head'):
        x = ops.conv2d(x, 1, self._fil(1280), 1)
        x = ops.batch_norm(x, params, training)
        x = ops.swish(x)
        ops.log_tensor(x, True)

        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
        ops.log_tensor(x, True)

        x = ops.dropout(x, params.dense_dropout_rate, training)
        x = ops.dense(x, params.num_classes)
        x = tf.cast(x, dtype=tf.float32, name='logits')
        ops.log_tensor(x, True)

    return x
