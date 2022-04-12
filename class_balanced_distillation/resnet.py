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

"""ResNet model for TF2."""

import functools
from typing import Optional, Sequence, Text, Union

from class_balanced_distillation import resnet_helpers
import tensorflow.compat.v2 as tf


class CosineLayer(tf.keras.Model):
  """Cosine Layer."""

  def __init__(self, in_features, num_classes, name=None):
    super(CosineLayer, self).__init__(name=name)

    self.num_classes = num_classes
    self.sigma = tf.Variable(20.0, dtype=tf.float32,
                             trainable=False, name='sigma')

    w_init = tf.keras.initializers.GlorotUniform()
    self.w = tf.Variable(initial_value=w_init(shape=(in_features, num_classes),
                                              dtype='float32'),
                         trainable=True, name='weights')

  def __call__(self, x, labels=None, training=False):
    x = tf.math.l2_normalize(x, 1)
    weights = tf.math.l2_normalize(self.w, 0)
    x = tf.matmul(x, weights)
    x = self.sigma * x

    return x


class BlockGroup(tf.keras.Model):
  """Higher level block for ResNet implementation."""

  def __init__(self,
               channels,
               num_blocks,
               stride,
               norm,
               block,
               name = None):
    super(BlockGroup, self).__init__(name=name)
    self._channels = channels * block.EXPANSION
    self._num_blocks = num_blocks
    self._stride = stride

    self._blocks = []
    for id_block in range(num_blocks):
      self._blocks.append(
          block(
              channels=self._channels,
              stride=stride if id_block == 0 else 1,
              use_projection=(id_block == 0),
              norm=norm,
              name='block_{}'.format(id_block)))

  def __call__(self, inputs, **norm_kwargs):
    net = inputs
    for block in self._blocks:
      net = block(net, **norm_kwargs)
    return net


class ResNet(tf.keras.Model):
  """ResNet model."""

  def __init__(self,
               blocks_per_group_list,
               num_classes,
               norm,
               block,
               channels_per_group_list = (64, 128, 256, 512),
               proj_dim=-1,
               name = None):
    """Constructs a ResNet model.

    Args:
      blocks_per_group_list: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      norm: The normalization object.
      block: Pointer to class of ResNet block (eg. BasicBlock).
      channels_per_group_list: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      proj_dim: Output dimensionality of the projection layer before the
        classifier. Set -1 to skip.
      name: Name of the module.
    """
    super(ResNet, self).__init__(name=name)

    # Number of blocks in each group for ResNet.
    if len(blocks_per_group_list) != 4:
      raise ValueError(
          '`blocks_per_group_list` must be of length 4 not {}'.format(
              len(blocks_per_group_list)))
    self._blocks_per_group_list = blocks_per_group_list

    # Number of channels in each group for ResNet.
    if len(channels_per_group_list) != 4:
      raise ValueError(
          '`channels_per_group_list` must be of length 4 not {}'.format(
              len(channels_per_group_list)))
    self._channels_per_group_list = channels_per_group_list

    self._initial_conv = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        use_bias=False,
        name='initial_conv')

    self._initial_norm = norm(name='initial_' + 'bn')

    self._block_groups = []
    strides = [1, 2, 2, 2]
    for i in range(4):
      self._block_groups.append(
          BlockGroup(
              channels=self._channels_per_group_list[i],
              num_blocks=self._blocks_per_group_list[i],
              stride=strides[i],
              norm=norm,
              block=block,
              name='block_group_%d' % i))

    self.proj_dim = proj_dim
    if self.proj_dim != -1:
      self.proj_layer = tf.keras.layers.Dense(
          units=self.proj_dim, name='proj_layer')
      self.embedding_dim = self.proj_dim
    else:
      self.embedding_dim = 512 * block.EXPANSION
    self.linear = CosineLayer(in_features=self.embedding_dim,
                              num_classes=num_classes)

  def __call__(self,
               inputs,
               labels=None,
               training=False,
               return_features=False):
    net = inputs
    net = self._initial_conv(net)
    net = self._initial_norm(net, training=training)
    net = tf.nn.relu(net)

    net = tf.nn.max_pool2d(
        net, ksize=3, strides=2, padding='SAME', name='initial_max_pool')

    for block_group in self._block_groups:
      net = block_group(net, training=training)

    net = tf.reduce_mean(net, axis=[1, 2], name='final_avg_pool')

    if self.proj_dim != -1:
      net = self.proj_layer(net)
      net = tf.nn.relu(net)

    net = tf.nn.l2_normalize(net, axis=1)

    if return_features:
      features = net

    net = self.linear(net)

    if return_features:
      return features, net
    else:
      return net


def resnet(num_layers,
           num_classes,
           proj_dim,
           name = None):
  """Constructs a ResNet model.

  Args:
    num_layers: The number of layers of ResNet.
    num_classes: The number of classes to classify the inputs into.
    proj_dim: Dimensions of the dense projection layer before the classifier
    name: Name of the module.

  Returns:
    model: the resnet model.
  """

  block_group_dict = {
      18: ([2, 2, 2, 2], 'basic'),
      26: ([2, 2, 2, 2], 'bottleneck'),
      34: ([3, 4, 6, 3], 'basic'),
      50: ([3, 4, 6, 3], 'bottleneck'),
      101: ([3, 4, 23, 3], 'bottleneck'),
      152: ([3, 8, 36, 3], 'bottleneck'),
  }

  resnet_type_dict = {
      'resnet': {
          'basic': resnet_helpers.BasicBlock,
          'bottleneck': resnet_helpers.BottleNeckBlockV1,
      },
  }

  # Check number of layers
  if num_layers in block_group_dict:
    block_groups, block_type = block_group_dict[num_layers]
  else:
    raise NotImplementedError(
        'Please choose among the '
        '18-, 26-, 34-, 50-, 101-, or 152-layer variant of ResNet.')

  norm = functools.partial(
      tf.keras.layers.BatchNormalization,
      momentum=0.9,
      epsilon=1e-5,
      )
  block = resnet_type_dict['resnet'][block_type]

  print('Initializing resnet-{}'.format(num_layers))

  model = ResNet(
      blocks_per_group_list=block_groups,
      num_classes=num_classes,
      norm=norm,
      block=block,
      proj_dim=proj_dim,
      name=name)

  return model
