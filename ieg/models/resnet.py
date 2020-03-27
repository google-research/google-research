# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small ResNet for CIFAR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ieg.models.networks import StrategyNetBase
from ieg.third_party.cifar10_resnet import resnet_model_fn

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class ResNet(StrategyNetBase):
  """ResNet meta class for CIFAR."""

  def __init__(self,
               num_classes,
               depth):
    super(ResNet, self).__init__()

    if (depth - 2) % 9 != 0:
      raise ValueError('depth == 9n+2')

    self.depth = depth
    self.num_classes = num_classes
    self.regularization_loss = 0
    # Used by MetaImage to compute l2 regularizer
    self.wd = 1e-4

  def save(self, path):
    self.model.save(path)

  def load_weights(self, path):
    self.model.load_weights(path)

  def __call__(self, inputs, name, training, reuse=True, custom_getter=None):

    num_res_blocks = int((self.depth - 2) / 9)
    if self.created and not reuse:
      tf.logging.error('Warning: are you parallel?')

    with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
      outputs = resnet_model_fn(
          inputs,
          training,
          self.num_classes,
          num_res_blocks,
          weight_decay=self.wd)

      if not isinstance(reuse, bool) or not reuse:
        # If it is tf.AUTO_REUSE or True to make sure regularization_loss is
        # added once.
        self.regularization_loss = self.get_regularization_loss(
            scope_name=name + '/')
        self.init(name, with_name='batch_normalization', outputs=outputs)
        self.count_parameters(name)

    return outputs
