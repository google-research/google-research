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

# Lint as: python3
"""Implementation for Contrastive classification head."""

import tensorflow.compat.v1 as tf


class ClassificationHead(tf.layers.Layer):
  """A classification head.

  Attributes:
    num_classes: The number of classes to classify into.
    kernel_initializer: An initializer to use for the weights.
    name: Name for this object.
  """

  def __init__(self,
               num_classes,
               kernel_initializer=tf.initializers.glorot_uniform(),
               name='ClassificationHead',
               **kwargs):
    super(ClassificationHead, self).__init__(name=name, **kwargs)

    self.dense_layer = tf.layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None)

  def call(self, inputs, training=None):
    del training  # unused.

    if inputs.shape.rank != 2:
      raise ValueError(
          f'Input shape {inputs.shape} is expected to have rank 2, but does '
          'not.')

    return self.dense_layer(inputs)
