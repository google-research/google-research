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
"""TODO(tsitsulin): add headers, tests, and improve style."""
from typing import List
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Layer


class DiffPooling(Layer):  # TODO(tsitsulin): add docstring pylint: disable=missing-class-docstring

  def __init__(self, k, do_unpool = True):
    super(DiffPooling, self).__init__()
    self.k = k
    self.do_unpool = do_unpool

  def build(self, input_shape):
    self.mlp = tf.keras.layers.Dense(
        self.k,
        activation='softmax',
        kernel_initializer='orthogonal',
        bias_initializer='zeros')
    super().build(input_shape)

  def call(self, inputs):
    features, graph = inputs

    assignments = self.mlp(features)
    assignments_pool = assignments / tf.math.reduce_sum(assignments, axis=0)

    graph_reconstruction = tf.matmul(assignments, assignments, transpose_b=True)
    linkprediction_loss = tf.norm(graph - graph_reconstruction)
    self.add_loss(linkprediction_loss)

    entropy_loss = -tf.reduce_sum(
        tf.multiply(assignments, tf.math.log(assignments + 1e-8)), axis=-1)
    entropy_loss = tf.reduce_mean(entropy_loss)
    self.add_loss(entropy_loss)

    features_pooled = tf.matmul(assignments_pool, features, transpose_a=True)
    features_pooled = tf.nn.selu(features_pooled)
    if self.do_unpool:
      features_pooled = tf.matmul(assignments_pool, features_pooled)
    return features_pooled, assignments
