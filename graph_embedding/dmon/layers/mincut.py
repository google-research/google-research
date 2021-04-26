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


class MincutPooling(Layer):  # pylint: disable=missing-class-docstring

  def __init__(self,
               k,
               orthogonality_regularization = 1.0,
               cluster_size_regularization = 0.0,
               dropout_rate = 0,
               mlp_sizes = None,
               do_unpool = True):
    super(MincutPooling, self).__init__()
    self.k = k
    self.orthogonality_regularization = orthogonality_regularization
    self.cluster_size_regularization = cluster_size_regularization
    self.dropout_rate = dropout_rate
    self.mlp_sizes = [] if mlp_sizes is None else mlp_sizes
    self.do_unpool = do_unpool

  def build(self, input_shape):
    mlp = []
    for size in self.mlp_sizes:
      mlp.append(tf.keras.layers.Dense(size, activation='selu'))
    mlp.append(
        tf.keras.layers.Dense(
            self.k, kernel_initializer='orthogonal', bias_initializer='zeros'))
    mlp.append(tf.keras.layers.Dropout(self.dropout_rate))
    self.mlp = tf.keras.models.Sequential(mlp)
    super().build(input_shape)

  def call(self, inputs):
    features, graph = inputs

    assignments = tf.nn.softmax(self.mlp(features), axis=1)
    assignments_pool = assignments / tf.math.reduce_sum(assignments, axis=0)

    graph_pooled = tf.transpose(
        tf.sparse.sparse_dense_matmul(graph, assignments))
    graph_pooled = tf.matmul(graph_pooled, assignments)
    numerator = tf.linalg.trace(graph_pooled)
    denominator = tf.transpose(assignments) * tf.sparse.reduce_sum(
        graph, axis=-1)
    denominator = tf.matmul(denominator, assignments)
    denominator = tf.linalg.trace(denominator)
    spectral_loss = -(numerator / denominator)

    self.add_loss(spectral_loss)

    pairwise = tf.matmul(assignments, assignments, transpose_a=True)
    identity = tf.eye(self.k)

    orthogonality_loss = tf.norm(pairwise / tf.norm(pairwise) -
                                 identity / tf.sqrt(float(self.k)))
    self.add_loss(self.orthogonality_regularization * orthogonality_loss)

    cluster_loss = tf.norm(tf.reduce_sum(
        pairwise, axis=1)) / graph.shape[1] * tf.sqrt(float(self.k)) - 1
    self.add_loss(self.cluster_size_regularization * cluster_loss)

    features_pooled = tf.matmul(assignments_pool, features, transpose_a=True)
    features_pooled = tf.nn.selu(features_pooled)
    if self.do_unpool:
      features_pooled = tf.matmul(assignments_pool, features_pooled)
    return features_pooled, assignments
