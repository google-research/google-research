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

# Lint as: python3
"""TODO(tsitsulin): add headers, tests, and improve style."""
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Layer  # pylint: disable=unused-import


class ModularityPooling(tf.keras.layers.Layer):  # pylint: disable=missing-class-docstring

  def __init__(self,  # pylint: disable=dangerous-default-value
               k,
               orthogonality_regularization=0.3,
               cluster_size_regularization=1.0,
               dropout_rate=0.75,
               mlp_sizes=[],
               do_unpool=True):
    super(ModularityPooling, self).__init__()
    self.k = k
    self.orthogonality_regularization = orthogonality_regularization
    self.cluster_size_regularization = cluster_size_regularization
    self.dropout_rate = dropout_rate
    self.mlp_sizes = mlp_sizes
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
    features, adjacency = inputs

    assignments = tf.nn.softmax(self.mlp(features), axis=1)
    assignments_pool = assignments / tf.math.reduce_sum(assignments, axis=0)

    degrees = tf.sparse.reduce_sum(adjacency, axis=0)
    degrees = tf.reshape(degrees, (-1, 1))
    m = tf.math.reduce_sum(degrees)

    graph_pooled = tf.transpose(
        tf.sparse.sparse_dense_matmul(adjacency, assignments))
    graph_pooled = tf.matmul(graph_pooled, assignments)

    ca = tf.matmul(assignments, degrees, transpose_a=True)  # S^T d
    cb = tf.matmul(degrees, assignments, transpose_a=True)  # d^T S

    normalizer = tf.matmul(ca, cb) / 2 / m
    spectral_loss = -tf.linalg.trace(graph_pooled - normalizer) / 2 / m

    self.add_loss(spectral_loss)

    pairwise = tf.matmul(assignments, assignments, transpose_a=True)
    identity = tf.eye(self.k)

    orthogonality_loss = tf.norm(pairwise / tf.norm(pairwise) -
                                 identity / tf.sqrt(float(self.k)))
    self.add_loss(self.orthogonality_regularization * orthogonality_loss)

    cluster_loss = tf.norm(tf.reduce_sum(
        pairwise, axis=1)) / adjacency.shape[1] * tf.sqrt(float(self.k)) - 1
    self.add_loss(self.cluster_size_regularization * cluster_loss)

    features_pooled = tf.matmul(assignments_pool, features, transpose_a=True)
    features_pooled = tf.nn.selu(features_pooled)
    if self.do_unpool:
      features_pooled = tf.matmul(assignments_pool, features_pooled)
    return features_pooled, assignments
