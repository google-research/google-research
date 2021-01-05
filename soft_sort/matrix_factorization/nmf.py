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
"""Functions use to solve Non-Negative Matrix Factorization."""

from typing import Optional, Tuple
import gin
import tensorflow.compat.v2 as tf

from soft_sort.matrix_factorization import measures


@gin.configurable
def update(u, v, x, min_entry = 1e-20):
  """Runs a NMF update."""
  ut = tf.transpose(u)
  v = v * tf.matmul(ut, x) / (min_entry + tf.matmul(tf.matmul(ut, u), v))
  vt = tf.transpose(v)
  u = u * tf.matmul(x, vt) / (min_entry + tf.matmul(u, tf.matmul(v, vt)))
  return u, v


@gin.configurable
class NMF:
  """Non-Negative Matrix Factorization: X = UV."""

  def __init__(self,
               low_rank = 10,
               num_iterations = 100,
               min_init = 1e-2):
    self._rank = low_rank
    self._num_iterations = num_iterations
    self._min_init = min_init
    self.losses = []
    self.ckpt_variables = {}
    self.x = None
    self.u = None
    self.v = None

  def reset(self, input_matrix, random_seed = None):
    """Initializes the variables based on the matrix to be factorized."""
    if random_seed is not None:
      tf.random.set_seed(random_seed)
    self.x = input_matrix

    shape = tf.shape(input_matrix)
    self.u = tf.random.uniform((shape[0], self._rank)) + self._min_init
    self.v = tf.random.uniform((self._rank, shape[1])) + self._min_init
    self.losses = []
    self.ckpt_variables = {}

  @property
  def reconstruction(self):
    return tf.matmul(self.u, self.v)

  def loss(self):
    return measures.kl_divergence(self.x, self.reconstruction)

  def update(self):
    self.u, self.v = update(self.u, self.v, self.x)
    loss = self.loss()
    self.losses.append(loss)
    return loss

  def __call__(self,
               input_matrix,
               epochs = None,
               random_seed = None):
    self.reset(input_matrix, random_seed=random_seed)
    epochs = self._num_iterations if epochs is None else epochs
    for _ in range(epochs):
      self.update()
    return self.u, self.v
