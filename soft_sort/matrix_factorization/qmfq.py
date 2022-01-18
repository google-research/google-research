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
"""Learn jointly the quantile normalization and the matrix factorization."""

import gin
import tensorflow.compat.v2 as tf

from soft_sort import ops
from soft_sort.matrix_factorization import measures
from soft_sort.matrix_factorization import nmf


@gin.configurable
class QMFQ:
  """Non negative matrix factorization with learned quantile normalization."""

  def __init__(self,
               low_rank: int,
               num_quantiles: int,
               num_nmf_updates: int = 100,
               learning_rate: float = 1e-3,
               optimizer: tf.optimizers.Optimizer = 'adam',
               **kwargs):
    self._low_rank = low_rank
    self._num_quantiles = num_quantiles
    self._num_nmf_updates = num_nmf_updates
    self._kwargs = kwargs
    self._learning_rate = learning_rate
    self._optimizer = tf.keras.optimizers.get(optimizer)
    self._nmf_factorizer = nmf.NMF(
        low_rank=self._low_rank, num_iterations=self._num_nmf_updates)

  def reset(self, x: tf.Tensor):
    """Initializes internal variable based on the inputs tensor x."""
    self.x = x
    self._maxima = tf.math.reduce_max(self.x, axis=-1)
    self._minima = tf.math.reduce_min(self.x, axis=-1)
    self._num_features = tf.shape(self.x)[0]
    self._num_individuals = tf.shape(self.x)[1]

    self.initialize_trainable_variables()
    self._optimizer.learning_rate = self._learning_rate
    self.losses = []
    self.inner_kl = []

  def initialize_trainable_variables(self):
    q_shape = (self._num_features, self._num_quantiles)
    n = tf.cast(self._num_individuals, dtype=self.x.dtype)
    self._q_prime = tf.Variable(-tf.math.log(n) * tf.ones(q_shape))
    self._b_prime = tf.Variable(tf.zeros(q_shape))
    self._q = tf.Variable(
        tf.zeros((self._num_features, self._num_quantiles - 1)))
    self._b = tf.Variable(tf.zeros(q_shape))
    self.trainable_variables = [self._q, self._b, self._q_prime, self._b_prime]
    self.ckpt_variables = {
        'q': self._q, 'b': self._b,
        'q_prime': self._q_prime, 'b_prime': self._b_prime,
        'optimizer': self._optimizer,
    }

  @property
  def quantiles(self):
    maxs = self._maxima[:, tf.newaxis]
    mins = self._minima[:, tf.newaxis]
    weights = tf.cumsum(tf.nn.softmax(self._q, axis=-1), axis=-1)
    weights = tf.concat([tf.zeros((self._num_features, 1)), weights], axis=1)
    return weights * (maxs - mins) + mins

  @property
  def quantiles_prime(self):
    return tf.math.cumsum(tf.math.exp(self._q_prime), axis=1)

  @property
  def target_weights(self):
    return tf.nn.softmax(self._b, axis=-1)

  @property
  def u(self):
    return self._nmf_factorizer.u

  @property
  def v(self):
    return self._nmf_factorizer.v

  @property
  def reconstruction(self):
    """Reconstruct the scale input."""
    quan_x = ops.soft_quantile_normalization(
        self.x,
        self.quantiles_prime,
        target_weights=tf.nn.softmax(self._b_prime, axis=-1),
        **self._kwargs)
    self._nmf_factorizer(quan_x, random_seed=0)
    self.inner_kl.append(self._nmf_factorizer.losses)
    return ops.soft_quantile_normalization(self._nmf_factorizer.reconstruction,
                                           self.quantiles,
                                           target_weights=self.target_weights,
                                           **self._kwargs)

  def loss(self):
    return measures.kl_divergence(self.x, self.reconstruction)

  def debug_str(self, loss, step):
    return '{} -->  step {}: loss={:.2f}'.format(
        self.__class__.__name__, step, loss)

  def update(self) -> float:
    with tf.GradientTape() as tape:
      loss = self.loss()
      grad = tape.gradient(loss, self.trainable_variables)
    self.losses.append(loss)
    self._optimizer.apply_gradients(zip(grad, self.trainable_variables))
    return loss

  def __call__(self, x: tf.Tensor, epochs: int = 5, reset=True):
    if reset:
      self.reset(x)

    for step in range(epochs):
      loss = self.update()
      print(self.debug_str(loss, step), end='\r')
    print(self.debug_str(loss, epochs), flush=True, end='\n')


