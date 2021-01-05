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
"""Learns jointly the quantile normalization and the matrix factorization."""

from typing import Optional

import gin
import tensorflow.compat.v2 as tf

from soft_sort import ops
from soft_sort.matrix_factorization import measures


@gin.configurable
class QMF:
  """Non negative matrix factorization with learned quantile normalization."""

  def __init__(self,
               low_rank: int,
               num_quantiles: int,
               batch_size: Optional[int] = None,
               learning_rate: float = 1e-2,
               regularization_factor: float = 1e-2,
               optimizer: tf.optimizers.Optimizer = 'adam',
               **kwargs):
    self._low_rank = low_rank
    self._num_quantiles = num_quantiles
    self._kwargs = kwargs
    self._learning_rate = learning_rate
    self._regularization_factor = regularization_factor
    self._optimizer = tf.keras.optimizers.get(optimizer)
    self._batch_size = batch_size

    self.x = None
    self._q = None
    self._b = None
    self._u = None
    self._v = None
    self._maxima = None
    self._minima = None
    self._num_features = None
    self._num_individuals = None
    self.losses = []
    self.dataset = None

  def initialize_trainable_variables(self):
    """Initializes the trainable variables."""
    q_shape = (self._num_features, self._num_quantiles)
    self._q = tf.Variable(
        tf.zeros((self._num_features, self._num_quantiles - 1)))
    self._b = tf.Variable(tf.zeros(q_shape))
    self._u = tf.Variable(
        tf.random.uniform((self._num_features, self._low_rank)) - 0.5)
    self._v = tf.Variable(
        tf.random.uniform((self._low_rank, self._num_individuals)) - 0.5)
    self.trainable_variables = [self._u, self._q, self._b, self._v]
    self.ckpt_variables = {
        'u': self._u, 'v': self._v, 'q': self._q, 'b': self._b,
        'optimizer': self._optimizer
    }

  def reset(self, inputs: tf.Tensor, batch_size: Optional[int] = None):
    """Initializes internal variable based on the inputs tensor x."""
    self.x = inputs
    self._maxima = tf.math.reduce_max(self.x, axis=-1)
    self._minima = tf.math.reduce_min(self.x, axis=-1)
    self._num_features = tf.shape(self.x)[0]
    self._num_individuals = tf.shape(self.x)[1]
    self.initialize_trainable_variables()
    self._optimizer.learning_rate = self._learning_rate
    self.losses = []

    if batch_size is None or batch_size <= 0:
      batch_size = self._batch_size
    self.dataset = self.to_dataset(self.x, batch_size=batch_size)

  @property
  def quantiles(self):
    return self.sliced_quantiles(rows=None)

  @property
  def uv(self):
    return self.sliced_uv(rows=None, cols=None)

  @property
  def u(self):
    return tf.math.exp(self._u)

  @property
  def v(self):
    return tf.math.exp(self._v)

  @property
  def target_weights(self):
    return self.sliced_target_weights(rows=None)

  @property
  def reconstruction(self):
    """Reconstruct the scale input."""
    return self.sliced_reconstruction(rows=None, cols=None)

  def regularization(self, rows=None, cols=None):
    u_pre = tf.gather(self._u, rows, axis=0) if rows is not None else self._u
    v_pre = tf.gather(self._v, cols, axis=1) if cols is not None else self._v
    avg = 0.5 * (
        tf.math.reduce_mean(u_pre ** 2) + tf.math.reduce_mean(v_pre ** 2))
    return self._regularization_factor * avg

  def sliced_uv(self, rows=None, cols=None):
    u_pre = tf.gather(self._u, rows, axis=0) if rows is not None else self._u
    v_pre = tf.gather(self._v, cols, axis=1) if cols is not None else self._v
    return tf.matmul(tf.math.exp(u_pre), tf.math.exp(v_pre))

  def sliced_target_weights(self, rows=None):
    b_pre = tf.gather(self._b, rows, axis=0) if rows is not None else self._b
    return tf.nn.softmax(b_pre, axis=-1)

  def sliced_quantiles(self, rows=None):
    """Gets the quantiles for given rows."""
    maxs = (self._maxima if rows is None
            else tf.gather(self._maxima, rows, axis=0))
    mins = (self._minima if rows is None
            else tf.gather(self._minima, rows, axis=0))
    maxs = maxs[:, tf.newaxis]
    mins = mins[:, tf.newaxis]
    q = tf.gather(self._q, rows, axis=0) if rows is not None else self._q

    weights = tf.cumsum(tf.nn.softmax(q, axis=-1), axis=-1)
    num_weights = tf.shape(weights)[0]

    weights = tf.concat([tf.zeros((num_weights, 1)), weights], axis=1)
    return weights * (maxs - mins) + mins

  def sliced_reconstruction(self, rows=None, cols=None):
    """Reconstructs the scale input."""
    return ops.soft_quantile_normalization(
        self.sliced_uv(rows, cols),
        self.sliced_quantiles(rows),
        target_weights=self.sliced_target_weights(rows),
        **self._kwargs)

  def debug_str(self, loss, epoch):
    return '{}: {} -->  loss={:.2f}'.format(
        self.__class__.__name__, epoch, loss)

  def to_dataset(self, x: tf.Tensor, batch_size):
    if batch_size is None or batch_size <= 0:
      return [(x, None)]

    indices = tf.range(tf.shape(x)[0])
    result = tf.data.Dataset.from_tensor_slices((x, indices))
    return result.shuffle(1024).batch(batch_size)

  def loss(self, x, rows=None, cols=None):
    reconstruction = self.sliced_reconstruction(rows, cols)
    reg = self.regularization(rows=rows, cols=cols)
    return measures.kl_divergence(x, reconstruction) + reg

  def update(self) -> float:
    """Goes once over the data and update the training variables.

    Returns:
     The loss (float) over the epoch.
    """
    mu = tf.keras.metrics.Mean()
    mu.reset_states()
    for _, (batch, indices) in enumerate(self.dataset):
      with tf.GradientTape() as tape:
        loss = self.loss(batch, rows=indices, cols=None)
        grad = tape.gradient(loss, self.trainable_variables)
      self._optimizer.apply_gradients(zip(grad, self.trainable_variables))
      mu.update_state(loss)
    loss = mu.result()
    self.losses.append(loss)
    return loss

  def __call__(self, inputs: tf.Tensor, epochs: int = 5, reset: bool = True):
    if reset:
      self.reset(inputs)

    for epoch in range(epochs):
      loss = self.update()
      print(self.debug_str(loss, epoch), end='\r')
