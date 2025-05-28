# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Model class."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf


tfk = tf.keras
tfkl = tf.keras.layers


def my_model(multihot_dim):
  """MLP."""
  internal_model = tfk.Sequential()
  internal_model.add(
      tfkl.CategoryEncoding(
          num_tokens=multihot_dim, output_mode='multi_hot', sparse=True
      )
  )
  internal_model.add(tfkl.Dense(128, activation='relu'))
  internal_model.add(tfkl.Dense(64, activation='relu'))
  internal_model.add(tfkl.Dense(1, activation='sigmoid'))
  return internal_model


class MeanMapModel(tfk.Model):
  """Mean Map training model."""

  def __init__(self, multihot_dim, reg, mu_xy):
    """Constructor."""
    super().__init__()
    self.encoding_layers = tfkl.CategoryEncoding(
        num_tokens=multihot_dim, output_mode='multi_hot', sparse=True
    )
    self.linear_layer = tfkl.Dense(
        1, kernel_regularizer=tfk.regularizers.L2(l2=reg)
    )
    self.mu_xy = mu_xy

  def compile(
      self,
      optimizer = 'adam',
      metrics = None,
      run_eagerly = False,
  ):
    """Compile."""
    super().compile(
        optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly
    )

  def call(self, features):
    """Call."""
    return self.linear_layer(self.encoding_layers(features))

  def loss_function(self, y_pred):
    """Loss function."""
    return tf.reduce_sum(
        tf.math.log(tf.math.exp(y_pred) + tf.math.exp(-y_pred))
    ) - tf.cast(tf.shape(y_pred)[0], dtype=tf.float32) * (
        self.linear_layer(self.mu_xy)[0][0]
    )

  def train_step(
      self, batch
  ):
    """Train step."""
    x, y = batch
    with tf.GradientTape() as tape:
      y_pred = self(x)
      loss = self.loss_function(y_pred)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(
      self, batch
  ):
    """Test step."""
    x, y = batch
    y_pred = self(x)
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}


class CustomModel(tfk.Model):
  """Custom model with access to penultimate layer."""

  def __init__(self, multihot_dim):
    """Constructor."""
    super().__init__()
    self.final_layer = tfkl.Dense(units=1, activation='sigmoid')
    self.sequential_layers = tfk.Sequential()
    self.sequential_layers.add(
        tfkl.CategoryEncoding(
            num_tokens=multihot_dim, output_mode='multi_hot', sparse=True
        )
    )
    self.sequential_layers.add(tfkl.Dense(128, activation='relu'))
    self.sequential_layers.add(tfkl.Dense(64, activation='relu'))

  def compile(
      self,
      optimizer = 'adam',
      bag_loss = None,
      metrics = None,
      run_eagerly = False,
  ):
    """Compile."""
    super().compile(
        optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly
    )
    self.bag_loss = bag_loss

  def get_rep(self):
    """Get penultimate layer representation model."""
    self.sequential_layers.trainable = False
    return self.sequential_layers

  def call(self, features):
    """Call."""
    return self.final_layer(self.sequential_layers(features))

  def train_step(
      self, batch
  ):
    """Train step."""
    x, y = batch
    with tf.GradientTape() as tape:
      y_pred = self(x)
      loss = self.bag_loss(x, y, y_pred)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(
      self, batch
  ):
    """Test step."""
    x, y = batch
    y_pred = self(x)
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}


class CustomModelRegression(tfk.Model):
  """Custom model with access to penultimate layer."""

  def __init__(
      self, n_catg, embed_size, vocab_sizes
  ):
    """Constructor."""
    super().__init__()
    self.n_catg = n_catg
    self.embedding_layers = []
    for i in range(n_catg):
      self.embedding_layers.append(tfkl.Embedding(vocab_sizes[i], embed_size))
    self.sequential_layers = tfk.Sequential()
    self.sequential_layers.add(tfkl.Dense(128, activation='relu'))
    self.sequential_layers.add(tfkl.Dense(64, activation='relu'))
    self.final_layer = tfkl.Dense(units=1, activation=None)

  def compile(
      self,
      bag_loss = lambda x, y, z, w: 0,
      optimizer = 'adam',
      metrics = None,
      run_eagerly = False,
  ):
    """Compile."""
    super().compile(
        optimizer=optimizer, loss=None, metrics=metrics, run_eagerly=run_eagerly
    )
    self.bag_loss = bag_loss

  def get_rep(self):
    """Get penultimate layer representation model."""
    self.sequential_layers.trainable = False
    return self.embedding_layers

  def penultimate_rep(self, x_catg, x_numer):
    embeddings = [x_numer]
    for i in range(self.n_catg):
      embeddings.append(self.embedding_layers[i](x_catg[:, i]))
    x = tf.concat(embeddings, axis=-1)
    return self.sequential_layers(x)

  def call(self, inp):
    """Call."""
    x_catg, x_numer = inp
    embeddings = [x_numer]
    for i in range(self.n_catg):
      embeddings.append(self.embedding_layers[i](x_catg[:, i]))
    x = tf.concat(embeddings, axis=-1)
    return self.final_layer(self.sequential_layers(x))

  def train_step(
      self, batch
  ):
    """Train step."""
    x_catg, x_numer, y = batch
    with tf.GradientTape() as tape:
      y_pred = self((x_catg, x_numer))
      loss = self.bag_loss(x_catg, x_numer, y, y_pred)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(
      self, batch
  ):
    """Test step."""
    x_catg, x_numer, y = batch
    y_pred = self((x_catg, x_numer))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}
