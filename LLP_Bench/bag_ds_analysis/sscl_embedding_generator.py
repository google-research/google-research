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

"""Generates Embeddings to compute bag distances for Criteo SSCL dataset."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import analysis_constants
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


tfk = tf.keras
tfkl = tf.keras.layers


_BATCH_SIZE = flags.DEFINE_integer('batch_size', 4096, 'Batch size.')
_EMBED_SIZE = flags.DEFINE_integer('embed_size', 16, 'Embed size.')
_LR = flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
_LOSS = flags.DEFINE_enum('loss', 'mse', ['mse', 'poisson'], 'Loss.')
_EPOCHS = flags.DEFINE_integer('epochs', 300, 'Epochs.')


class CustomModel(tfk.Model):
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
    self.sequential_layers.add(tfkl.Dense(units=1, activation=None))

  def compile(
      self,
      optimizer = 'adam',
      loss = tfk.losses.MeanSquaredError(),
      metrics = None,
      run_eagerly = False,
  ):
    """Compile."""
    super().compile(
        optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly
    )

  def get_rep(self):
    """Get penultimate layer representation model."""
    self.sequential_layers.trainable = False
    return self.embedding_layers

  def call(self, inp):
    """Call."""
    x_catg, x_numer = inp
    embeddings = [x_numer]
    for i in range(self.n_catg):
      embeddings.append(self.embedding_layers[i](x_catg[:, i]))
    x = tf.concat(embeddings, axis=-1)
    return self.sequential_layers(x)

  def train_step(
      self, batch
  ):
    """Train step."""
    x_catg, x_numer, y = batch
    with tf.GradientTape() as tape:
      y_pred = self((x_catg, x_numer))
      loss = self.compiled_loss(y, y_pred)
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


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Program Started')
  data_df = pd.read_csv(
      '../data/preprocessed_dataset/preprocessed_criteo_sscl.csv'
  )
  logging.info('Data Loaded')
  target = 'Y'
  numerical_cols = ['N' + str(i) for i in range(1, 4)]
  categorical_cols = ['C' + str(i) for i in range(1, 18)]
  train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=1)
  x_catg, x_numer, y = (
      np.array(train_df[categorical_cols]),
      np.array(train_df[numerical_cols]),
      np.array(train_df[target]),
  )
  train_ds = tf.data.Dataset.from_tensor_slices((
      tf.convert_to_tensor(x_catg, dtype=tf.float32),
      tf.convert_to_tensor(x_numer, dtype=tf.float32),
      tf.convert_to_tensor(y, dtype=tf.float32),
  )).batch(_BATCH_SIZE.value)
  x_catg_test, x_numer_test, y_test = (
      np.array(test_df[categorical_cols]),
      np.array(test_df[numerical_cols]),
      np.array(test_df[target]),
  )
  test_ds = tf.data.Dataset.from_tensor_slices((
      tf.convert_to_tensor(x_catg_test, dtype=tf.float32),
      tf.convert_to_tensor(x_numer_test, dtype=tf.float32),
      tf.convert_to_tensor(y_test, dtype=tf.float32),
  )).batch(_BATCH_SIZE.value)
  logging.info('Datasets Created')
  vocab_sizes = analysis_constants.VOCAB_SIZES
  model = CustomModel(
      n_catg=17, embed_size=_EMBED_SIZE.value, vocab_sizes=vocab_sizes
  )
  logging.info('Model Created')
  opt = tfk.optimizers.Adam(learning_rate=_LR.value)
  loss = {
      'mse': tfk.losses.MeanSquaredError(
          reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
      ),
      'poisson': tfk.losses.Poisson(
          reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
      ),
  }[_LOSS.value]
  model.compile(
      optimizer=opt,
      loss=loss,
      metrics=[tfk.metrics.MeanSquaredError(name='mse')],
  )
  earlystopping_callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_mse', patience=5, mode='min', restore_best_weights=True
  )
  hist = model.fit(
      train_ds,
      validation_data=test_ds,
      epochs=_EPOCHS.value,
      callbacks=[earlystopping_callback],
  )
  logging.info(
      'Model Training Complete, Best MSE: %d', min(hist.history['val_mse'])
  )
  embeddings = model.get_rep()
  embedding_dir = '../results/autoint_embeddings/' + _LOSS.value + '_'
  for i in range(model.n_catg):
    file_name = embedding_dir + 'C' + str(i + 1) + '.npy'
    np.save(file_name, embeddings[i].get_weights())
  logging.info('Program Completed')


if __name__ == '__main__':
  app.run(main)
