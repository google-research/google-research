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

# Lint as: python3
"""Trains a model to predict the last value in a TrainingDataItem.

That is, takes a tuple of (inputs, sub-expression, target-expression),
and predict whether sub-expression is a sub-expression of target-expression.
"""

import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from bustle.bustle_python import model_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('training_data_file', '/tmp/training_data.json',
                    'file containing the training data as JSON.')
flags.DEFINE_string('output_dir', '/tmp/saved_model_dir',
                    'directory for saving the trained model.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train for.')


def train_model(x_train, y_train, x_valid, y_valid):
  """Trains the model, given the dataset as numpy arrays."""
  vocab_size = len(model_utils.VAL_IDXS)
  print('vocab_size: {}'.format(vocab_size))
  embedding_dim = 2
  data_dim = x_train.shape[1]
  expanded_dim = data_dim * embedding_dim
  print('expanded_dim = {}'.format(expanded_dim))

  net = tf.keras.Sequential()
  net.add(
      tf.keras.layers.Embedding(
          input_dim=vocab_size, output_dim=embedding_dim,
          input_length=data_dim))
  net.add(tf.keras.layers.Flatten())
  assert net.output_shape[1] == expanded_dim
  net.add(tf.keras.layers.Dense(64, activation='relu'))
  net.add(tf.keras.layers.Dense(64, activation='relu'))
  net.add(tf.keras.layers.Dense(1, activation=None))

  optimizer = tf.keras.optimizers.Adam(
      learning_rate=1e-5,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)
  net.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=['accuracy'])

  print(net.summary())

  # This can return a history dict if we want it
  net.fit(
      x_train,
      y_train,
      batch_size=128,
      epochs=FLAGS.epochs,
      # should run on this at end of epochs:
      validation_data=(x_valid, y_valid))

  net.save(os.path.expanduser(FLAGS.output_dir), save_format='tf')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  np.random.seed(0)

  x_train, y_train, _, x_valid, y_valid, _, _ = model_utils.load_data_from_json(
      os.path.expanduser(FLAGS.training_data_file))
  train_model(x_train, y_train, x_valid, y_valid)


if __name__ == '__main__':
  app.run(main)
