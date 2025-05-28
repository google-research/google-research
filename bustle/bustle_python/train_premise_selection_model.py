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
"""Trains a model to predict Operation presence in a TrainingDataItem.

That is, takes a tuple of (inputs, sub-expression, target-expression),
ignore the target expression, and use the signature computed on the inputs
and the sub-expression to predict which operations are included in the
sub-expression.
We can then use these predictions for premise selection.
"""

import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from bustle.bustle_python import model_utils

FLAGS = flags.FLAGS
LABEL_SMOOTHING_COEFFICIENT = 0.1

flags.DEFINE_string('training_data_file', '/tmp/training_data.json',
                    'file containing the training data as JSON.')
flags.DEFINE_string('output_dir', '/tmp/saved_premise_selection_model_dir',
                    'directory for saving the trained model.')
flags.DEFINE_integer('epochs', 1000,
                     'number of epochs to train for.')


def train_model(x_train, op_train, x_valid, op_valid):
  """Trains the model, given the dataset as numpy arrays."""
  vocab_size = len(model_utils.VAL_IDXS)
  print('vocab_size: {}'.format(vocab_size))
  embedding_dim = 4
  data_dim = x_train.shape[1]
  output_shape = op_train.shape[1]
  expanded_dim = data_dim * embedding_dim

  net = tf.keras.Sequential()
  net.add(
      tf.keras.layers.Embedding(
          input_dim=vocab_size, output_dim=embedding_dim,
          input_length=data_dim))
  net.add(tf.keras.layers.Flatten())
  assert net.output_shape[1] == expanded_dim
  net.add(tf.keras.layers.Dense(512, activation='relu'))
  net.add(tf.keras.layers.Dense(256, activation='relu'))
  net.add(tf.keras.layers.Dense(128, activation='relu'))
  net.add(tf.keras.layers.Dense(output_shape, activation='sigmoid'))

  optimizer = tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)
  net.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.BinaryCrossentropy(
          from_logits=False, label_smoothing=LABEL_SMOOTHING_COEFFICIENT),
      metrics=[
          'binary_accuracy',
          # What % of predicted positives were true positives
          tf.keras.metrics.Precision(),
          # What % of all real positives did you predict were positive
          tf.keras.metrics.Recall()
      ])

  print(net.summary())

  # This can return a history dict if we want it
  net.fit(
      x_train,
      op_train,
      batch_size=128,
      epochs=FLAGS.epochs,
      # should run on this at end of epochs:
      validation_data=(x_valid, op_valid))

  # Compute the per-operation accuracies.
  predictions = net.predict(x_valid)
  predictions[predictions >= 0.5] = 1
  predictions[predictions < 0.5] = 0
  correct_predictions = (predictions == op_valid)
  per_operation_accuracies = np.mean(correct_predictions, axis=0)
  per_operation_fraction_tru = np.mean(op_valid, axis=0)
  print('Per Operation Accuracies: ')
  for acc, ground_truth in zip(per_operation_accuracies,
                               per_operation_fraction_tru):
    print('Accuracy: {:.3f}, Fraction Actually True: {:.3f}'.format(
        acc, ground_truth))

  true_positives = np.logical_and(predictions == op_valid, predictions == 1)
  false_positives = np.logical_and(predictions != op_valid, predictions == 1)
  true_negatives = np.logical_and(predictions == op_valid, predictions == 0)
  false_negatives = np.logical_and(predictions != op_valid, predictions == 0)

  per_op_tru_pos = np.mean(true_positives, axis=0)
  per_op_fls_pos = np.mean(false_positives, axis=0)
  per_op_tru_neg = np.mean(true_negatives, axis=0)
  per_op_fls_neg = np.mean(false_negatives, axis=0)

  print('Per Op True Positives: {}'.format(per_op_tru_pos))
  print('Per Op False Positives: {}'.format(per_op_fls_pos))
  print('Per Op True Negatives: {}'.format(per_op_tru_neg))
  print('Per Op False Negatives: {}'.format(per_op_fls_neg))

  print('Total True Positives: {:.3f}'.format(np.mean(per_op_tru_pos)))
  print('Total False Positives: {:.3f}'.format(np.mean(per_op_fls_pos)))
  print('Total True Negatives: {:.3f}'.format(np.mean(per_op_tru_neg)))
  print('Total False Negatives: {:.3f}'.format(np.mean(per_op_fls_neg)))

  val_binary_accuracy = np.mean(per_operation_accuracies)
  print('Manually Computed Val Binary Acc: {:.3f}'.format(val_binary_accuracy))

  net.save(os.path.expanduser(FLAGS.output_dir), save_format='tf')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  np.random.seed(0)
  np.set_printoptions(precision=2)

  (x_train, _, op_train, x_valid, _, op_valid,
   example_signature_size) = model_utils.load_data_from_json(
       FLAGS.training_data_file)
  # For premise selection, we only use the I/O example signatures. Zero out the
  # value signature part.
  x_train[:, example_signature_size:] = 0
  x_valid[:, example_signature_size:] = 0
  train_model(x_train, op_train, x_valid, op_valid)


if __name__ == '__main__':
  app.run(main)
