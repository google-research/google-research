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
"""Train and evaluate a RNN language model on quantum bitstrings.

Train and evaluate a RNN language model on quantum bitstrings. This trains a LM,
then samples from it to evaluate fidelity.
"""

import os
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import scipy
from scipy import stats
import scipy.special
import tensorflow.compat.v2 as tf

from quantum_sample_learning import data_loader


flags.DEFINE_string('checkpoint_dir', './training_checkpoints_lm',
                    'Where to save checkpoints')
flags.DEFINE_string('save_data', '', 'Where to generate data (optional).')
flags.DEFINE_string('eval_sample_file', '',
                    'A file of samples to evaluate (optional).')
flags.DEFINE_boolean(
    'eval_has_separator', False,
    'Set if the numbers in the samples are separated by spaces.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('eval_samples', 500000,
                     'Number of samples for evaluation.')
flags.DEFINE_integer('training_eval_samples', 4000,
                     'Number of samples for evaluation during training.')
flags.DEFINE_integer('num_qubits', 20, 'Number of qubits to be learnt')
flags.DEFINE_integer('rnn_units', 256, 'Number of RNN hidden units.')
flags.DEFINE_integer(
    'num_moments', -2,
    'If > 12, then use training data generated with this number of moments.')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_boolean('use_adamax', False,
                     'Use the Adamax optimizer.')
flags.DEFINE_boolean('eval_during_training', True,
                     'Perform eval while training.')
flags.DEFINE_float('kl_smoothing', 1, 'The KL smoothing factor.')
flags.DEFINE_boolean(
    'save_test_counts', False, 'Whether to save test counts distribution.')
flags.DEFINE_string(
    'probabilities_path', 'quantum_sample_learning/data/q12c0.txt',
    'The path of the theoretical distribution')
flags.DEFINE_string(
    'experimental_bitstrings_path',
    'quantum_sample_learning/data/experimental_samples_q12c0d14.txt',
    'The path of the experiment measurements')
flags.DEFINE_integer('train_size', 500000, 'Training set size to generate')
flags.DEFINE_boolean('use_theoretical_distribution', True,
                     'Use the theoretical bitstring distribution.')
flags.DEFINE_integer(
    'subset_parity_size', 0,
    'size of the subset for reordering the bit strings according to the '
    'parity defined by the bit string of length specified here')
flags.DEFINE_boolean('random_subset', False,
                     'Randomly choose which subset of bits to '
                     'evaluate the subset parity on.')
flags.DEFINE_boolean('porter_thomas', False,
                     'Sample from Poter-Thomas distribution')


FLAGS = flags.FLAGS


_BUFFER_SIZE = 10000


class SampleEvalCallback(tf.keras.callbacks.Callback):
  """Class for returning sampling."""

  def __init__(self, probabilities, theory_fidelity, theory_logistic_fidelity):
    super(SampleEvalCallback, self).__init__()
    self.probabilities = probabilities
    self.theory_fidelity = theory_fidelity
    self.theory_logistic_fidelity = theory_logistic_fidelity

  def on_epoch_end(self, epoch, logs=None):
    del logs
    evaluate(
        weights=self.model.get_weights(),
        probabilities=self.probabilities,
        eval_samples=FLAGS.training_eval_samples,
        epoch=epoch,
        theory_fid_linear=self.theory_fidelity,
        theory_fid_logistic=self.theory_logistic_fidelity)


def build_model(batch_size, stateful=False):
  """Builds model."""
  model = tf.keras.Sequential([
      tf.keras.layers.LSTM(
          FLAGS.rnn_units,
          return_sequences=True,
          stateful=stateful,
          recurrent_initializer='glorot_uniform',
          batch_input_shape=[batch_size, None, 1]),
      tf.keras.layers.Dense(1)
  ])
  return model


def train(data, probabilities, theory_fidelity, theory_logistic_fidelity):
  """Trains the model."""
  logging.info('Start training.')
  model = build_model(FLAGS.batch_size)

  def loss(labels, logits):
    return tf.keras.losses.binary_crossentropy(labels, logits, True)

  if FLAGS.use_adamax:
    optimizer = tf.keras.optimizers.Adamax(FLAGS.learning_rate)
  else:
    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  model.compile(optimizer=optimizer, loss=loss)
  model.summary()

  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')

  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix, save_weights_only=True, save_best_only=True)
  sample_eval_callback = SampleEvalCallback(
      probabilities, theory_fidelity, theory_logistic_fidelity)
  callbacks = [checkpoint_callback]
  if FLAGS.eval_during_training:
    callbacks.append(sample_eval_callback)
  model.fit(data, epochs=FLAGS.epochs, callbacks=callbacks)
  return model


def calculate_metrics(sample_data, probabilities, epoch):
  """Calculates the metrics."""
  def convert_bitstring_array_to_probabilities(array):
    return probabilities[data_loader.convert_binary_digits_array_to_bitstrings(
        array)]

  sampled_probabilities = convert_bitstring_array_to_probabilities(sample_data)
  mean_p = np.mean(sampled_probabilities)
  logging.info('Min sampled probability %f', np.min(sampled_probabilities))
  logging.info('Max sampled probability %f', np.max(sampled_probabilities))
  logging.info('Mean sampled probability %f', mean_p)
  logging.info('Space size %d', probabilities.size)
  fidelity = probabilities.size * mean_p - 1
  logistic_fidelity = np.log(probabilities.size) + np.euler_gamma + np.mean(
      np.log(sampled_probabilities))
  logging.info('Linear Fidelity: %f', fidelity)
  logging.info('Logistic Fidelity: %f', logistic_fidelity)
  int_array = data_loader.convert_binary_digits_array_to_bitstrings(sample_data)
  values_in_test, counts_in_test = np.unique(int_array, return_counts=True)
  test_counts = np.zeros(2 ** FLAGS.num_qubits)
  test_counts[values_in_test] = counts_in_test
  chisquare_result = stats.chisquare(f_obs=test_counts,
                                     f_exp=probabilities * len(int_array))
  logging.info('chisquare p value: %f', chisquare_result.pvalue)
  if FLAGS.save_test_counts:
    if FLAGS.num_qubits > 25:
      raise ValueError(
          f'num_qubits ({FLAGS.num_qubits}) is too large to save test_counts')
    with tf.io.gfile.GFile(
        os.path.join(FLAGS.checkpoint_dir, f'f_obs_{epoch}.npy'), 'wb') as f:
      np.save(f, test_counts)
    with tf.io.gfile.GFile(
        os.path.join(FLAGS.checkpoint_dir, f'f_exp_{epoch}.npy'), 'wb') as f:
      np.save(f, probabilities * len(int_array))

  train_array = np.zeros((2 ** FLAGS.num_qubits))
  for i in data_loader.convert_binary_digits_array_to_bitstrings(sample_data):
    train_array[i] += 1
  # Smooth and normalize
  adj_train_array = (train_array + FLAGS.kl_smoothing) / np.sum(train_array)
  adj_train_array = adj_train_array / np.sum(adj_train_array)
  kl_div = np.sum(scipy.special.kl_div(probabilities, adj_train_array))
  logging.info('KL Divergence: %f', kl_div)

  return fidelity, logistic_fidelity, kl_div, chisquare_result.pvalue


def evaluate(
    weights,
    probabilities,
    eval_samples,
    epoch,
    theory_fid_linear,
    theory_fid_logistic):
  """Evaluates bitsting samples generated by the LSTMs."""
  model = build_model(FLAGS.batch_size, True)
  model.build(tf.TensorShape([FLAGS.batch_size, None, 1]))
  model.set_weights(weights)
  # Whole sequence sampling and fidelity
  eval_samples = eval_samples // FLAGS.batch_size * FLAGS.batch_size
  sample_data = np.zeros(((eval_samples, FLAGS.num_qubits)), np.int32)
  sample_n = 0
  model.summary()

  while sample_n < eval_samples:
    model.reset_states()
    input_eval = tf.zeros([FLAGS.batch_size, 1, 1])
    output_eval = tf.reshape(model(input_eval), [FLAGS.batch_size])
    output_prob = 1 / (1 + np.exp(-output_eval.numpy()))
    sample_data[sample_n:sample_n + FLAGS.batch_size,
                0] = np.random.binomial(1, output_prob)
    for i in range(FLAGS.num_qubits - 1):
      input_eval = tf.cast(
          tf.reshape(sample_data[sample_n:sample_n + FLAGS.batch_size, i],
                     [FLAGS.batch_size, 1, 1]), tf.float32)
      output_eval = tf.reshape(model(input_eval), [FLAGS.batch_size])
      output_prob = 1 / (1 + np.exp(-output_eval.numpy()))
      sample_data[sample_n:sample_n + FLAGS.batch_size,
                  i + 1] = np.random.binomial(1, output_prob)
    sample_n += FLAGS.batch_size

  fidelity, logistic_fidelity, kl_div, pvalue = calculate_metrics(
      sample_data, probabilities, epoch)

  logging.info('Number of bitstrings used in eval: %f', sample_data.shape[0])
  logging.info('chi2_pvalue: %f', pvalue)
  logging.info('theoretical_linear_xeb: %f', theory_fid_linear)
  logging.info('theoretical_logistic_xeb: %f', theory_fid_logistic)
  logging.info('linear_xeb: %f', fidelity)
  logging.info('logistic_xeb: %f', logistic_fidelity)
  logging.info('kl_div: %f', kl_div)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not tf.io.gfile.exists(FLAGS.checkpoint_dir):
    tf.io.gfile.makedirs(FLAGS.checkpoint_dir)
    logging.info('Model will be saved to: %s', FLAGS.checkpoint_dir)

  (
      train_data,
      probabilities,
      theory_fidelity,
      theory_logistic_fidelity
      ) = data_loader.load_data(
          num_qubits=FLAGS.num_qubits,
          use_theoretical_distribution=FLAGS.use_theoretical_distribution,
          probabilities_path=FLAGS.probabilities_path,
          subset_parity_size=FLAGS.subset_parity_size,
          random_subset=FLAGS.random_subset,
          porter_thomas=FLAGS.porter_thomas,
          experimental_bitstrings_path=FLAGS.experimental_bitstrings_path,
          train_size=FLAGS.train_size)
  data = tf.data.Dataset.from_tensor_slices(train_data)

  if FLAGS.save_data:
    with tf.io.gfile.GFile(FLAGS.save_data, 'w') as f:
      for row in train_data:
        line = ''
        for num in row:
          line += str(num) + ' '
        line += '\n '
        f.write(line)
    sys.exit()

  if FLAGS.eval_sample_file:
    sample_array = np.zeros((500000, FLAGS.num_qubits))
    with tf.io.gfile.GFile(FLAGS.eval_sample_file) as f:
      i = 0
      for row in f:
        if FLAGS.eval_has_separator:
          sample = [int(c) for c in list(row.rstrip().split())]
        else:
          sample = [int(c) for c in list(row.rstrip())]
        sample_array[i] = np.asarray(sample)
        i += 1
    sample_array = sample_array[:i, :]
    calculate_metrics(sample_array, probabilities, epoch=0)
    sys.exit()

  def _build_example(chunk):
    input_seq = tf.cast(tf.concat([[0], chunk], 0), tf.float32)
    target = tf.concat([chunk, [0]], 0)
    input_seq = input_seq[:-1]
    target = target[:-1]
    return tf.expand_dims(input_seq, 1), tf.expand_dims(target, 1)

  data = data.map(_build_example).shuffle(_BUFFER_SIZE).batch(
      FLAGS.batch_size, drop_remainder=True)
  model = train(data, probabilities, theory_fidelity, theory_logistic_fidelity)
  evaluate(
      weights=model.get_weights(),
      probabilities=probabilities,
      eval_samples=FLAGS.eval_samples,
      epoch=FLAGS.epochs,
      theory_fid_linear=theory_fidelity,
      theory_fid_logistic=theory_logistic_fidelity)


if __name__ == '__main__':
  app.run(main)
