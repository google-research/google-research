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

"""Experiments with mnist data.

This experiment uses MNIST handwritten digit database, and train on it with
classifiers of varying complexity. The error metric was taken to be the maximum
of the error rates for each digit.

See the paper for a detailed explanation of the experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS
flags.DEFINE_bool('uniform_weights', False, 'whether to use uniform weights')
flags.DEFINE_bool('random_weights', False, 'whether to use random weights')
flags.DEFINE_bool('random_alpha', False, 'whether to use random alphas')
flags.DEFINE_float('sampling_radius', 1.0, 'radius in the sampling ball')
flags.DEFINE_integer('num_parallel_alphas', 20, 'number of parallel alphas')
flags.DEFINE_integer('num_alpha_batches', 10, 'number of alpha batches')
flags.DEFINE_integer('classifier_hidden_nodes', 50, 'classifier hidden nodes')

LEARNING_RATE = 0.001
TRAINING_STEPS = 10000
BATCH_SIZE = 100
INPUT_DIM = 784
OUTPUT_DIM = 10
TRAIN_INPUT_SIZE = 60000


def metric(y, logits, all_digits=False):
  result = np.sum(
      y * (np.argmax(y, axis=1) != np.argmax(logits, axis=1))[:, None],
      axis=0) / np.sum(
          y, axis=0)
  if all_digits:
    return result
  else:
    return np.max(result)


def classifier(x):
  layer1 = tf.layers.dense(
      inputs=x, units=FLAGS.classifier_hidden_nodes, activation=tf.sigmoid)
  logits = tf.layers.dense(inputs=layer1, units=OUTPUT_DIM)
  return logits


def optimization(logits, y, random_weights, alpha, learning_rate):
  if FLAGS.random_weights:
    weights = random_weights
  else:
    weights = tf.sigmoid(tf.matmul(y, alpha))
  weights /= tf.reduce_mean(weights)
  loss = tf.losses.sigmoid_cross_entropy(
      multi_class_labels=y, logits=logits, weights=weights)
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  return optimizer, loss


def sample_from_ball(size=(1, 1), sampling_radius=1):
  count, dim = size
  points = np.random.normal(size=size)
  points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
  scales = sampling_radius * np.power(
      np.random.uniform(size=(count, 1)), 1 / dim)
  points *= scales
  return points


def main(_):
  mnist = input_data.read_data_sets('/tmp/data/', one_hot=True, seed=12345)
  random_weight_vector = np.random.uniform(
      low=0.1, high=1.9, size=TRAIN_INPUT_SIZE)

  x = tf.placeholder(tf.float32, shape=(None, INPUT_DIM), name='x')
  y = tf.placeholder(tf.float32, shape=(None, OUTPUT_DIM), name='y')
  weight = tf.placeholder(tf.float32, shape=(None, OUTPUT_DIM), name='weight')
  parallel_alphas = tf.placeholder(
      tf.float32,
      shape=(FLAGS.num_parallel_alphas, OUTPUT_DIM),
      name='parallel_alphas')
  unstack_parallel_alphas = tf.unstack(parallel_alphas, axis=0)
  parallel_logits = []
  parallel_losses = []
  parallel_optimizers = []
  validation_metrics = []
  test_metrics = []
  all_test_metrics = []

  with tf.variable_scope('classifier'):
    for alpha_index in range(FLAGS.num_parallel_alphas):
      logits = classifier(x)
      alpha = tf.reshape(
          unstack_parallel_alphas[alpha_index], shape=[OUTPUT_DIM, 1])
      optimizer, loss = optimization(logits, y, weight, alpha, LEARNING_RATE)
      parallel_logits.append(logits)
      parallel_losses.append(loss)
      parallel_optimizers.append(optimizer)

  init = tf.global_variables_initializer()
  classifiers_init = tf.variables_initializer(
      tf.global_variables(scope='classifier'))
  with tf.Session() as sess:
    sess.run(init)

    # GetCandidatesAlpha (Algorithm 2 in paper)
    sample_alphas = np.zeros(shape=(0, OUTPUT_DIM))
    for alpha_batch_index in range(FLAGS.num_alpha_batches):
      sess.run(classifiers_init)
      if FLAGS.uniform_weights:
        alpha_batch = np.zeros(shape=(FLAGS.num_parallel_alphas, OUTPUT_DIM))
      elif FLAGS.random_alpha or alpha_batch_index < 1:
        alpha_batch = sample_from_ball(
            size=(FLAGS.num_parallel_alphas, OUTPUT_DIM),
            sampling_radius=FLAGS.sampling_radius)
        sample_alphas = np.concatenate([sample_alphas, alpha_batch])
      else:
        # Use LCB to generate candidates.
        alpha_batch = np.zeros(shape=(0, OUTPUT_DIM))
        sample_metrics = validation_metrics[:]
        for alpha_index in range(FLAGS.num_parallel_alphas):
          kernel = RBF(
              length_scale=FLAGS.sampling_radius,
              length_scale_bounds=(FLAGS.sampling_radius * 1e-3,
                                   FLAGS.sampling_radius *
                                   1e3)) * ConstantKernel(1.0, (1e-3, 1e3))
          gp = GaussianProcessRegressor(
              kernel=kernel, alpha=1e-4).fit(sample_alphas,
                                             np.log1p(sample_metrics))
          candidates = sample_from_ball((10000, OUTPUT_DIM),
                                        FLAGS.sampling_radius)

          metric_mles, metric_stds = gp.predict(candidates, return_std=True)
          metric_lcbs = np.maximum(
              np.expm1(metric_mles - 1.0 * metric_stds), 0.0)
          metric_lcbs += np.random.random(
              size=metric_lcbs.shape) * 0.001  # break ties
          best_index = np.argmin(metric_lcbs)

          best_alpha = [candidates[best_index]]
          best_alpha_metric_estimate = np.minimum(
              np.expm1(metric_mles[best_index] + 1.0 * metric_stds[best_index]),
              1.0)
          alpha_batch = np.concatenate([alpha_batch, best_alpha])

          sample_alphas = np.concatenate([sample_alphas, best_alpha])
          sample_metrics.append(best_alpha_metric_estimate)

      # Training classifiers
      for step in range(TRAINING_STEPS):
        batch_index = range(step * BATCH_SIZE % TRAIN_INPUT_SIZE,
                            step * BATCH_SIZE % TRAIN_INPUT_SIZE + BATCH_SIZE)
        (batch_x, batch_y) = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
        batch_weight = [
            [random_weight_vector[i]] * OUTPUT_DIM for i in batch_index
        ]
        _, _ = sess.run(
            [parallel_optimizers, parallel_losses],
            feed_dict={
                x: batch_x,
                y: batch_y,
                weight: batch_weight,
                parallel_alphas: alpha_batch,
            })

      parallel_validation_logits = sess.run(
          parallel_logits,
          feed_dict={
              x: mnist.validation.images,
              y: mnist.validation.labels,
          })
      parallel_validation_metrics = [
          metric(mnist.validation.labels, validation_logits, all_digits=False)
          for validation_logits in parallel_validation_logits
      ]
      validation_metrics.extend(parallel_validation_metrics)

      parallel_test_logits = sess.run(
          parallel_logits,
          feed_dict={
              x: mnist.test.images,
              y: mnist.test.labels,
          })
      parallel_test_metrics = [
          metric(mnist.test.labels, test_logits, all_digits=False)
          for test_logits in parallel_test_logits
      ]
      test_metrics.extend(parallel_test_metrics)

      parallel_all_test_metrics = [
          metric(mnist.test.labels, test_logits, all_digits=True)
          for test_logits in parallel_test_logits
      ]
      all_test_metrics.extend(parallel_all_test_metrics)

  best_observed_index = np.argmin(validation_metrics)
  print('[metric] validation={}'.format(
      validation_metrics[best_observed_index]))
  print('[metric] test={}'.format(test_metrics[best_observed_index]))
  for i in range(10):
    print('[all test metrics] {}={}'.format(
        i, all_test_metrics[best_observed_index][i]))


if __name__ == '__main__':
  app.run(main)
