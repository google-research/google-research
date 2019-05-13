# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Experiments with wine data.

In this experiment, we study the effect of the embedding dimension and the
number of batches on the MOEW performance. We use the wine reviews dataset from
Kaggle (https://www.kaggle.com/dbahri/wine-ratings). The task is to predict the
price of the wine using 39 Boolean features describing characteristic of the
wine and the quality score (points), for a total of 40 features. We calculate
the error in percentage of the correct price, and want the model to have good
accuracy across all price ranges. We thus set the test metric to be the worst of
the errors among 4 quartiles of the price.

See the paper for a detailed explanation of the experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('training_data_path', '', 'training dataset')
flags.DEFINE_string('testing_data_path', '', 'testing dataset')
flags.DEFINE_string('validation_data_path', '', 'validation dataset')
flags.DEFINE_bool('uniform_weights', False, 'whether to use uniform weights')
flags.DEFINE_float('sampling_radius', 5.0, 'radius in the sampling ball')
flags.DEFINE_integer('embedding_dim', 10, 'embedding dimension')
flags.DEFINE_integer('num_parallel_thetas', 20, 'number of parallel thetas')
flags.DEFINE_integer('num_theta_batches', 10, 'number of theta batches')
flags.DEFINE_integer('num_random_batches', 1, 'number of random batches')
flags.DEFINE_float('p_q_value', 1.0, 'value of p and q in GPR sampling')
flags.DEFINE_integer('autoencoder_hidden_nodes', 100, 'number of hidden nodes')

LEARNING_RATE = 0.001
BATCH_SIZE = 1000
TRAINING_STEPS = 10000
FEATURES = [
    'acid', 'angular', 'austere', 'barnyard', 'bright', 'butter', 'cassis',
    'charcoal', 'cigar', 'complex', 'cream', 'crisp', 'dense', 'earth',
    'elegant', 'flabby', 'flamboyant', 'fleshy', 'food friendly', 'grip',
    'hint of', 'intellectual', 'jam', 'juicy', 'laser', 'lees', 'mineral',
    'oak', 'opulent', 'refined', 'silk', 'steel', 'structure', 'tannin',
    'tight', 'toast', 'unctuous', 'unoaked', 'velvet', 'points'
]


def metric(labels, predictions, prices):
  """Metric used for the experiment."""
  labels = np.exp(labels)
  predictions = np.exp(predictions)
  price_quartile_1 = [1.0 if p <= 17 else 0.0 for p in prices]
  price_quartile_2 = [1.0 if p > 17 and p <= 25 else 0.0 for p in prices]
  price_quartile_3 = [1.0 if p > 25 and p <= 42 else 0.0 for p in prices]
  price_quartile_4 = [1.0 if p > 42 else 0.0 for p in prices]

  def errf(label, prediction):
    return np.abs(label - prediction) / label

  price_quartile_1_se = [
      pq * errf(label, prediction)
      for (label, prediction, pq) in zip(labels, predictions, price_quartile_1)
  ]
  price_quartile_2_se = [
      pq * errf(label, prediction)
      for (label, prediction, pq) in zip(labels, predictions, price_quartile_2)
  ]
  price_quartile_3_se = [
      pq * errf(label, prediction)
      for (label, prediction, pq) in zip(labels, predictions, price_quartile_3)
  ]
  price_quartile_4_se = [
      pq * errf(label, prediction)
      for (label, prediction, pq) in zip(labels, predictions, price_quartile_4)
  ]
  print([
      np.sum(price_quartile_1_se) / np.sum(price_quartile_1),
      np.sum(price_quartile_2_se) / np.sum(price_quartile_2),
      np.sum(price_quartile_3_se) / np.sum(price_quartile_3),
      np.sum(price_quartile_4_se) / np.sum(price_quartile_4)
  ])
  return np.max([
      np.sum(price_quartile_1_se) / np.sum(price_quartile_1),
      np.sum(price_quartile_2_se) / np.sum(price_quartile_2),
      np.sum(price_quartile_3_se) / np.sum(price_quartile_3),
      np.sum(price_quartile_4_se) / np.sum(price_quartile_4)
  ])


def regressor(x):
  layer1 = tf.layers.dense(inputs=x, units=20, activation=tf.sigmoid)
  layer2 = tf.layers.dense(inputs=layer1, units=10, activation=tf.sigmoid)
  output = tf.layers.dense(inputs=layer2, units=1)
  return output


def optimization(output, y, embedding, theta, learning_rate):
  weights = tf.sigmoid(tf.matmul(embedding, theta))
  weights /= tf.reduce_mean(weights)
  loss = tf.losses.mean_squared_error(
      labels=y, predictions=output, weights=weights)
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  return optimizer, loss


def sample_from_ball(size=(1, 1), sampling_radius=2):
  count, dim = size
  points = np.random.normal(size=size)
  points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
  scales = np.power(np.random.uniform(size=(count, 1)), 1 / dim)
  points *= scales * sampling_radius
  return points


def add_price_quantiles(df):
  prices = df['price']
  df['price_quartile_1'] = [1.0 if p <= 17 else 0.0 for p in prices]
  df['price_quartile_2'] = [1.0 if p > 17 and p <= 25 else 0.0 for p in prices]
  df['price_quartile_3'] = [1.0 if p > 25 and p <= 42 else 0.0 for p in prices]
  df['price_quartile_4'] = [1.0 if p > 42 else 0.0 for p in prices]
  return df


def main(_):
  num_parallel_thetas = FLAGS.num_parallel_thetas
  num_theta_batches = FLAGS.num_theta_batches
  num_steps_autoencoder = 0 if FLAGS.uniform_weights else TRAINING_STEPS

  input_dim = len(FEATURES)

  training_df = pd.read_csv(FLAGS.training_data_path, header=0, sep=',')
  testing_df = pd.read_csv(FLAGS.testing_data_path, header=0, sep=',')
  validation_df = pd.read_csv(FLAGS.validation_data_path, header=0, sep=',')

  add_price_quantiles(training_df)
  add_price_quantiles(testing_df)
  add_price_quantiles(validation_df)

  train_labels = np.log(training_df['price'])
  validation_labels = np.log(validation_df['price'])
  test_labels = np.log(testing_df['price'])
  train_features = training_df[FEATURES]
  validation_features = validation_df[FEATURES]
  test_features = testing_df[FEATURES]
  validation_price = validation_df['price']
  test_price = testing_df['price']

  tf.reset_default_graph()
  x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
  y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

  xy = tf.concat([x, y], axis=1)
  autoencoder_layer1 = tf.layers.dense(
      inputs=xy, units=100, activation=tf.sigmoid)
  autoencoder_embedding_layer = tf.layers.dense(
      inputs=autoencoder_layer1,
      units=FLAGS.embedding_dim,
      activation=tf.sigmoid)
  autoencoder_layer3 = tf.layers.dense(
      inputs=autoencoder_embedding_layer, units=100, activation=tf.sigmoid)
  autoencoder_out_x = tf.layers.dense(
      inputs=autoencoder_layer3, units=input_dim)
  autoencoder_out_y = tf.layers.dense(inputs=autoencoder_layer3, units=1)

  autoencoder_y_loss = tf.losses.mean_squared_error(
      labels=y, predictions=autoencoder_out_y)
  autoencoder_x_loss = tf.losses.mean_squared_error(
      labels=x, predictions=autoencoder_out_x)
  autoencoder_loss = autoencoder_x_loss + autoencoder_y_loss
  autoencoder_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
      autoencoder_loss)

  parallel_outputs = []
  parallel_losses = []
  parallel_optimizers = []

  parallel_thetas = tf.placeholder(
      tf.float32,
      shape=(num_parallel_thetas, FLAGS.embedding_dim),
      name='parallel_thetas')
  unstack_parallel_thetas = tf.unstack(parallel_thetas, axis=0)
  embedding = tf.placeholder(
      tf.float32, shape=(None, FLAGS.embedding_dim), name='embedding')

  with tf.variable_scope('regressors'):
    for theta_index in range(num_parallel_thetas):
      output = regressor(x)
      theta = tf.reshape(
          unstack_parallel_thetas[theta_index], shape=[FLAGS.embedding_dim, 1])
      optimizer, loss = optimization(output, y, embedding, theta, LEARNING_RATE)

      parallel_outputs.append(output)
      parallel_losses.append(loss)
      parallel_optimizers.append(optimizer)

  init = tf.global_variables_initializer()
  regressors_init = tf.variables_initializer(
      tf.global_variables(scope='regressors'))

  kernel = RBF(
      length_scale=FLAGS.sampling_radius,
      length_scale_bounds=(FLAGS.sampling_radius * 1e-3, FLAGS.sampling_radius *
                           1e3)) * ConstantKernel(1.0, (1e-3, 1e3))

  thetas = np.zeros(shape=(0, FLAGS.embedding_dim))
  validation_metrics = []
  test_metrics = []

  with tf.Session() as sess:
    sess.run(init)

    # Training autoencoder
    for _ in range(num_steps_autoencoder):
      batch_index = random.sample(range(len(train_labels)), BATCH_SIZE)
      batch_x = train_features.iloc[batch_index, :].values
      batch_y = train_labels.iloc[batch_index].values.reshape(BATCH_SIZE, 1)
      _, _ = sess.run([autoencoder_optimizer, autoencoder_loss],
                      feed_dict={
                          x: batch_x,
                          y: batch_y,
                      })

    # GetCandidatesAlpha (Algorithm 2 in paper)
    for theta_batch_index in range(num_theta_batches):
      sess.run(regressors_init)
      if FLAGS.uniform_weights:
        theta_batch = np.zeros(shape=(num_parallel_thetas, FLAGS.embedding_dim))
      elif theta_batch_index == 0:
        # We first start uniformly.
        theta_batch = sample_from_ball(
            size=(num_parallel_thetas, FLAGS.embedding_dim),
            sampling_radius=FLAGS.sampling_radius)
      else:
        # Use UCB to generate candidates.
        theta_batch = np.zeros(shape=(0, FLAGS.embedding_dim))
        sample_thetas = np.copy(thetas)
        sample_validation_metrics = validation_metrics[:]
        candidates = sample_from_ball(
            size=(10000, FLAGS.embedding_dim),
            sampling_radius=FLAGS.sampling_radius)
        for theta_index in range(num_parallel_thetas):
          gp = GaussianProcessRegressor(
              kernel=kernel, alpha=1e-4).fit(sample_thetas,
                                             sample_validation_metrics)

          metric_mles, metric_stds = gp.predict(candidates, return_std=True)
          metric_lcbs = metric_mles - FLAGS.p_q_value * metric_stds

          best_index = np.argmin(metric_lcbs)
          best_theta = [candidates[best_index]]
          best_theta_metric_ucb = metric_mles[best_index] \
            + FLAGS.p_q_value * metric_stds[best_index]
          theta_batch = np.concatenate([theta_batch, best_theta])

          # Add candidate to the GP, assuming the metric observation is the LCB.
          sample_thetas = np.concatenate([sample_thetas, best_theta])
          sample_validation_metrics.append(best_theta_metric_ucb)

      # Training regressors
      for _ in range(TRAINING_STEPS):
        batch_index = random.sample(range(len(train_labels)), BATCH_SIZE)
        batch_x = train_features.iloc[batch_index, :].values
        batch_y = train_labels.iloc[batch_index].values.reshape(BATCH_SIZE, 1)
        batch_embedding = sess.run(
            autoencoder_embedding_layer, feed_dict={
                x: batch_x,
                y: batch_y,
            })
        _, _ = sess.run(
            [parallel_optimizers, parallel_losses],
            feed_dict={
                x: batch_x,
                y: batch_y,
                embedding: batch_embedding,
                parallel_thetas: theta_batch,
            })

      parallel_validation_outputs = sess.run(
          parallel_outputs,
          feed_dict={
              x: validation_features.values,
              y: validation_labels.values.reshape(len(validation_labels), 1),
          })
      parallel_validation_metrics = [
          metric(validation_labels, validation_output, validation_price)
          for validation_output in parallel_validation_outputs
      ]
      thetas = np.concatenate([thetas, theta_batch])
      validation_metrics.extend(parallel_validation_metrics)

      parallel_test_outputs = sess.run(
          parallel_outputs,
          feed_dict={
              x: test_features.values,
              y: test_labels.values.reshape(len(test_labels), 1),
          })
      parallel_test_metrics = [
          metric(test_labels, test_output, test_price)
          for test_output in parallel_test_outputs
      ]
      test_metrics.extend(parallel_test_metrics)

  best_observed_index = np.argmin(validation_metrics)
  print('[metric] validation={}'.format(
      validation_metrics[best_observed_index]))
  print('[metric] test={}'.format(test_metrics[best_observed_index]))

  return 0


if __name__ == '__main__':
  app.run(main)
