# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Experiments with crime data.

In this example, we examine MOEW on the Communities and Crime dataset from the
UCI Machine Learning Repository, which contains the violent crime rate of
communities. The goal is to predict whether a community has violent crime rate
per 100k population above 0.28.

In addition to obtaining an accurate classifier, we also aim to improve its
fairness. To this end, we divided the communities into 4 groups based on
the quartiles of white population percentage in each community. We seek a
classifier with high accuracy, but that has similar false positive rates (FPR)
across racial groups. Therefore, we evaluate classifiers based on two metrics:
overall accuracy across all communities and the difference between the highest
and lowest FPR across four racial groups (fairness violation).

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
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('training_data_path', '', 'training dataset')
flags.DEFINE_string('testing_data_path', '', 'testing dataset')
flags.DEFINE_string('validation_data_path', '', 'validation dataset')
flags.DEFINE_bool('uniform_weights', False, 'use uniform weights')
flags.DEFINE_bool('propensity_weights', False, 'use propensity weihts')
flags.DEFINE_bool('post_shift', True, 'apply post shift')
flags.DEFINE_float('sampling_radius', 1.0, 'sampling radius')

EMBEDDING_DIM = 4
LEARNING_RATE = 0.001
NUM_PARALLEL_ALPHAS = 5
NUM_ALPHA_BATCHES = 10
BATCH_SIZE = 100
TRAINING_STEPS = 10000
OUTPUT_DIM = 1
COVERAGE = 0.278

FEATURES = [
    'population', 'householdsize', 'agePct12t21', 'agePct12t29', 'agePct16t24',
    'agePct65up', 'numbUrban', 'pctUrban', 'medIncome', 'pctWWage',
    'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire',
    'medFamInc', 'perCapInc', 'NumUnderPov', 'PctPopUnderPov',
    'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu',
    'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
    'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
    'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom',
    'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
    'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
    'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup',
    'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
    'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos',
    'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart',
    'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ',
    'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg',
    'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
    'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LandArea', 'PopDens',
    'PctUsePubTrans', 'racepctblack', 'racePctAsian', 'racePctWhite',
    'racePctHisp'
]


def get_fpr(sorted_group, cut_index):
  n = np.sum(label == 0.0 for (_, label) in sorted_group)
  fp = np.sum(label == 0.0 for (_, label) in sorted_group[:cut_index])
  return float(fp) / n


def errors(sorted_group, cut_index):
  fp = np.sum(label == 0.0 for (_, label) in sorted_group[:cut_index])
  fn = np.sum(label == 1.0 for (_, label) in sorted_group[cut_index:])
  return fp + fn


def find_threshold(labels, predictions, wqs, post_shift):
  """Finds the post shift threshold for each group."""
  if post_shift:
    sorted_groups = []
    for q in range(1, 5):
      sorted_group = sorted(
          [(prediction[0], label)
           for (prediction, label, wq) in zip(predictions, labels, wqs)
           if wq == q],
          reverse=True)
      sorted_groups.append(sorted_group)

    cut_indices = [
        int(len(sorted_group) * COVERAGE) for sorted_group in sorted_groups
    ]

    for _ in range(1000):
      fprs = [
          get_fpr(sorted_group, cut_index)
          for (sorted_group, cut_index) in zip(sorted_groups, cut_indices)
      ]
      min_fpr_index = np.argmin(fprs)
      max_fpr_index = np.argmax(fprs)
      cut_indices[min_fpr_index] = min(cut_indices[min_fpr_index] + 1,
                                       len(sorted_groups[min_fpr_index]))
      cut_indices[max_fpr_index] = max(cut_indices[max_fpr_index] - 1, 0)

    thresholds = [sorted_groups[q][cut_indices[q]][0] for q in range(4)]
    return thresholds
  else:
    return [np.percentile(predictions, 100 - COVERAGE * 100)] * 4


def metrics(labels, predictions, wqs, thresholds):
  """Metric used for the experiment."""
  sorted_groups = []
  cut_indices = []
  for q in range(1, 5):
    sorted_group = sorted(
        [(prediciton[0], label)
         for (prediciton, label, wq) in zip(predictions, labels, wqs)
         if wq == q],
        reverse=True)
    sorted_groups.append(sorted_group)
    cut_index = int(len(sorted_group) * 0.3)
    for i in range(len(sorted_group)):
      if sorted_group[i][0] <= thresholds[q - 1]:
        cut_index = i
        break
    cut_indices.append(cut_index)

  fprs = [
      get_fpr(sorted_group, cut_index)
      for (sorted_group, cut_index) in zip(sorted_groups, cut_indices)
  ]
  fairness_violation = max(fprs) - min(fprs)

  errs = [
      errors(sorted_group, cut_index)
      for (sorted_group, cut_index) in zip(sorted_groups, cut_indices)
  ]
  acc = 1.0 - np.sum(errs) / len(labels)

  return (acc, fairness_violation)


def classifier(x):
  logits = tf.layers.dense(inputs=x, units=1)
  return logits


def optimization(logits, y, population, embedding, alpha):
  """Loss and optimization method."""
  if FLAGS.uniform_weights:
    weights = tf.ones(shape=tf.shape(population))
  else:
    weights = tf.where(
        tf.greater(population, 0.01), tf.fill(tf.shape(population), 0.16),
        tf.fill(tf.shape(population), 2.5))
    if not FLAGS.propensity_weights:
      weights = tf.sigmoid(tf.matmul(embedding, alpha)) * weights
  weights /= tf.reduce_mean(weights)
  loss = tf.losses.hinge_loss(labels=y, logits=logits, weights=weights)
  optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
  return optimizer, loss


def sample_from_ball(size=(1, 1), sampling_radius=2):
  count, dim = size
  points = np.random.normal(size=size)
  points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
  scales = np.power(np.random.uniform(size=(count, 1)), 1 / dim)
  points *= scales * sampling_radius
  return points


def main(_):
  num_steps_autoencoder = 0 if FLAGS.uniform_weights else TRAINING_STEPS

  training_df = pd.read_csv(FLAGS.training_data_path, header=0, sep=',')
  testing_df = pd.read_csv(FLAGS.testing_data_path, header=0, sep=',')
  validation_df = pd.read_csv(FLAGS.validation_data_path, header=0, sep=',')

  train_labels = training_df['label']
  validation_labels = validation_df['label']
  test_labels = testing_df['label']
  train_population = training_df['population']
  train_features = training_df[FEATURES]
  validation_features = validation_df[FEATURES]
  test_features = testing_df[FEATURES]
  train_wqs = training_df['racePctWhite_quantile']
  validation_wqs = validation_df['racePctWhite_quantile']
  test_wqs = testing_df['racePctWhite_quantile']

  tf.reset_default_graph()
  x = tf.placeholder(tf.float32, shape=(None, len(FEATURES)), name='x')
  y = tf.placeholder(tf.float32, shape=(None, OUTPUT_DIM), name='y')
  population = tf.placeholder(
      tf.float32, shape=(None, OUTPUT_DIM), name='population')

  xy = tf.concat([x, y], axis=1)
  autoencoder_layer1 = tf.layers.dense(
      inputs=xy, units=10, activation=tf.sigmoid)
  autoencoder_embedding_layer = tf.layers.dense(
      inputs=autoencoder_layer1, units=EMBEDDING_DIM, activation=tf.sigmoid)
  autoencoder_layer3 = tf.layers.dense(
      inputs=autoencoder_embedding_layer, units=10, activation=tf.sigmoid)
  autoencoder_out_x = tf.layers.dense(
      inputs=autoencoder_layer3, units=len(FEATURES))
  autoencoder_out_y_logits = tf.layers.dense(
      inputs=autoencoder_layer3, units=OUTPUT_DIM)

  autoencoder_y_loss = tf.losses.hinge_loss(
      labels=y, logits=autoencoder_out_y_logits)
  autoencoder_x_loss = tf.losses.mean_squared_error(
      labels=x, predictions=autoencoder_out_x)
  autoencoder_loss = autoencoder_x_loss + autoencoder_y_loss
  autoencoder_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
      autoencoder_loss)

  parallel_logits = []
  parallel_losses = []
  parallel_optimizers = []

  parallel_alphas = tf.placeholder(
      tf.float32,
      shape=(NUM_PARALLEL_ALPHAS, EMBEDDING_DIM),
      name='parallel_alphas')
  unstack_parallel_alphas = tf.unstack(parallel_alphas, axis=0)
  embedding = tf.placeholder(
      tf.float32, shape=(None, EMBEDDING_DIM), name='embedding')

  with tf.variable_scope('classifiers'):
    for alpha_index in range(NUM_PARALLEL_ALPHAS):
      logits = classifier(x)
      alpha = tf.reshape(
          unstack_parallel_alphas[alpha_index], shape=[EMBEDDING_DIM, 1])
      optimizer, loss = optimization(logits, y, population, embedding, alpha)

      parallel_logits.append(logits)
      parallel_losses.append(loss)
      parallel_optimizers.append(optimizer)

  init = tf.global_variables_initializer()
  classifiers_init = tf.variables_initializer(
      tf.global_variables(scope='classifiers'))

  kernel = RBF(
      length_scale=FLAGS.sampling_radius,
      length_scale_bounds=(FLAGS.sampling_radius * 1e-3, FLAGS.sampling_radius *
                           1e3)) * ConstantKernel(1.0, (1e-3, 1e3))

  alphas = np.zeros(shape=(0, EMBEDDING_DIM))
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
    for alpha_batch_index in range(NUM_ALPHA_BATCHES):
      sess.run(classifiers_init)
      if FLAGS.uniform_weights:
        alpha_batch = np.zeros(shape=(NUM_PARALLEL_ALPHAS, EMBEDDING_DIM))
      elif alpha_batch_index == 0:
        # We first start uniformly.
        alpha_batch = sample_from_ball(
            size=(NUM_PARALLEL_ALPHAS, EMBEDDING_DIM),
            sampling_radius=FLAGS.sampling_radius)
      else:
        # Use UCB to generate candidates.
        alpha_batch = np.zeros(shape=(0, EMBEDDING_DIM))
        sample_alphas = np.copy(alphas)
        sample_validation_metrics = [m[0] for m in validation_metrics]
        candidates = sample_from_ball(
            size=(10000, EMBEDDING_DIM), sampling_radius=FLAGS.sampling_radius)
        for alpha_index in range(NUM_PARALLEL_ALPHAS):
          gp = GaussianProcessRegressor(
              kernel=kernel, alpha=1e-1).fit(sample_alphas,
                                             sample_validation_metrics)

          metric_mles, metric_stds = gp.predict(candidates, return_std=True)
          metric_lcbs = metric_mles - 1.0 * metric_stds

          best_index = np.argmin(metric_lcbs)
          best_alpha = [candidates[best_index]]
          best_alpha_metric_ucb = metric_mles[best_index] \
            + 1.0 * metric_stds[best_index]
          alpha_batch = np.concatenate([alpha_batch, best_alpha])

          # Add candidate to the GP, assuming the metric observation is the LCB.
          sample_alphas = np.concatenate([sample_alphas, best_alpha])
          sample_validation_metrics.append(best_alpha_metric_ucb)

      # Training classifiers
      for _ in range(TRAINING_STEPS):
        batch_index = random.sample(range(len(train_labels)), BATCH_SIZE)
        batch_x = train_features.iloc[batch_index, :].values
        batch_y = train_labels.iloc[batch_index].values.reshape(BATCH_SIZE, 1)
        batch_population = train_population.iloc[batch_index].values.reshape(
            BATCH_SIZE, 1)
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
                population: batch_population,
                embedding: batch_embedding,
                parallel_alphas: alpha_batch,
            })

      parallel_train_logits = sess.run(
          parallel_logits,
          feed_dict={
              x: train_features.values,
              y: train_labels.values.reshape(len(train_labels), 1),
          })
      alphas = np.concatenate([alphas, alpha_batch])
      parallel_validation_logits = sess.run(
          parallel_logits,
          feed_dict={
              x: validation_features.values,
              y: validation_labels.values.reshape(len(validation_labels), 1),
          })
      parallel_test_logits = sess.run(
          parallel_logits,
          feed_dict={
              x: test_features.values,
              y: test_labels.values.reshape(len(test_labels), 1),
          })
      parallel_thresholds = [
          find_threshold(train_labels, train_logits, train_wqs,
                         FLAGS.post_shift)
          for train_logits in parallel_train_logits
      ]
      logits_thresholds = zip(parallel_validation_logits, parallel_thresholds)
      parallel_validation_metrics = [
          metrics(validation_labels, logits, validation_wqs, thresholds)
          for (logits, thresholds) in logits_thresholds
      ]
      validation_metrics.extend(parallel_validation_metrics)
      parallel_test_metrics = [
          metrics(test_labels, test_logits, test_wqs, thresholds)
          for (test_logits,
               thresholds) in zip(parallel_test_logits, parallel_thresholds)
      ]
      test_metrics.extend(parallel_test_metrics)

  best_observed_index = np.argmin([m[0] for m in validation_metrics])
  print('[metric] validation_acc={}'.format(
      validation_metrics[best_observed_index][0]))
  print('[metric] validation_violation={}'.format(
      validation_metrics[best_observed_index][1]))
  print('[metric] test_acc={}'.format(test_metrics[best_observed_index][0]))
  print('[metric] test_violation={}'.format(
      test_metrics[best_observed_index][1]))

  return 0


if __name__ == '__main__':
  app.run(main)
