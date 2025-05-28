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

"""Tensorflow implementation of LogME for classification.

You, Kaichao, et al. "Logme: Practical assessment of pre-trained models for
transfer learning." International Conference on Machine Learning. PMLR, 2021.
http://proceedings.mlr.press/v139/you21b/you21b.pdf

We base our code on the Optimized Fixed Point Iterations implemented in:
https://github.com/thuml/LogME, proposed in the arxiv 2021 paper (Algorithm 3):
"Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
https://arxiv.org/abs/2110.10545
"""

import math

import tensorflow as tf

from stable_transfer.transferability import transfer_experiment


def truncated_svd(features):
  """Calculate truncated svd."""
  s, u, v = tf.linalg.svd(tf.matmul(features, features, transpose_a=True))
  s = tf.sqrt(s)
  u_times_sigma = tf.matmul(features, v)
  rank_indices = tf.where(tf.greater(s, 1e-10))
  s = tf.gather_nd(s, rank_indices)
  u = tf.squeeze(tf.gather(u_times_sigma, rank_indices, axis=-1)) / s
  return s, u


def get_per_class_evidence(d, n, sigma, alpha, beta, res2, m2):
  """Compute the evidence L(alpha, beta) as in the paper (eq. 2)."""
  evidence = n * tf.math.log(beta)
  evidence += d * tf.math.log(alpha)
  evidence -= n * tf.math.log(2 * math.pi)
  evidence -= beta * res2
  evidence -= alpha * m2
  evidence -= tf.reduce_sum(tf.math.log(alpha + beta * sigma))
  return 0.5 * evidence


def get_optimized_per_class_params(n, sigma, u, y_c, tol=1e-2, max_iter=11):
  """Compute alpha and beta as in https://arxiv.org/abs/2110.10545 (Alg. 3)."""
  x = tf.matmul(u, y_c, transpose_a=True)
  x2 = tf.squeeze(x ** 2)
  res_x2 = tf.reduce_sum(y_c ** 2) - tf.reduce_sum(x2)
  alpha, beta = 1, 1
  for _ in range(max_iter):
    t = alpha / beta
    gamma = tf.reduce_sum(sigma / (sigma + t))
    m2 = tf.reduce_sum((sigma * x2 / ((t + sigma) ** 2)))
    res2 = tf.reduce_sum(x2 / ((1 + sigma / t) ** 2)) + res_x2
    alpha = gamma / (m2 + 1e-5)
    beta = (n - gamma) / (res2 + 1e-5)
    if tf.abs((alpha / beta) - t) / t <= tol:
      break
  return alpha, beta, res2, m2


def get_logme_score(features, target_labels):
  """Return the LogME score for classification.

  Args:
    features: matrix [N, D] of source features obtained from the target data,
      where N is the number of datapoints and D their dimensionionality.
    target_labels: ground truth target labels of dimension [N, 1].

  Returns:
    logme: transferability metric score.
  """

  d = features.shape[1]
  n = features.shape[0]
  if d > n:
    s, u, = truncated_svd(features)
  else:
    s, u, _ = tf.linalg.svd(features)
  sigma = (s ** 2)
  evidences = []

  unique_labels, _ = tf.unique(target_labels)
  num_target_classes = tf.reduce_max(target_labels) + 1
  if num_target_classes != unique_labels.shape[0]:
    raise ValueError('Labels need to be in the range [0, num_target_classes).')
  one_hot_targets = tf.one_hot(target_labels, depth=num_target_classes)

  for label in list(unique_labels):
    one_hot_label = tf.one_hot(label, depth=num_target_classes)
    y_c = tf.matmul(one_hot_targets, tf.expand_dims(one_hot_label, axis=-1))
    alpha, beta, res2, m2 = get_optimized_per_class_params(n, sigma, u, y_c)
    evidences.append(
        get_per_class_evidence(d, n, sigma, alpha, beta, res2, m2) / n)

  logme = tf.reduce_mean(evidences)
  return logme


@transfer_experiment.load_or_compute
def get_train_logme(experiment):
  """Compute LogME on the target training data."""
  features, labels = experiment.model_output_on_target_train_dataset('features')
  logme = get_logme_score(features, labels)
  return dict(logme=float(logme))
