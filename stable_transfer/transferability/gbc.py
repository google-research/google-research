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

"""Implementation of Gaussian Bhattacharyya Coefficient (GBC).

Pándy, Michal, et al. "Transferability Estimation using Bhattacharyya Class
Separability." https://arxiv.org/abs/2111.12780.
"""

import tensorflow as tf

from stable_transfer.transferability import transfer_experiment


def compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
  """Compute Bhattacharyya distance between diagonal or spherical Gaussians."""
  avg_sigma = (sigma1 + sigma2) / 2
  first_part = tf.reduce_sum((mu1 - mu2)**2 / avg_sigma) / 8
  second_part = tf.reduce_sum(tf.math.log(avg_sigma))
  second_part -= 0.5 * (tf.reduce_sum(tf.math.log(sigma1)))
  second_part -= 0.5 * (tf.reduce_sum(tf.math.log(sigma2)))
  return first_part + 0.5 * second_part


def get_bhattacharyya_distance(per_class_stats, c1, c2, gaussian_type):
  """Return Bhattacharyya distance between 2 diagonal or spherical gaussians."""
  mu1 = per_class_stats[c1]['mean']
  mu2 = per_class_stats[c2]['mean']
  sigma1 = per_class_stats[c1]['variance']
  sigma2 = per_class_stats[c2]['variance']
  if gaussian_type == 'spherical':
    sigma1 = tf.reduce_mean(sigma1)
    sigma2 = tf.reduce_mean(sigma2)
  return compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2)


def compute_per_class_mean_and_variance(features, target_labels, unique_labels):
  """Compute features mean and variance for each class."""
  per_class_stats = {}
  for label in unique_labels:
    label = int(label)  # For correct indexing
    per_class_stats[label] = {}
    class_ids = tf.equal(target_labels, label)
    class_features = tf.gather_nd(features, tf.where(class_ids))
    mean = tf.reduce_mean(class_features, axis=0)
    variance = tf.math.reduce_variance(class_features, axis=0)
    per_class_stats[label]['mean'] = mean
    # Avoid 0 variance in cases of constant features with tf.maximum
    per_class_stats[label]['variance'] = tf.maximum(variance, 1e-4)
  return per_class_stats


def get_gbc_score(features, target_labels, gaussian_type):
  """Compute Gaussian Bhattacharyya Coefficient (GBC).

  Args:
    features: source features from the target data.
    target_labels: ground truth labels in the target label space.
    gaussian_type: type of gaussian used to represent class features. The
      possibilities are spherical (default) or diagonal.

  Returns:
    gbc: transferability metric score.
  """

  assert gaussian_type in ('diagonal', 'spherical')
  unique_labels, _ = tf.unique(target_labels)
  unique_labels = list(unique_labels)
  per_class_stats = compute_per_class_mean_and_variance(
      features, target_labels, unique_labels)

  per_class_bhattacharyya_distance = []
  for c1 in unique_labels:
    temp_metric = []
    for c2 in unique_labels:
      if c1 != c2:
        bhattacharyya_distance = get_bhattacharyya_distance(
            per_class_stats, int(c1), int(c2), gaussian_type)
        temp_metric.append(tf.exp(-bhattacharyya_distance))
    per_class_bhattacharyya_distance.append(tf.reduce_sum(temp_metric))
  gbc = -tf.reduce_sum(per_class_bhattacharyya_distance)

  return gbc


@transfer_experiment.load_or_compute
def get_train_gbc(experiment):
  """Compute GBC on the target training data."""
  features, labels = experiment.model_output_on_target_train_dataset('features')
  gbc = get_gbc_score(
      features, labels, experiment.config.experiment.gbc.gaussian_type)
  return dict(gbc=float(gbc))
