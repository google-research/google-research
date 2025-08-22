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

"""H-score transferability metric implementation.

Bao, Yajie, et al. "An information-theoretic metric of transferability for task
transfer learning.". https://openreview.net/pdf?id=BkxAUjRqY7.

We follow the official paper implementation in
https://github.com/YaojieBao/An-Information-theoretic-Metric-of-Transferability:

Input: features x, target labels y

CovX = getCov(x)
g = zeros_like(x)

for y_i in set(y):
  m_i = mean(x[y==y_i, :], axis=0)
  g[y==y_i] = m_i

CovY = getCov(g)
hscore = np.trace(np.dot(np.linalg.pinv(CovX, rcond=1e-15), CovY))

"""

import tensorflow as tf

from stable_transfer.transferability import transfer_experiment


def get_covariance(matrix):
  """Compute the covariance of a given matrix."""
  zero_mean_matrix = matrix - tf.reduce_mean(matrix, axis=0, keepdims=True)
  cov = tf.matmul(zero_mean_matrix, zero_mean_matrix, transpose_a=True) / (
      matrix.shape[0] - 1)
  return cov


def get_hscore(features, target_labels):
  """Compute H score metric based on Bao et al. 2019.

  Args:
    features: source features from the target data.
    target_labels: ground truth labels in the target label space.

  Returns:
    hscore: transferability metric score.
  """

  covariance_features = get_covariance(features)
  inter_class_features = tf.zeros_like(features)
  unique_labels, _ = tf.unique(target_labels)

  for label in list(unique_labels):
    label_indices = tf.where(tf.equal(target_labels, label))
    label_mean = tf.reduce_mean(
        tf.gather_nd(features, label_indices), axis=0, keepdims=True)
    label_mean_repeated = tf.repeat(label_mean, label_indices.shape[0], axis=0)
    inter_class_features = tf.tensor_scatter_nd_update(
        inter_class_features, label_indices, label_mean_repeated)

  inter_class_covariance = get_covariance(inter_class_features)
  hscore = tf.linalg.trace(
      tf.linalg.pinv(covariance_features, rcond=1e-5) * inter_class_covariance)

  return hscore


@transfer_experiment.load_or_compute
def get_train_hscore(experiment):
  """Compute H score on the target training data."""
  features, labels = experiment.model_output_on_target_train_dataset('features')
  hscore = get_hscore(features, labels)
  return dict(hscore=float(hscore))
