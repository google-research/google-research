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

"""Baseline mechanisms for differentially private linear regression.

This file implements the non-private and (epsilon, delta)-DP AdaSSP linear
regression algorithms.
"""

import numpy as np
import sklearn.linear_model
import tensorflow.compat.v2 as tf

from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras


def nondp(features, labels):
  """Returns model computed using standard non-DP linear regression.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature in
      the last column.
    labels: Vector of labels.

  Returns:
    Vector of regression coefficients.
  """
  _, d = features.shape
  model = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(
      features, labels)
  model_v = np.zeros((d, 1))
  model_v[:, 0] = model.coef_
  return model_v


def adassp(features, labels, epsilon, delta, rho=0.05):
  """Returns model computed using AdaSSP DP linear regression.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Computed model satisfies (epsilon, delta)-DP.
    delta: Computed model satisfies (epsilon, delta)-DP.
    rho: Failure probability. The default of 0.05 is the one used in
      https://arxiv.org/pdf/1803.02596.pdf.

  Returns:
    Vector of regression coefficients. AdaSSP is described in Algorithm 2 of
    https://arxiv.org/pdf/1803.02596.pdf.
  """
  _, d = features.shape
  # these bounds are data-dependent and not dp
  bound_x = np.amax(np.linalg.norm(features, axis=1))
  bound_y = np.amax(np.abs(labels))
  lambda_min = max(0,
                   np.amin(np.linalg.eigvals(np.matmul(features.T, features))))
  z = np.random.normal(size=1)
  sensitivity = np.sqrt(np.log(6 / delta)) / (epsilon / 3)
  private_lambda = max(
      0, lambda_min + sensitivity * (bound_x**2) * z -
      (bound_x**2) * np.log(6 / delta) / (epsilon / 3))
  final_lambda = max(
      0,
      np.sqrt(d * np.log(6 / delta) * np.log(2 * (d**2) / rho)) * (bound_x**2) /
      (epsilon / 3) - private_lambda)
  # generate symmetric noise_matrix where each upper entry is iid N(0,1)
  noise_matrix = np.random.normal(size=(d, d))
  noise_matrix = np.triu(noise_matrix)
  noise_matrix = noise_matrix + noise_matrix.T - np.diag(np.diag(noise_matrix))
  priv_xx = np.matmul(features.T,
                      features) + sensitivity * (bound_x**2) * noise_matrix
  priv_xy = np.dot(features.T, labels).flatten(
  ) + sensitivity * bound_x * bound_y * np.random.normal(size=d)
  model_adassp = np.matmul(
      np.linalg.pinv(priv_xx + final_lambda * np.eye(d)), priv_xy)
  return model_adassp


def dpsgd(features, labels, params):
  """Returns linear regression model computed by DPSGD with given params.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    params: Dictionary of parameters (num_epochs, clip_norm, learning_rate,
      noise_multiplier) used by DPSGD. To verify that the noise multiplier
      provides the desired DP guarantee given the remaining hyperparameters, use
      compute_dp_sgd_privacy.
  """
  n, _ = features.shape
  order = np.arange(n)
  np.random.shuffle(order)

  max_samples = int((n//params["batch_size"])*params["batch_size"])
  features_processed = features[order][:max_samples]
  labels_processed = labels[order][:max_samples]

  model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, use_bias=False))
  model.compile(
      optimizer=dp_optimizer_keras.DPKerasAdamOptimizer(
          l2_norm_clip=params["clip_norm"],
          learning_rate=params["learning_rate"],
          noise_multiplier=params["noise_multiplier"],
          num_microbatches=params["batch_size"]),
      loss=tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE))

  model.fit(features_processed, labels_processed, epochs=params["num_epochs"],
            batch_size=params["batch_size"], verbose=0)
  return np.squeeze(model.layers[0].get_weights()[0])
