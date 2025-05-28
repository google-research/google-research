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

# pylint: skip-file
"""Helper functions for running Gasthaus et al. comparison."""

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from quantile_regression import qr_lib


def build_gasthaus_dnn_model(num_features,
                             num_hidden_layers=1,
                             hidden_dim=7,
                             keypoints_L=10):
  """DNN model setup with multiple heads.

  The heads correspond to quantile function parameters from Gasthaus et al.

  Args:
    num_features: number of input features, not including q.
    num_hidden_layers: number of hidden layers in the dnn.
    hidden_dim: dimension of each hidden layer.
    keypoints_L: number of keypoints in the parameterized quantile function (L
      in the Gasthaus paper).
  """
  input_dim = num_features
  y_layers = [tf.keras.layers.Input(shape=(1,)) for _ in range(input_dim)]
  merged_layer = tf.keras.layers.concatenate(y_layers)
  print(merged_layer)
  z_layer = tf.keras.layers.Dense(
      units=hidden_dim,
      activation="relu",
      # kernel_initializer='ones',
      bias_initializer="zeros")(
          merged_layer)

  for _ in range(num_hidden_layers - 1):
    new_z_layer = tf.keras.layers.Dense(
        units=hidden_dim,
        activation="relu",
        # kernel_initializer='ones',
        bias_initializer="zeros")(
            z_layer)
    z_layer = new_z_layer

  beta_layer = tf.keras.layers.Dense(
      units=keypoints_L,
      activation="softplus",
      # kernel_initializer='zeros',
      bias_initializer="zeros",
      name="beta")(
          z_layer)

  delta_layer = tf.keras.layers.Dense(
      units=keypoints_L,
      activation="softmax",
      # kernel_initializer='zeros',
      bias_initializer="zeros",
      name="delta")(
          z_layer)

  gamma_layer = tf.keras.layers.Dense(
      units=1,
      activation=None,
      # kernel_initializer='zeros',
      bias_initializer="zeros",
      name="gamma")(
          z_layer)

  # Concatenate outputs into a single output tensor.
  # (Note: Multiple outputs makes defining the CRPS loss hard, since each output
  # needs its own independent loss function.)
  merged_output_layer = tf.keras.layers.concatenate(
      [beta_layer, delta_layer, gamma_layer])
  print(merged_output_layer)

  keras_model = tf.keras.models.Model(
      inputs=y_layers, outputs=merged_output_layer)
  return keras_model


def convert_beta_to_b(beta):
  beta_aug = tf.concat([tf.zeros(shape=(len(beta), 1)), beta], axis=1)
  b = beta_aug[:, 1:] - beta_aug[:, :-1]
  b = tf.concat([b, tf.zeros(shape=(len(b), 1))], axis=1)
  return b


def convert_delta_to_d(delta):
  """Computes tensor d from tensor delta."""
  # Can't just use exclusive=True since we need L+1 elements.
  d = tf.cumsum(delta, axis=1)
  d = tf.concat([tf.zeros(shape=(len(d), 1)), d], axis=1)
  return d


def pwl_function_s_single(tau, b, d, gamma):
  """Computes an s tensor from a tensor of taus, and a single b, d, gamma.

  Args:
    tau: tensor of shape (n, 1)
    b: tensor of shape (L+1,)
    d: tensor of shape (L+1,)
    gamma: tensor of shape(1,)
  """
  return gamma + tf.reduce_sum(b * tf.maximum(tau - d, 0), axis=1)


def pwl_function_s_tensor(tau, b, d, gamma):
  """Computes s tensor from tau, b, d, gamma tensors.

  Args:
    tau: tensor of shape (n, 1)
    b: tensor of shape (n, L+1)
    d: tensor of shape (n, L+1)
    gamma: tensor of shape (n, 1)
  """
  sum_term = tf.reduce_sum(b * tf.maximum(tau - d, 0), axis=1)
  return gamma + tf.reshape(sum_term, shape=gamma.shape)


def get_a_tilde(input_tuple):
  """Computes a single a_tilde from a single b, d, gamma.

  Args:
    y: tensor of shape (1,).
    b: tensor of shape (L+1,).
    d: tensor of shape (L+1,).
    gamma: tensor of shape (1,).
  """
  y, b, d, gamma = input_tuple
  s = pwl_function_s_single(tf.reshape(d, shape=(-1, 1)), b, d, gamma)
  ind = tf.maximum(tf.sign(y - s), 0)
  ind = tf.reshape(ind, b.shape)
  numerator = y - gamma + tf.reduce_sum(ind * b * d)
  denominator = tf.reduce_sum(ind * b)
  return tf.math.divide_no_nan(numerator, denominator)


def compute_CRPS_tensor(y_true, model_output):
  """Computes a single CRPS value from a single b, d, gamma.

  Args:
    y_true: tensor of shape (batch size, 1).
    model_output: list of tensors [beta, delta, gamma]
      beta: tensor of shape (batch size, L).
      delta: tensor of shape (batch size, L).
      gamma: tensor of shape (batch size, 1).
  """
  keypoints_L = int((model_output.shape[1] - 1) / 2)
  beta, delta, gamma = tf.split(
      model_output, num_or_size_splits=[keypoints_L, keypoints_L, 1], axis=1)
  b = convert_beta_to_b(beta)  # b should have shape (batch size, L+1).
  d = convert_delta_to_d(delta)  # d should have shape (batch size, L+1).
  a_tilde = tf.map_fn(
      fn=get_a_tilde,
      elems=(y_true, b, d, gamma),
      fn_output_signature=tf.float32)

  def get_sum_term(input_tuple):
    a_tilde, b, d = input_tuple
    sum_inner_term_1 = (tf.ones_like(d) - tf.math.pow(d, 3)) / 3
    sum_inner_term_2 = tf.math.square(tf.maximum(d, a_tilde))
    sum_inner_term_3 = 2 * tf.maximum(d, a_tilde) * d
    sum_inner_term = sum_inner_term_1 - d - sum_inner_term_2 + sum_inner_term_3
    return tf.reduce_sum(tf.multiply(b, sum_inner_term))

  sum_term = tf.map_fn(
      fn=get_sum_term, elems=(a_tilde, b, d), fn_output_signature=tf.float32)
  sum_term = tf.reshape(sum_term, shape=(-1, 1))
  CRPS_term_1 = tf.multiply(2 * a_tilde - tf.ones_like(a_tilde), y_true)
  CRPS_term_2 = tf.multiply(tf.ones_like(a_tilde) - 2 * a_tilde, gamma)
  # print('CRPS_term_1', CRPS_term_1)
  # print('CRPS_term_2', CRPS_term_2)
  # print('sum_term', sum_term)
  CRPS_results = CRPS_term_1 + CRPS_term_2 + sum_term
  return CRPS_results


def train_gasthaus_CRPS(input_model,
                        train_xs,
                        train_ys,
                        model_step_size=0.1,
                        epochs=40,
                        batch_size=2000):
  """Trains Gasthaus model using CRPS loss."""

  def CRPS_loss(y_true, output):
    """y_true is a tensor of shape [batch size, 1].

    output is a tensor of shape [batch size, L, L, 1].

    """
    CRPS_tensor = compute_CRPS_tensor(y_true, output)
    return tf.reduce_mean(CRPS_tensor)

  input_model.compile(
      loss=CRPS_loss, optimizer=tf.keras.optimizers.Adam(model_step_size))
  input_model.fit(
      train_xs, train_ys, epochs=epochs, batch_size=batch_size, verbose=1)
  return input_model


def calculate_y_pred_tensor_gasthaus(model_fun, xs, qs):
  """Calculates the predicted inverse CDF of input qs for each of input xs.

  Args:
    model_fun: trained Gasthaus tf model
    xs: tensor of shape (num examples, num features)
    qs: tensor of shape (num examples,)

  Returns:
    y_pred_tensor: tensor of shape (num examples,) where each entry is an
      inverse CDF prediction.
  """
  model_output = model_fun(xs)
  keypoints_L = int((model_output.shape[1] - 1) / 2)
  beta, delta, gamma = tf.split(
      model_output, num_or_size_splits=[keypoints_L, keypoints_L, 1], axis=1)
  b = convert_beta_to_b(beta)  # b should have shape (batch size, L+1).
  d = convert_delta_to_d(delta)  # d should have shape (batch size, L+1).
  # Compute actual inverse CDF of q over all individual b,d,gammas
  y_pred_tensor = pwl_function_s_tensor(
      tf.reshape(qs, shape=(-1, 1)), b, d, gamma)
  return y_pred_tensor


def calculate_q_loss_gasthaus(input_df,
                              model_fun,
                              feature_names,
                              label_name,
                              q_values=None,
                              normalize_features=False):
  """Calculates the pinball loss.

  Args:
    input_df: dataframe over which to calculate q loss.
    model_fun: keras model to evaluate.
    feature_names: feature names.
    label_name: label name.
    q_values: numpy array of q values over which to calculate the pinball loss.
      The pinball loss will be calculated for each q value and averaged over the
      q values in this list. If empty, perform a uniform random sample of q
      values.
    normalize_features: whether to normalize features.
  """
  xs, ys = qr_lib.extract_features(
      input_df,
      feature_names=feature_names,
      label_name=label_name,
      normalize_features=normalize_features)
  q_losses = []
  if q_values is None:
    # append random q's
    qs = np.random.uniform(0.0, 1.0, size=len(input_df))
    y_pred_tensor = calculate_y_pred_tensor_gasthaus(
        model_fun, xs, tf.convert_to_tensor(qs, dtype=tf.float32))
    y_pred_array = y_pred_tensor.numpy().flatten()
    diff = ys - y_pred_array
    q_loss = np.mean(
        np.maximum(diff, 0.0) * qs + np.minimum(diff, 0.0) * (qs - 1.0))
    q_losses.append(q_loss)
  else:
    for q_value in q_values:
      qs = np.full(xs[-1].shape, q_value)
      y_pred_tensor = calculate_y_pred_tensor_gasthaus(
          model_fun, xs, tf.convert_to_tensor(qs, dtype=tf.float32))
      y_pred_array = y_pred_tensor.numpy().flatten()
      diff = ys - y_pred_array
      q_loss = np.mean(
          np.maximum(diff, 0.0) * qs + np.minimum(diff, 0.0) * (qs - 1.0))
      q_losses.append(q_loss)
  return np.mean(np.array(q_losses))


def calculate_calib_viol_gasthaus(input_df,
                                  model_fun,
                                  feature_names,
                                  label_name,
                                  q_values=None,
                                  normalize_features=False):
  """Calculates the average calibration violation over q_values.

  Args:
    input_df: dataframe over which to calculate q loss.
    model_fun: keras model to evaluate.
    feature_names: feature names.
    label_name: label name.
    q_values: numpy array of q values over which to calculate the pinball loss.
      The pinball loss will be calculated for each q value and averaged over the
      q values in this list. If empty, perform a uniform random sample of q
      values.
    normalize_features: whether to normalize features.
  """
  xs, ys = qr_lib.extract_features(
      input_df,
      feature_names=feature_names,
      label_name=label_name,
      normalize_features=normalize_features)
  calibration_viols = []
  for q_value in q_values:
    qs = np.full(xs[-1].shape, q_value)
    y_pred_tensor = calculate_y_pred_tensor_gasthaus(
        model_fun, xs, tf.convert_to_tensor(qs, dtype=tf.float32))
    y_pred_array = y_pred_tensor.numpy().flatten()
    marginal_rate = np.sum(((y_pred_array - ys) > 0).astype(float)) / len(ys)
    calibration_viol = marginal_rate - q_value
    calibration_viols.append(calibration_viol)
  return np.mean(np.abs(calibration_viols)), np.mean(
      np.square(calibration_viols))


def calculate_avg_CRPS_loss(input_df,
                            model_fun,
                            feature_names,
                            label_name,
                            normalize_features=False):
  """Calculates the CRPS loss.

  Args:
    input_df: dataframe over which to calculate q loss.
    model_fun: keras model to evaluate.
    feature_names: feature names.
    label_name: label name.
    normalize_features: whether to normalize features.
  """
  xs, ys = qr_lib.extract_features(
      input_df,
      feature_names=feature_names,
      label_name=label_name,
      normalize_features=normalize_features)
  y_tensor = tf.convert_to_tensor(ys, dtype=tf.float32)
  y_tensor = tf.reshape(y_tensor, shape=(-1, 1))
  model_output = model_fun(xs)
  CRPS_tensor = compute_CRPS_tensor(y_tensor, model_output)
  avg_CRPS_loss = tf.reduce_mean(CRPS_tensor)
  return avg_CRPS_loss.numpy()
