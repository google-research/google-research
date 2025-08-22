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
"""Helper functions for quantile regression training and evaluation."""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
import math

import sys

from absl import app
from absl import flags


def extract_features(dataframe,
                     label_name,
                     feature_names,
                     normalize_features=False):
  """Extracts features and label from the dataframe into np arrays.

  If normalize_features is True, returns normalized features.

  """
  features = []
  for feature_name in feature_names:
    features.append(dataframe[feature_name].values.astype(float))
  if normalize_features:
    features = [
        np.divide(
            arr - np.mean(arr),
            np.std(arr),
            out=np.zeros_like(arr),
            where=np.std(arr) != 0) for arr in features
    ]
  labels = dataframe[label_name].values.astype(float)
  return features, labels


def print_metric(dataset_name, metric_name, metric_value):
  """Prints metrics."""
  print("[metric] %s.%s=%f" % (dataset_name, metric_name, metric_value))


# This function is common to both linear and lattice model setups.
def compute_quantiles(features,
                      num_keypoints=10,
                      clip_min=None,
                      clip_max=None,
                      missing_value=None):
  """Computes quantiles for features to feed to TFL configs."""
  # Clip min and max if desired.
  if clip_min is not None:
    features = np.maximum(features, clip_min)
    features = np.append(features, clip_min)
  if clip_max is not None:
    features = np.minimum(features, clip_max)
    features = np.append(features, clip_max)
  # Make features unique.
  unique_features = np.unique(features)
  # Remove missing values if specified.
  if missing_value is not None:
    unique_features = np.delete(unique_features,
                                np.where(unique_features == missing_value))
  # Compute and return quantiles over unique non-missing feature values.
  if len(unique_features) >= num_keypoints:
    return np.quantile(
        unique_features,
        np.linspace(0., 1., num=num_keypoints),
        interpolation="nearest").astype(float)
  else:
    # Ensure no duplicates when num_keypoints > number of unique feature values.
    return unique_features


def get_marginal_rate(pred_tensor, y_array):
  r"""Marginal rate is 1/n \sum_{i=1}^n I(y_i \leq f(x_i, q))."""
  pred_array = pred_tensor.numpy().flatten()
  return np.sum(((pred_array - y_array) > 0).astype(float)) / len(y_array)


def calculate_q_loss(input_df,
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
  xs, ys = extract_features(
      input_df,
      feature_names=feature_names,
      label_name=label_name,
      normalize_features=normalize_features)
  q_losses = []
  if q_values is None:
    # append random q's
    q = np.random.uniform(0.0, 1.0, size=len(input_df))
    y_pred_tensor = model_fun(xs + [q])
    y_pred_array = y_pred_tensor.numpy().flatten()
    diff = ys - y_pred_array
    q_loss = np.mean(
        np.maximum(diff, 0.0) * q + np.minimum(diff, 0.0) * (q - 1.0))
    q_losses.append(q_loss)
  else:
    for q_value in q_values:
      q = np.full(xs[-1].shape, q_value)
      y_pred_tensor = model_fun(xs + [q])
      y_pred_array = y_pred_tensor.numpy().flatten()
      diff = ys - y_pred_array
      q_loss = np.mean(
          np.maximum(diff, 0.0) * q + np.minimum(diff, 0.0) * (q - 1.0))
      q_losses.append(q_loss)
  return np.mean(np.array(q_losses))


def get_rate_constraint_viols(xs, ys, model_fun, desired_rates=[]):
  """Gets constraint violations for desired quantiles between start and stop."""
  rate_constraint_viols = []
  for desired_rate in desired_rates:
    q_array = np.full(xs[-1].shape, desired_rate)
    pred = model_fun(xs + [q_array])
    marginal_rate = get_marginal_rate(pred, ys)
    rate_constraint_viol = marginal_rate - desired_rate
    rate_constraint_viols.append(rate_constraint_viol)
  return desired_rates, rate_constraint_viols


def get_avg_calibration_viols(input_df,
                              model_fun,
                              feature_names,
                              label_name,
                              q_values=None,
                              normalize_features=False):
  """Gets average calibration violation of model over desired quantiles."""
  xs, ys = extract_features(
      input_df,
      feature_names=feature_names,
      label_name=label_name,
      normalize_features=normalize_features)
  _, viols = get_rate_constraint_viols(xs, ys, model_fun, q_values)
  return np.mean(np.abs(viols)), np.mean(np.square(viols))


def build_dnn_model(num_features, num_layers=4, hidden_dim=4):
  """DNN model setup.

  Args:
    num_features: number of input features, not including q.
    num_layers: number of layers in the dnn.
    hidden_dim: dimension of each hidden layer.
  """
  input_dim = num_features + 1
  y_layers = [tf.keras.layers.Input(shape=(1,)) for _ in range(input_dim)]
  # z1 layer. units is the hidden dimension: z1 = function()(input to that function)
  merged_layer = tf.keras.layers.concatenate(y_layers)
  print(merged_layer)
  units = 1 if num_layers == 1 else hidden_dim
  activation = None if num_layers == 1 else "relu"
  z_layer = tf.keras.layers.Dense(
      units=units,
      activation=activation,
  )(
      merged_layer)

  for i in range(num_layers - 1):
    # If last layer, the output is 1d, the last layer is i = NUM_DNN_LAYERS-2
    units = 1 if i == num_layers - 2 else hidden_dim
    new_z_layer = tf.keras.layers.Dense(
        units=units,
        activation=None,
    )(
        z_layer)
    z_layer = new_z_layer
    # If not last layer, apply ReLU activation
    if i < num_layers - 2:
      z_layer = tf.keras.layers.Activation("relu")(z_layer)

  keras_model = tf.keras.models.Model(inputs=y_layers, outputs=z_layer)
  return keras_model


def train_pinball_keras(input_model,
                        train_xs,
                        train_ys,
                        q,
                        model_step_size=0.1,
                        epochs=40,
                        batch_size=2000):
  """Trains a model using the pinball loss with Keras."""
  model_inputs = input_model.inputs
  model_output = input_model.output
  stacked_output = tf.keras.layers.Lambda(lambda inputs: tf.stack(inputs, 1))(
      [model_output, model_inputs[-1]])
  model = tf.keras.Model(inputs=model_inputs, outputs=stacked_output)

  def q_loss(y_true, stacked_output):
    y_pred, q = tf.split(stacked_output, 2, axis=1)
    diff = y_true - y_pred
    # Using reduce_mean here so that the loss see on batches during training
    # matches the final calculated loss on the full dataset.
    return tf.reduce_mean(
        tf.maximum(diff, 0.0) * q + tf.minimum(diff, 0.0) * (q - 1.0))

  model.compile(
      loss=q_loss, optimizer=tf.keras.optimizers.Adam(model_step_size))
  model.fit(
      train_xs + [q], train_ys, epochs=epochs, batch_size=batch_size, verbose=2)
  return input_model
