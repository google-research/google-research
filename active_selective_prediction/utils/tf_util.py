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

"""Tensorflow utils."""

from typing import List, Tuple
import tensorflow as tf


@tf.function
def get_model_feature(
    model,
    batch_x
):
  """Gets model's features on the given inputs."""
  features = model.get_feature(batch_x, training=False)
  return features


@tf.function
def get_model_output(
    model,
    batch_x
):
  """Gets model's outputs on the given inputs."""
  outputs = model(batch_x, training=False)
  return outputs


@tf.function
def get_model_output_and_feature(
    model,
    batch_x
):
  """Gets model's outputs and features on the given inputs."""
  outputs, features = model.get_output_and_feature(batch_x, training=False)
  return outputs, features


@tf.function
def get_model_prediction(
    model,
    batch_x
):
  """Gets model's predictions on the given inputs."""
  outputs = model(batch_x, training=False)
  preds = tf.argmax(outputs, axis=1)
  return preds


@tf.function
def get_model_confidence(
    model,
    batch_x
):
  """Gets model's confidences on the given inputs."""
  outputs = model(batch_x, training=False)
  confs = tf.math.reduce_max(outputs, axis=1)
  return confs


@tf.function
def get_model_margin(
    model,
    batch_x
):
  """Gets model's margins on the given inputs."""
  outputs = model(batch_x, training=False)
  sorted_outputs = tf.sort(outputs, direction='DESCENDING', axis=1)
  margins = sorted_outputs[:, 0] - sorted_outputs[:, 1]
  return margins


@tf.function
def get_ensemble_model_output(
    models,
    batch_x,
    ensemble_method
):
  """Gets ensemble model's outputs on the given inputs."""
  batch_ensemble_output = 0
  if ensemble_method == 'hard':
    num_classes = None
  for model in models:
    batch_output = model(batch_x, training=False)
    if ensemble_method == 'hard':
      batch_pred = tf.argmax(batch_output, axis=1)
      if num_classes is None:
        num_classes = batch_output.shape[1]
      batch_one_hot_output = tf.one_hot(batch_pred, num_classes)
      batch_ensemble_output += batch_one_hot_output
    elif ensemble_method == 'soft':
      batch_ensemble_output += batch_output
    else:
      raise ValueError(f'Not supported ensemble method: {ensemble_method}!')
  return batch_ensemble_output / len(models)


@tf.function
def get_ensemble_model_feature(
    models,
    batch_x
):
  """Gets ensemble model's features on the given inputs."""
  batch_feature_list = []
  for model in models:
    batch_feature = model.get_feature(batch_x, training=False)
    batch_feature_list.append(batch_feature)
  # Concatenates the features of the models in the ensemble.
  concat_batch_feature = tf.concat(batch_feature_list, axis=1)
  return concat_batch_feature


@tf.function
def get_ensemble_model_output_and_feature(
    models,
    batch_x,
    ensemble_method,
    temperature = 1.0,
):
  """Gets ensemble model's outputs and features on the given inputs."""
  batch_ensemble_output = 0
  batch_feature_list = []
  if ensemble_method == 'hard':
    num_classes = None
  for model in models:
    batch_output, batch_feature = model.get_output_and_feature(
        batch_x, training=False, temperature=temperature,
    )
    batch_feature_list.append(batch_feature)
    if ensemble_method == 'hard':
      batch_pred = tf.argmax(batch_output, axis=1)
      if num_classes is None:
        num_classes = batch_output.shape[1]
      batch_one_hot_output = tf.one_hot(batch_pred, num_classes)
      batch_ensemble_output += batch_one_hot_output
    elif ensemble_method == 'soft':
      batch_ensemble_output += batch_output
    else:
      raise ValueError(f'Not supported ensemble method: {ensemble_method}!')
  # Concatenates the features of the models in the ensemble.
  concat_batch_feature = tf.concat(batch_feature_list, axis=1)
  return batch_ensemble_output / len(models), concat_batch_feature


@tf.function
def get_ensemble_model_prediction(
    models,
    batch_x,
    ensemble_method,
):
  """Gets ensemble model's predictions on the given inputs.

  Args:
    models: a list of models
    batch_x: a batch of inputs
    ensemble_method: the method to construct ensemble

  Returns:
    The ensemble model's predictions
  """
  batch_ensemble_output = 0
  if ensemble_method == 'hard':
    num_classes = None
  for model in models:
    batch_output = model(batch_x, training=False)
    if ensemble_method == 'hard':
      batch_pred = tf.argmax(batch_output, axis=1)
      if num_classes is None:
        num_classes = batch_output.shape[1]
      batch_one_hot_output = tf.one_hot(batch_pred, num_classes)
      batch_ensemble_output += batch_one_hot_output
    elif ensemble_method == 'soft':
      batch_ensemble_output += batch_output
    else:
      raise ValueError(f'Not supported ensemble method: {ensemble_method}!')
  batch_preds = tf.argmax(batch_ensemble_output / len(models), axis=1)
  return batch_preds


@tf.function
def get_ensemble_model_confidence(
    models,
    batch_x,
    ensemble_method
):
  """Gets ensemble model's confidences on the given inputs.

  Args:
    models: a list of models
    batch_x: a batch of inputs
    ensemble_method: the method to construct ensemble

  Returns:
    The ensemble model's confidences
  """
  batch_ensemble_output = 0
  if ensemble_method == 'hard':
    num_classes = None
  for model in models:
    batch_output = model(batch_x, training=False)
    if ensemble_method == 'hard':
      batch_pred = tf.argmax(batch_output, axis=1)
      if num_classes is None:
        num_classes = batch_output.shape[1]
      batch_one_hot_output = tf.one_hot(batch_pred, num_classes)
      batch_ensemble_output += batch_one_hot_output
    elif ensemble_method == 'soft':
      batch_ensemble_output += batch_output
    else:
      raise ValueError(f'Not supported ensemble method: {ensemble_method}!')
  batch_confs = tf.math.reduce_max(batch_ensemble_output / len(models), axis=1)
  return batch_confs


@tf.function
def get_ensemble_model_margin(
    models,
    batch_x,
    ensemble_method
):
  """Gets ensemble model's margins on the given inputs.

  Args:
    models: a list of models
    batch_x: a batch of inputs
    ensemble_method: the method to construct ensemble

  Returns:
    The ensemble model's margins
  """
  batch_ensemble_output = 0
  if ensemble_method == 'hard':
    num_classes = None
  for model in models:
    batch_output = model(batch_x, training=False)
    if ensemble_method == 'hard':
      batch_pred = tf.argmax(batch_output, axis=1)
      if num_classes is None:
        num_classes = batch_output.shape[1]
      batch_one_hot_output = tf.one_hot(batch_pred, num_classes)
      batch_ensemble_output += batch_one_hot_output
    elif ensemble_method == 'soft':
      batch_ensemble_output += batch_output
    else:
      raise ValueError(f'Not supported ensemble method: {ensemble_method}!')
  batch_ensemble_output = batch_ensemble_output / len(models)
  batch_sorted_ensemble_outputs = tf.sort(
      batch_ensemble_output, direction='DESCENDING', axis=1
  )
  batch_margins = (
      batch_sorted_ensemble_outputs[:, 0] - batch_sorted_ensemble_outputs[:, 1]
  )
  return batch_margins


def evaluate_acc(
    model,
    ds
):
  """Evaluates model's accuracy on the dataset."""
  n = 0
  correct = 0
  for batch_x, batch_y in ds:
    batch_pred = get_model_prediction(model, batch_x)
    correct += tf.math.reduce_sum(
        tf.cast(batch_pred == batch_y, dtype=tf.int32)
    )
    n += batch_y.shape[0]
  return correct / n


def evaluate_ensemble_acc(
    models,
    ds
):
  """Evaluates ensemble's accuracy on the dataset."""
  n = 0
  correct = 0
  for batch_x, batch_y in ds:
    batch_pred = get_ensemble_model_prediction(
        models,
        batch_x,
        ensemble_method='soft',
    )
    correct += tf.math.reduce_sum(
        tf.cast(batch_pred == batch_y, dtype=tf.int32)
    )
    n += batch_y.shape[0]
  return correct / n


def evaluate_loss(
    model,
    ds,
    loss_func_name = 'CE'
):
  """Evaluates model's cross-entropy loss on the dataset."""
  loss = 0
  if loss_func_name == 'CE':
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM
    )
  else:
    raise ValueError(f'Not supported loss function {loss_func_name}!')
  n = 0
  for batch_x, batch_y in ds:
    batch_output = get_model_output(model, batch_x)
    loss += loss_func(batch_y, batch_output)
    n += batch_y.shape[0]
  return loss / n


def entropy_loss(
    outputs,
    epsilon = 1e-6
):
  """Computes entropy loss."""
  loss = -tf.reduce_sum(outputs*tf.math.log(outputs+epsilon), axis=1)
  return loss
