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

# Lint as: python3
"""Build a deep GAM model graph."""

import functools
from typing import Union, List, Optional, Tuple, Callable, Dict
import numpy as np
from sklearn import metrics as sk_metrics
import tensorflow.compat.v1 as tf

from neural_additive_models import models

# To suppress warnings in the sigmoid function
np.warnings.filterwarnings('ignore')
TfInput = models.TfInput
LossFunction = Callable[[tf.keras.Model, TfInput, TfInput], tf.Tensor]
GraphOpsAndTensors = Dict[str, Union[tf.Tensor, tf.Operation, tf.keras.Model]]
EvaluationMetric = Callable[[tf.Session], float]


def cross_entropy_loss(model, inputs,
                       targets):
  """Cross entropy loss for binary classification.

  Args:
    model: Neural network model (NAM/DNN).
    inputs: Input values to be fed into the model for computing predictions.
    targets: Binary class labels.

  Returns:
    Cross-entropy loss between model predictions and the targets.
  """
  predictions = model(inputs, training=True)
  logits = tf.stack([predictions, tf.zeros_like(predictions)], axis=1)
  labels = tf.stack([targets, 1 - targets], axis=1)
  loss_vals = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss_vals)


def penalized_loss(loss_func,
                   model,
                   inputs,
                   targets,
                   output_regularization,
                   l2_regularization = 0.0,
                   use_dnn = False):
  """Computes penalized loss with L2 regularization and output penalty.

  Args:
    loss_func: Loss function.
    model: Neural network model.
    inputs: Input values to be fed into the model for computing predictions.
    targets: Target values containing either real values or binary labels.
    output_regularization: Coefficient for feature output penalty.
    l2_regularization: Coefficient for L2 regularization.
    use_dnn: Whether using DNN or not when computing L2 regularization.

  Returns:
    The penalized loss.
  """
  loss = loss_func(model, inputs, targets)
  reg_loss = 0.0
  if output_regularization > 0:
    reg_loss += output_regularization * feature_output_regularization(
        model, inputs)
  if l2_regularization > 0:
    num_networks = 1 if use_dnn else len(model.feature_nns)
    reg_loss += l2_regularization * weight_decay(
        model, num_networks=num_networks)
  return loss + reg_loss


def penalized_cross_entropy_loss(model,
                                 inputs,
                                 targets,
                                 output_regularization,
                                 l2_regularization = 0.0,
                                 use_dnn = False):
  """Cross entropy loss with L2 regularization and output penalty."""
  return penalized_loss(cross_entropy_loss, model, inputs, targets,
                        output_regularization, l2_regularization, use_dnn)


def penalized_mse_loss(model,
                       inputs,
                       targets,
                       output_regularization,
                       l2_regularization = 0.0,
                       use_dnn = False):
  """Mean Squared Error with L2 regularization and output penalty."""
  return penalized_loss(mse_loss, model, inputs, targets, output_regularization,
                        l2_regularization, use_dnn)


def feature_output_regularization(model,
                                  inputs):
  """Penalizes the L2 norm of the prediction of each feature net."""
  per_feature_outputs = model.calc_outputs(inputs, training=False)
  per_feature_norm = [  # L2 Regularization
      tf.reduce_mean(tf.square(outputs)) for outputs in per_feature_outputs
  ]
  return tf.add_n(per_feature_norm) / len(per_feature_norm)


def weight_decay(model, num_networks = 1):
  """Penalizes the L2 norm of weights in each feature net."""
  l2_losses = [tf.nn.l2_loss(x) for x in model.trainable_variables]
  return tf.add_n(l2_losses) / num_networks


def mse_loss(model, inputs,
             targets):
  """Mean squared error loss for regression."""
  predicted = model(inputs, training=True)
  return tf.losses.mean_squared_error(predicted, targets)


def accuracy(model, inputs,
             targets):
  """Accuracy for a binary classification model."""
  pred = model(inputs, training=False)
  binary_pred = tf.cast(pred > 0, dtype=tf.int32)
  correct = tf.equal(binary_pred, tf.cast(targets > 0.5, dtype=tf.int32))
  return tf.reduce_mean(tf.cast(correct, tf.float32))


def generate_predictions(pred_tensor, dataset_init_op,
                         sess):
  """Iterates over the `pred_tensor` to compute predictions.

  Args:
    pred_tensor: Nested structure representing the next prediction element
      obtained from the `get_next` call on a `tf.compat.v1.data.Iterator`.
    dataset_init_op: Dataset iterator initializer for `pred_tensor`.
    sess: Tensorflow session.

  Returns:
    Predictions obtained over the dataset iterated using `pred_tensor`.
  """
  sess.run(dataset_init_op)
  y_pred = []
  while True:
    try:
      y_pred.extend(sess.run(pred_tensor))
    except tf.errors.OutOfRangeError:
      break
  return y_pred


def sigmoid(x):
  """Sigmoid function."""
  if isinstance(x, list):
    x = np.array(x)
  return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def calculate_metric(y_true,
                     predictions,
                     regression = True):
  """Calculates the evaluation metric."""
  if regression:
    return rmse(y_true, predictions)
  else:
    return sk_metrics.roc_auc_score(y_true, sigmoid(predictions))


def roc_auc_score(sess, y_true, pred_tensor,
                  dataset_init_op):
  """Calculates the ROC AUC score."""
  # Assumes that pred_tensor already applies the sigmoid transformation
  y_pred = generate_predictions(pred_tensor, dataset_init_op, sess)
  return sk_metrics.roc_auc_score(y_true, y_pred)


def rmse_loss(sess, y_true, pred_tensor,
              dataset_init_op):
  """Calculates the RMSE error."""
  y_pred = generate_predictions(pred_tensor, dataset_init_op, sess)
  return rmse(y_true, y_pred)


def rmse(y_true, y_pred):
  """Root mean squared error between true and predicted values."""
  return float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)))


def grad(
    model,
    inputs,
    targets,
    loss_fn = cross_entropy_loss,
    train_vars = None
):
  """Calculates gradient w.r.t. `train_vars` of the `loss_fn` for `model`."""
  loss_value = loss_fn(model, inputs, targets)
  if train_vars is None:
    train_vars = model.trainable_variables
  return loss_value, tf.gradients(loss_value, train_vars)


def create_balanced_dataset(x_train, y_train,
                            batch_size):
  """Creates a balanced training dataset by upsampling the rare class.

  Args:
    x_train: Training data with input features.
    y_train: Binary class labels.
    batch_size: Batch size for sampling during training.

  Returns:
    A dataset from which (x,y) pairs of size `batch_size` can be
    sampled containing equal proportion of the two classes.
  """

  def partition_dataset(
      x_train, y_train
  ):
    neg_mask = (y_train == 0)
    x_train_neg = x_train[neg_mask]
    y_train_neg = np.zeros(len(x_train_neg), dtype=np.float32)
    x_train_pos = x_train[~neg_mask]
    y_train_pos = np.ones(len(x_train_pos), dtype=np.float32)
    return (x_train_pos, y_train_pos), (x_train_neg, y_train_neg)

  pos, neg = partition_dataset(x_train, y_train)
  pos_dataset = tf.data.Dataset.from_tensor_slices(pos).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(pos[0])))
  neg_dataset = tf.data.Dataset.from_tensor_slices(neg).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(neg[0])))
  dataset = tf.data.experimental.sample_from_datasets(
      [pos_dataset, neg_dataset])
  ds_tensors = dataset.batch(batch_size)
  return ds_tensors


def create_iterators(
    datasets,
    batch_size):
  """Create tf.Dataset iterators from a tuple of one or more numpy arrays.

  Args:
    datasets: Single or pair of input numpy arrays containing  features.
    batch_size: Batch size for iterating over the datasets.

  Returns:
    Sampling tensor and Initializable iterator(s) for the input datasets.
  """
  tf_datasets = [
      tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
      for data in datasets
  ]
  input_iterator = tf.data.Iterator.from_structure(tf_datasets[0].output_types,
                                                   tf_datasets[0].output_shapes)
  init_ops = [input_iterator.make_initializer(data) for data in tf_datasets]
  x_batch = input_iterator.get_next()
  return x_batch, init_ops


def create_nam_model(x_train,
                     dropout,
                     feature_dropout = 0.0,
                     num_basis_functions = 1000,
                     units_multiplier = 2,
                     activation = 'exu',
                     name_scope = 'model',
                     shallow = True,
                     trainable = True):
  """Create the NAM model."""
  num_unique_vals = [
      len(np.unique(x_train[:, i])) for i in range(x_train.shape[1])
  ]
  num_units = [
      min(num_basis_functions, i * units_multiplier) for i in num_unique_vals
  ]
  num_inputs = x_train.shape[-1]
  nn_model = models.NAM(
      num_inputs=num_inputs,
      num_units=num_units,
      dropout=np.float32(dropout),
      feature_dropout=np.float32(feature_dropout),
      activation=activation,
      shallow=shallow,
      trainable=trainable,
      name_scope=name_scope)
  return nn_model


def build_graph(
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rate,
    batch_size,
    output_regularization,
    dropout,
    decay_rate,
    shallow,
    l2_regularization = 0.0,
    feature_dropout = 0.0,
    num_basis_functions = 1000,
    units_multiplier = 2,
    activation = 'exu',
    name_scope = 'model',
    regression = False,
    use_dnn = False,
    trainable = True
):
  """Constructs the computation graph with specified hyperparameters."""
  if regression:
    ds_tensors = tf.data.Dataset.from_tensor_slices((x_train, y_train)).apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=len(x_train[0])))
    ds_tensors = ds_tensors.batch(batch_size)
  else:
    # Create a balanced dataset to handle class imbalance
    ds_tensors = create_balanced_dataset(x_train, y_train, batch_size)
  x_batch, (train_init_op, test_init_op) = create_iterators((x_train, x_test),
                                                            batch_size)

  if use_dnn:
    nn_model = models.DNN(dropout=dropout, trainable=trainable)
  else:
    nn_model = create_nam_model(
        x_train=x_train,
        dropout=dropout,
        feature_dropout=feature_dropout,
        activation=activation,
        num_basis_functions=num_basis_functions,
        shallow=shallow,
        units_multiplier=units_multiplier,
        trainable=trainable,
        name_scope=name_scope)

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.Variable(learning_rate, trainable=False)
  lr_decay_op = learning_rate.assign(decay_rate * learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)

  predictions = nn_model(x_batch, training=False)
  tf.logging.info(nn_model.summary())
  train_vars = nn_model.trainable_variables
  if regression:
    loss_fn, y_pred = penalized_mse_loss, predictions
  else:
    # Apply sigmoid transformation for binary classification
    loss_fn, y_pred = penalized_cross_entropy_loss, tf.nn.sigmoid(predictions)
  loss_fn = functools.partial(
      loss_fn,
      output_regularization=output_regularization,
      l2_regularization=l2_regularization,
      use_dnn=use_dnn)

  iterator = ds_tensors.make_initializable_iterator()
  x1, y1 = iterator.get_next()
  loss_tensor, grads = grad(nn_model, x1, y1, loss_fn, train_vars)
  update_step = optimizer.apply_gradients(
      zip(grads, train_vars), global_step=global_step)
  avg_loss, avg_loss_update_op = tf.metrics.mean(
      loss_tensor, name='avg_train_loss')
  tf.summary.scalar('avg_train_loss', avg_loss)

  running_mean_vars = tf.get_collection(
      tf.GraphKeys.LOCAL_VARIABLES, scope='avg_train_loss')
  running_vars_initializer = tf.variables_initializer(
      var_list=running_mean_vars)

  # Use RMSE for regression and ROC AUC for classification.
  evaluation_metric = rmse_loss if regression else roc_auc_score
  train_metric = functools.partial(
      evaluation_metric,
      y_true=y_train,
      pred_tensor=y_pred,
      dataset_init_op=train_init_op)
  test_metric = functools.partial(
      evaluation_metric,
      y_true=y_test,
      pred_tensor=y_pred,
      dataset_init_op=test_init_op)

  summary_op = tf.summary.merge_all()

  graph_tensors = {
      'train_op': [update_step, avg_loss_update_op],
      'lr_decay_op': lr_decay_op,
      'summary_op': summary_op,
      'iterator_initializer': iterator.initializer,
      'running_vars_initializer': running_vars_initializer,
      'nn_model': nn_model,
      'global_step': global_step,
  }
  eval_metric_scores = {'test': test_metric, 'train': train_metric}
  return graph_tensors, eval_metric_scores
