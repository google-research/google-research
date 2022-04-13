# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# pylint: disable=dangerous-default-value
"""A custom estimator to establish a simple baseline for robust learning.

Implements a fully connected feedforward network with standard ERM objective.

We implement our own baseline instead of using a canned estimator
for the following reasons:
  - Canned estimators might have model improvements (e.g., gradient clipping)
    turned-on by default. In order to ensure that we are not comparing
    apples to oranges we use exactly the same model as baseline.
  - Canned estimators expect ``label'' as a tensor. But, our data input_fn
    return labels as a dictionary of tensors, including subgroup information.
  - We use the protected group information in the labels dictionary to compute
      additional fairness eval_metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics


class _BaselineModel():
  """TensorFlow BaselineModel base class.

  BaselineModel class can be used to instantiate a feedforward DNN
  classifier with standard ERM objective.
  """

  def __init__(
      self,
      feature_columns,
      label_column_name,
      config,
      model_dir,
      hidden_units=[64, 32],
      batch_size=256,
      learning_rate=0.01,
      optimizer='Adagrad',
      activation=tf.nn.relu
      ):
    """Initializes a baseline estimator.

    Args:
      feature_columns: list of feature_columns.
      label_column_name: (string) name of the target variable.
      config: `RunConfig` object to configure the runtime settings.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator
        to continue training a previously saved model.
      hidden_units: List with number of hidden units per layer for the
        shared bottom.  All layers are fully connected.
        Ex. `[64, 32]` means first layer has 64 nodes and second one has 32.
      batch_size: (int) batch size.
      learning_rate: learning rate.
      optimizer: An instance of `tf.Optimizer` used to train the model.
      activation: Activation function applied to each layer.

    Raises:
      ValueError: if label_column_name not specified.
      ValueError: if hidden_units is not a list.

    """
    if not label_column_name:
      raise ValueError('Need to specify a label_column_name.')

    if not isinstance(hidden_units, list):
      raise ValueError('hidden_units should be a list.')

    self._feature_columns = feature_columns
    self._learning_rate = learning_rate
    self._optimizer = optimizer
    self._model_dir = model_dir
    self._hidden_units = hidden_units
    self._config = config
    self._activation = activation
    self._batch_size = batch_size
    self._label_column_name = label_column_name

  def _loss(self, labels, logits):
    """Computes sigmoid cross entropy loss.

    Args:
      labels: Labels.
      logits: Logits.

    Returns:
      loss: (scalar) loss
    """
    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(sigmoid_loss)

  def _get_model_fn(self):
    """Method that gets a model_fn for creating an `Estimator` Object."""

    def model_fn(features, labels, mode):
      """BaselineModel model_fn.

      Args:
        features: `Tensor` or `dict` of `Tensor`.
        labels: A `dict` of `Tensor` Objects. Expects to have a key/value pair
          for the key self.label_column_name.
        mode: Defines whether this is training, evaluation or prediction. See
          `ModeKeys`. Currently PREDICT mode is not implemented.

      Returns:
        An instance of `tf.estimator.EstimatorSpec', which encapsulates the
        `mode`, `predictions`, `loss` and the `train_op`. Note that here
        `predictions` is either a `Tensor` or a `dict` of `Tensor` objects,
        representing the prediction of the bianry classification model.
        'loss` is a scalar containing the loss of the step and `train_op` is the
        op for training.
      """

      # Instantiates a tensor with true class labels
      class_labels = labels[self._label_column_name]

      tf.logging.info('model_fn for mode: {}'.format(mode))

      with tf.name_scope('model'):
        input_layer = tf.feature_column.input_layer(features,
                                                    self._feature_columns)
        layer = input_layer
        for unit in self._hidden_units:
          layer = tf.layers.Dense(unit, activation=self._activation)(layer)
        logits = tf.layers.Dense(1)(layer)
        sigmoid_output = tf.nn.sigmoid(logits, name='sigmoid')
        class_predictions = tf.cast(tf.greater(sigmoid_output, 0.5), tf.float32)
        tf.summary.histogram('class_predictions', class_predictions)

      # Initializes Loss Functions
      loss = self._loss(class_labels, logits)
      # Sets up dictionaries used for computing performance metrics
      predictions = {
          (self._label_column_name, 'class_ids'):
              tf.reshape(class_predictions, [-1]),
          (self._label_column_name, 'logistic'):
              tf.reshape(sigmoid_output, [-1])
      }

      class_id_kwargs = {
          'labels': class_labels,
          'predictions': class_predictions
      }
      logistics_kwargs = {'labels': class_labels, 'predictions': sigmoid_output}

      # EVAL Mode
      if mode == tf_estimator.ModeKeys.EVAL:
        with tf.name_scope('eval_metrics'):
          eval_metric_ops = {
              'accuracy': tf.metrics.accuracy(**class_id_kwargs),
              'precision': tf.metrics.precision(**class_id_kwargs),
              'recall': tf.metrics.recall(**class_id_kwargs),
              'fp': tf.metrics.false_positives(**class_id_kwargs),
              'fn': tf.metrics.false_negatives(**class_id_kwargs),
              'tp': tf.metrics.true_positives(**class_id_kwargs),
              'tn': tf.metrics.true_negatives(**class_id_kwargs),
              'fpr': contrib_metrics.streaming_false_positive_rate(**class_id_kwargs),  # pylint: disable=line-too-long
              'fnr': contrib_metrics.streaming_false_negative_rate(**class_id_kwargs),  # pylint: disable=line-too-long
              'auc': tf.metrics.auc(curve='ROC', **logistics_kwargs),
              'aucpr': tf.metrics.auc(curve='PR', **logistics_kwargs)
          }

          # EstimatorSpec object for evaluation
          estimator_spec = tf_estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              loss=loss,
              eval_metric_ops=eval_metric_ops)

      # TRAIN Mode
      if mode == tf_estimator.ModeKeys.TRAIN:
        train_op_primary = contrib_layers.optimize_loss(
            loss=loss,
            learning_rate=self._learning_rate,
            global_step=contrib_framework.get_global_step(),
            optimizer=self._optimizer)

        estimator_spec = tf_estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op_primary)

      return estimator_spec

    return model_fn


class _BaselineEstimator(tf_estimator.Estimator):
  """An estimator based on the core estimator."""

  def __init__(self, *args, **kwargs):
    """Initializes the estimator."""
    self.model = _BaselineModel(*args, **kwargs)
    super(_BaselineEstimator, self).__init__(
        model_fn=self.model._get_model_fn(),  # pylint: disable=protected-access
        model_dir=self.model._model_dir,  # pylint: disable=protected-access
        config=self.model._config  # pylint: disable=protected-access
    )


def get_estimator(*args, **kwargs):
  return _BaselineEstimator(*args, **kwargs)
