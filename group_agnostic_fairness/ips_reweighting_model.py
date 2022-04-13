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
"""A custom estimator to establish a naive reweighting baseline.

Implements a DNN classifier with re-weighted risk minimization objective,
where the weights are inverse propensity scores of the example.

Expects model_fn parameter "label" to be `dict` of `Tensor` Objects,
with key/value pair for the keys:
"IPS_example_weights_with_label" and "IPS_example_weights_without_label",
and their corresponding values being inverse propensity weight of the example.

This code merely loads the weights from the "label" dictionary, and set them as
example weights. Actual, IPS weights are precomputed somewhere else, and added
to the "label" dictionary in input_fn().
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics

IPS_WITH_LABEL_TARGET_COLUMN_NAME = 'IPS_example_weights_with_label'
IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME = 'IPS_example_weights_without_label'


class _IPSReweightingModel():
  """TensorFlow _IPSReweightingModel base class.

  _IPSReweightingModel class can be used to instantiate a feedforward DNN
  classifier with Inverse Propensity re-weighted risk minimization objective.
  """

  def __init__(
      self,
      feature_columns,
      label_column_name,
      config,
      model_dir,
      reweighting_type='IPS_without_label',
      hidden_units=[64, 32],
      batch_size=256,
      learning_rate=0.01,
      optimizer='Adagrad',
      activation=tf.nn.relu
      ):
    """Initializes a IPS reweighting estimator.

    Args:
      feature_columns: list of feature_columns.
      label_column_name: (string) name of the target variable.
      config: `RunConfig` object to configure the runtime settings.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator
        to continue training a previously saved model.
      reweighting_type: (string) name of the type of re-weighting to be
        performed. Expects values in ['IPS_with_label','IPS_without_label'].
        IPS stands for inverse propensity score, wherein each example is
        assigned a weight inversely proportionate their propensity of appearing
        in training distribution. Concretely, ips-weight = 1/p(x),
        where p(x) is the probability of x in training distribution.
        In "IPS_without_label", each example is assigned a weight as the inverse
        propensity score of their subgroup. For example, 1/p("Black Female").
        In "IPS_with_label", each example is assigned a weight as the inverse
        propensity score of their subgroup and class membership. For example,
        1/p("Black Female", "class 0")).
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
      ValueError: if reweighting_type not in
      ['IPS_with_label','IPS_without_label'].

    """
    if not label_column_name:
      raise ValueError('Need to specify a label_column_name.')

    if not isinstance(hidden_units, list):
      raise ValueError('hidden_units should be a list')

    if reweighting_type not in ('IPS_with_label', 'IPS_without_label'):
      raise ValueError('Invalid reweighting_type: {}.'.format(reweighting_type))

    self._reweighting_type = reweighting_type
    self._feature_columns = feature_columns
    self._learning_rate = learning_rate
    self._optimizer = optimizer
    self._model_dir = model_dir
    self._hidden_units = hidden_units
    self._config = config
    self._activation = activation
    self._batch_size = batch_size
    self._label_column_name = label_column_name

  def _loss(self, labels, logits, example_weights):
    """Computes weighted sigmoid cross entropy loss.

    Args:
      labels: Labels.
      logits: Logits.
      example_weights: example_weights.

    Returns:
      loss: (scalar) loss
    """
    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    weighted_loss = example_weights * sigmoid_loss
    return tf.reduce_mean(weighted_loss)

  def _get_model_fn(self):
    """Method that gets a model_fn for creating an `Estimator` Object."""

    def model_fn(features, labels, mode):
      """BaselineModel model_fn.

      Args:
        features: `Tensor` or `dict` of `Tensor`.
        labels: A `dict` of `Tensor` Objects. Expects to have a key/value pair
          for the key self.label_column_name, "IPS_example_weights_with_label",
          and "IPS_example_weights_without_label".
          IPS stands for inverse propensity score, wherein each example is
          assigned a weight inversely proportionate their propensity of
          appearing in training distribution. Concretely, ips-weight = 1/p(x),
          where p(x) is the probability of x in training distribution.
          In "IPS_without_label", each example is given a weight as the inverse
          propensity score of their subgroup. For example, 1/p("Black Female").
          In "IPS_with_label", each example is assigned a weight as the inverse
          propensity score of their subgroup and class membership. For example,
          1/p("Black Female", "class 0")).
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

      ips_example_weights_with_label = labels[IPS_WITH_LABEL_TARGET_COLUMN_NAME]
      ips_example_weights_without_label = labels[
          IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME]

      tf.logging.info('model_fn for mode: {}'.format(mode))

      with tf.name_scope('model'):
        input_layer = tf.feature_column.input_layer(features,
                                                    self._feature_columns)
        layer = input_layer
        for unit in self._hidden_units:
          layer = tf.layers.Dense(unit, activation=self._activation)(layer)
        logits = tf.layers.Dense(1)(layer)
        sigmoid_output = tf.nn.sigmoid(logits, name='sigmoid')
        class_predictions = tf.cast(tf.greater(sigmoid_output, 0.5), tf.float32)  # pylint: disable=line-too-long
        tf.summary.histogram('class_predictions', class_predictions)

      if self._reweighting_type == 'IPS_with_label':
        example_weights = ips_example_weights_with_label
      elif self._reweighting_type == 'IPS_without_label':
        example_weights = ips_example_weights_without_label

      # Initializes Loss Functions
      loss = self._loss(class_labels, logits, example_weights)

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


class _IPSReweightingEstimator(tf_estimator.Estimator):
  """An estimator based on the core estimator."""

  def __init__(self, *args, **kwargs):
    """Initializes the estimator."""
    self.model = _IPSReweightingModel(*args, **kwargs)
    super(_IPSReweightingEstimator, self).__init__(
        model_fn=self.model._get_model_fn(),  # pylint: disable=protected-access
        model_dir=self.model._model_dir,  # pylint: disable=protected-access
        config=self.model._config  # pylint: disable=protected-access
    )


def get_estimator(*args, **kwargs):
  return _IPSReweightingEstimator(*args, **kwargs)
