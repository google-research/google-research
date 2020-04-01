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
# pylint: disable=dangerous-default-value
"""A custom estimator for PseudoAFL model (adversarial subgroup reweighting) robust learning model.

Adversarial Subgroup Reweighting estimator trains a robust learning model with
two DNNs:
A primary DNN that trains for the main task, and has access to ALL FEATURES.
A adversarial DNN that has access to ONLY PROTECTED FEATURES. The aim of the
adversary is to adversarially assign examples-weights based on their protected
group membership, to maximize learner's loss.

The two models are jointly trained to optimize for a min max problem between
primary and adversary by alternating between the two loss functions.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics


class _AdversarialSubgroupReweightingModel():
  """TensorFlow _AdversarialSubgroupReweightingModel base class.

  Adversarial Subgroup Reweighting estimator trains a robust learning model with
  two DNNs:
  A primary DNN that trains for the main task, and has access to all features.
  A adversarial DNN that has access to only protected features. The aim of the
  adversary is to adversarially assign examples-weights based on their protected
  group membership, to maximize learner's loss.

  The two models are jointly trained to optimize for a min max problem between
  primary and adversary by alternating between the two loss functions.
  """

  def __init__(self,
               feature_columns,
               label_column_name,
               protected_column_names,
               config,
               model_dir,
               primary_hidden_units=[64, 32],
               adversary_hidden_units=[32],
               batch_size=256,
               primary_learning_rate=0.01,
               adversary_learning_rate=0.01,
               optimizer='Adagrad',
               activation=tf.nn.relu,
               pretrain_steps=5000):
    """Initializes a robust estimator.

    Args:
      feature_columns: list of feature_columns.
      label_column_name: (string) name of the target variable.
      protected_column_names: list of protected column names.
      config: `RunConfig` object to configure the runtime settings.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator
        to continue training a previously saved model.
      primary_hidden_units: List with number of hidden units per layer for the
        shared bottom.  All layers are fully connected.
        Ex. `[64, 32]` means first layer has 64 nodes and second one has 32.
      adversary_hidden_units: List with number of hidden units per layer for the
        shared bottom.  All layers are fully connected.
        Ex. `[32]` means first layer has 32 nodes.
      batch_size: (int) batch size.
      primary_learning_rate: learning rate of primary DNN.
      adversary_learning_rate: learning rate of adversary DNN.
      optimizer: An instance of `tf.Optimizer` used to train the model.
      activation: Activation function applied to each layer.
      pretrain_steps: (int) The number of training steps for whih the model
        should train only primary model, before switching to alternate training
        between primary and adversary.

    Raises:
      ValueError: if label_column_name not specified.
      ValueError: if protected_column_names is not a list.
      ValueError: if primary_hidden_units is not a list.
      ValueError: if adversary_hidden_units is not a list.
      ValueError: if protected_column_names not in feature_columns

    """
    if not label_column_name:
      raise ValueError('Need to specify a label_column_name.')

    if not isinstance(protected_column_names, list):
      raise ValueError('protected_column_names should be a list.')

    if not isinstance(primary_hidden_units, list):
      raise ValueError('primary_hidden_units should be a list.')

    if not isinstance(adversary_hidden_units, list):
      raise ValueError('adversary_hidden_units should be a list.')

    feature_columns_names = [x[0] for x in  feature_columns]
    for col in protected_column_names:
      if col not in feature_columns_names:
        raise ValueError(
            'protected column <{}> should be in feature_columns.'.format(col))

    self._feature_columns = feature_columns
    self._label_column_name = label_column_name
    self._protected_column_names = protected_column_names
    self._config = config
    self._model_dir = model_dir
    self._primary_hidden_units = primary_hidden_units
    self._adversary_hidden_units = adversary_hidden_units
    self._batch_size = batch_size
    self._primary_learning_rate = primary_learning_rate
    self._adversary_learning_rate = adversary_learning_rate
    self._optimizer = optimizer
    self._activation = activation

    self._pretrain_steps = pretrain_steps

  def _primary_loss(self, labels, logits, example_weights):
    """Computes weighted sigmoid cross entropy loss.

    Args:
      labels: Labels.
      logits: Logits.
      example_weights: a float tensor of shape [batch_size, 1] for the
        reweighting values for each example in the batch.

    Returns:
      loss: (scalar) loss
    """
    with tf.name_scope(None, 'primary_loss', (logits, labels)) as name:
      sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits, name=name)
      primary_weighted_loss = (example_weights * sigmoid_loss)
      primary_weighted_loss = tf.reduce_mean(primary_weighted_loss)
      return primary_weighted_loss

  def _adversary_loss(self,
                      labels,
                      logits,
                      example_weights):
    """Computes (negative) cross-entropy loss over labels and logits from primary task.

    At the end of this function, the calculated loss
    is multiplied with -1, so that it can be maximized later on by minimizing
    the output of this function.

    Args:
      labels: Labels.
      logits: Logits.
      example_weights: a float tensor of shape [batch_size, 1] for the
        reweighting values for each example in the batch.

    Returns:
      loss: (scalar) loss
    """
    with tf.name_scope(None, 'adversary_loss', (logits, labels)):
      # Computes sigmoid cross-entropy loss
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)

      # Multiplies loss by -1 so that the adversary loss is maximimized.
      adversary_weighted_loss = -(example_weights * loss)

      return tf.reduce_mean(adversary_weighted_loss)

  def _get_or_create_global_step_var(self):
    """Return the global_step variable, creating it if it does not exist.

    Prefer GetGlobalStep if a tensor rather than a tf.Variable is sufficient.

    Returns:
      The global_step variable, or a new created one if it does not exist.
    """
    return tf.train.get_or_create_global_step()

  def _get_adversary_features_and_feature_columns(self, features):
    """Return adversary features and feature columns.

    Filters and returns only protected features and feature columns.

    Args:
      features: `dict` of `Tensor`.

    Returns:
      adversary_features: `dict` of `Tensor`.
      adversary_feature_columns: a list of feature_columns.
    """

    # Filter features_columns andkeep only protected feature_columns.
    # # Feature_columns is a list of tf.feature_column, via x[0] we are
    # # accessing the name of the <feature_column>.
    adversary_feature_columns = [
        x for x in self._feature_columns if x[0] in self._protected_column_names
    ]

    adversarial_features = {
        key: features[key]
        for key in features
        if key in self._protected_column_names
        }

    return adversarial_features, adversary_feature_columns

  def _compute_example_weights(self, adv_output_layer):
    """Applies sigmoid to adversary output layer and returns normalized example weight."""
    example_weights = tf.nn.sigmoid(adv_output_layer)
    mean_example_weights = tf.reduce_mean(example_weights)
    example_weights /= tf.maximum(mean_example_weights, 1e-4)
    example_weights = tf.ones_like(example_weights)+example_weights
    return example_weights

  def _get_model_fn(self):
    """Method that gets a model_fn for creating an `Estimator` Object."""

    def model_fn(features, labels, mode):
      """robustModel model_fn.

      Args:
        features: `dict` of `Tensor`.
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

      Raises:
        ValueError: if protected_column_names not in feature_columns
      """
      for col in self._protected_column_names:
        if col not in features.keys():
          raise ValueError(
              'Protected column <{}> should be in features.'.format(col))

      # Instantiates a tensor with true class labels
      class_labels = labels[self._label_column_name]

      # Initialize a global step variable used for alternate training
      current_step = self._get_or_create_global_step_var()

      tf.logging.info('model_fn for mode: {}'.format(mode))

      with tf.name_scope('primary_NN'):
        with tf.variable_scope('primary'):
          input_layer = tf.feature_column.input_layer(features,
                                                      self._feature_columns)
          layer = input_layer
          for unit in self._primary_hidden_units:
            layer = tf.layers.Dense(unit, activation=self._activation)(layer)
          logits = tf.layers.Dense(1)(layer)
          sigmoid_output = tf.nn.sigmoid(logits, name='sigmoid')
          class_predictions = tf.cast(tf.greater(sigmoid_output, 0.5), tf.float32)  # pylint: disable=line-too-long
          tf.summary.histogram('class_predictions', class_predictions)

      with tf.name_scope('adversary_NN'):
        with tf.variable_scope('adversary'):
          # Filters and keeps only protected features and feature columns.
          adversarial_features, adversary_feature_columns = self._get_adversary_features_and_feature_columns(features)  # pylint: disable=line-too-long

          adv_input_layer = tf.feature_column.input_layer(
              adversarial_features, adversary_feature_columns)

          adv_layer = adv_input_layer
          for adv_unit in self._adversary_hidden_units:
            adv_layer = tf.layers.Dense(adv_unit)(adv_layer)
          adv_output_layer = tf.layers.Dense(1, use_bias=True)(adv_layer)

          example_weights = tf.cond(
              tf.greater(current_step, self._pretrain_steps),
              true_fn=lambda: self._compute_example_weights(adv_output_layer),
              false_fn=lambda: tf.ones_like(class_labels))

      # Initializes Loss Functions
      primary_loss = self._primary_loss(class_labels, logits, example_weights)
      adversary_loss = self._adversary_loss(class_labels, logits,
                                            example_weights)

      # Sets up dictionaries used for computing performance metrics
      predictions = {
          (self._label_column_name, 'class_ids'):
              tf.reshape(class_predictions, [-1]),
          (self._label_column_name, 'logistic'):
              tf.reshape(sigmoid_output, [-1]),
          ('example_weights'):
              tf.reshape(example_weights, [-1])
      }

      class_id_kwargs = {
          'labels': class_labels,
          'predictions': class_predictions
      }
      logistics_kwargs = {'labels': class_labels, 'predictions': sigmoid_output}

      # EVAL Mode
      if mode == tf.estimator.ModeKeys.EVAL:
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
          estimator_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              loss=primary_loss,
              eval_metric_ops=eval_metric_ops)

      # TRAIN Mode
      if mode == tf.estimator.ModeKeys.TRAIN:
        # Filters trainable variables for each task
        all_trainable_vars = tf.trainable_variables()
        primary_trainable_vars = [
            v for v in all_trainable_vars if 'primary' in v.op.name
        ]
        adversary_trainable_vars = [
            v for v in all_trainable_vars if 'adversary' in v.op.name
        ]

        # TRAIN_OP for adversary DNN
        train_op_adversary = contrib_layers.optimize_loss(
            loss=adversary_loss,
            variables=adversary_trainable_vars,
            global_step=contrib_framework.get_global_step(),
            learning_rate=self._adversary_learning_rate,
            optimizer=self._optimizer)

        # TRAIN_OP for primary DNN
        train_op_primary = contrib_layers.optimize_loss(
            loss=primary_loss,
            variables=primary_trainable_vars,
            global_step=contrib_framework.get_global_step(),
            learning_rate=self._primary_learning_rate,
            optimizer=self._optimizer)

        # Upto ``pretrain_steps'' trains primary only.
        # Beyond ``pretrain_steps'' alternates between primary and adversary.
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=primary_loss + adversary_loss,
            train_op=tf.cond(
                tf.greater(current_step, self._pretrain_steps),
                true_fn=lambda: tf.group([train_op_primary, train_op_adversary]),  # pylint: disable=line-too-long
                false_fn=lambda: tf.group([train_op_primary])))

      return estimator_spec

    return model_fn


class _AdversarialSubgroupReweightingEstimator(tf.estimator.Estimator):
  """An estimator based on the core estimator."""

  def __init__(self, *args, **kwargs):
    """Initializes the estimator."""
    self.model = _AdversarialSubgroupReweightingModel(*args, **kwargs)
    super(_AdversarialSubgroupReweightingEstimator, self).__init__(
        model_fn=self.model._get_model_fn(),  # pylint: disable=protected-access
        model_dir=self.model._model_dir,  # pylint: disable=protected-access
        config=self.model._config  # pylint: disable=protected-access
    )


def get_estimator(*args, **kwargs):
  return _AdversarialSubgroupReweightingEstimator(*args, **kwargs)
