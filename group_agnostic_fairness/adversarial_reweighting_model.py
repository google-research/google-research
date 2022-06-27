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
"""A custom estimator for adversarial reweighting model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics


class _AdversarialReweightingModel():
  """TensorFlow AdversarialReweightingModel base class.

  AdversarialReweightingModel class can be used to define an adversarial
  reweighting estimator.

  Adversarial reweighting estimator can be used to train a model with two DNNs:
  A primary DNN that trains for the main task.
  A adversarial DNN that aims to assign weights to examples based on the
  primary's example loss.

  The two models are jointly trained to optimize for a min max problem between
  primary and adversary by alternating between the two loss functions.
  """

  def __init__(
      self,
      feature_columns,
      label_column_name,
      config,
      model_dir,
      primary_hidden_units=[64, 32],
      adversary_hidden_units=[32],
      batch_size=256,
      primary_learning_rate=0.01,
      adversary_learning_rate=0.01,
      optimizer='Adagrad',
      activation=tf.nn.relu,
      adversary_loss_type='ce_loss',
      adversary_include_label=True,
      upweight_positive_instance_only=False,
      pretrain_steps=5000
      ):
    """Initializes an adversarial reweighting estimator.

    Args:
      feature_columns: list of feature_columns.
      label_column_name: (string) name of the target variable.
      config: `RunConfig` object to configure the runtime settings.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator to
        continue training a previously saved model.
      primary_hidden_units: List with number of hidden units per layer for the
        shared bottom.  All layers are fully connected. Ex. `[64, 32]` means
        first layer has 64 nodes and second one has 32.
      adversary_hidden_units: List with number of hidden units per layer for the
        shared bottom.  All layers are fully connected. Ex. `[32]` means first
        layer has 32 nodes.
      batch_size: (int) batch size.
      primary_learning_rate: learning rate of primary DNN.
      adversary_learning_rate: learning rate of adversary DNN.
      optimizer: An instance of `tf.Optimizer` used to train the model.
      activation: Activation function applied to each layer.
      adversary_loss_type: (string) specifying the type of loss function to be
        used in adversary. Takes values in ["hinge_loss", "ce_loss"], which
        stand for hinge loss, and sigmoid cross entropy loss, respectively.
      adversary_include_label: Boolean flag. If set, adds label as input to the
        adversary feature columns.
      upweight_positive_instance_only: Boolean flag. If set, weights only
        positive examples in adversary hinge_loss.
      pretrain_steps: (int) The number of training steps for whih the model
        should train only primary model, before switching to alternate training
        between primary and adversary.

    Raises:
      ValueError: if label_column_name not specified.
      ValueError: if primary_hidden_units is not a list.
      ValueError: if adversary_hidden_units is not a list.

    """
    if not label_column_name:
      raise ValueError('Need to specify a label_column_name.')

    if not isinstance(primary_hidden_units, list):
      raise ValueError('primary_hidden_units should be a list of size 2.')

    if not isinstance(adversary_hidden_units, list):
      raise ValueError('adversary_hidden_units should be a list of size 1.')

    self._feature_columns = feature_columns
    self._primary_learning_rate = primary_learning_rate
    self._adversary_learning_rate = adversary_learning_rate
    self._optimizer = optimizer
    self._model_dir = model_dir
    self._primary_hidden_units = primary_hidden_units
    self._adversary_hidden_units = adversary_hidden_units
    self._config = config
    self._activation = activation
    self._batch_size = batch_size
    self._label_column_name = label_column_name
    self._adversary_include_label = adversary_include_label
    self._adversary_loss_type = adversary_loss_type
    self._pretrain_steps = pretrain_steps
    self._upweight_positive_instance_only = upweight_positive_instance_only

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

  def _get_hinge_loss(self, labels, logits, pos_weights):
    """Computes hinge loss over labels and logits from primary task.

    Args:
      labels: Labels.
      logits: Logits.
      pos_weights: a float tensor of shape [batch_size, 1]. Assigns weight 1
      for positive examples, and weight 0 for negative examples in the batch.

    Returns:
      loss: a float tensor of shape [batch_size, 1] containing hinge loss.
    """
    # If set, gives weight to only positive instances
    if self._upweight_positive_instance_only:
      hinge_loss = tf.losses.hinge_loss(
          labels=labels, logits=logits, weights=pos_weights, reduction='none')
    else:
      hinge_loss = tf.losses.hinge_loss(labels=labels,
                                        logits=logits,
                                        reduction='none')
    # To avoid numerical errors at loss = ``0''
    hinge_loss = tf.maximum(hinge_loss, 0.1)
    return hinge_loss

  def _get_cross_entropy_loss(self, labels, logits):
    """Computes cross-entropy loss over labels and logits from primary task."""
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

  def _adversary_loss(self,
                      labels,
                      logits,
                      pos_weights,
                      example_weights,
                      adversary_loss_type='hinge_loss'):
    """Computes (negative) adversary loss.

    At the end of this function, the calculated loss
    is multiplied with -1, so that it can be maximized later on by minimizing
    the output of this function.

    Args:
      labels: Labels.
      logits: Logits.
      pos_weights: a float tensor of shape [batch_size, 1]
        to compute weighted hinge_loss
      example_weights: a float tensor of shape [batch_size, 1] for the
        reweighting values for each example in the batch.
      adversary_loss_type: (string) flag defining which loss type to use.
        Takes values in ["hinge_loss","ce_loss"].

    Returns:
      loss: (scalar) loss
    """
    with tf.name_scope(None, 'adversary_loss', (logits, labels, pos_weights)):
      if adversary_loss_type == 'hinge_loss':
        loss = self._get_hinge_loss(labels, logits, pos_weights)
        tf.summary.histogram('hinge_loss', loss)
      elif adversary_loss_type == 'ce_loss':
        loss = self._get_cross_entropy_loss(labels, logits)
        tf.summary.histogram('ce_loss', loss)

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

  def _get_adversary_features_and_feature_columns(self, features, targets):
    """Return adversary features and feature columns."""
    adversarial_features = features.copy()
    adversary_feature_columns = self._feature_columns[:]
    # Adds label to adversarial features
    if self._adversary_include_label:
      adversary_feature_columns.append(
          tf.feature_column.numeric_column(self._label_column_name))
      adversarial_features[self._label_column_name] = targets[
          self._label_column_name]

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
      """AdversarialReweightingModel model_fn.

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

      # Instantiates a tensor with weight for positive class examples only
      pos_weights = tf.cast(tf.equal(labels[self._label_column_name], 1),
                            dtype=tf.float32)

      # Instantiates a tensor with true class labels
      class_labels = labels[self._label_column_name]

      # Initialize a global step variable used for alternate training
      current_step = self._get_or_create_global_step_var()

      if mode == tf_estimator.ModeKeys.EVAL:
        tf.logging.info('model_fn: EVAL, {}'.format(mode))
      elif mode == tf_estimator.ModeKeys.TRAIN:
        tf.logging.info('model_fn: TRAIN, {}'.format(mode))

      # Creates a DNN architecture for primary binary classification task
      with tf.name_scope('primary_NN'):
        with tf.variable_scope('primary'):
          input_layer = tf.feature_column.input_layer(features,
                                                      self._feature_columns)
          h1 = tf.layers.Dense(self._primary_hidden_units[0],
                               activation=self._activation)(input_layer)
          h2 = tf.layers.Dense(self._primary_hidden_units[1],
                               activation=self._activation)(h1)
          logits = tf.layers.Dense(1)(h2)
          sigmoid_output = tf.nn.sigmoid(logits, name='sigmoid')
          class_predictions = tf.cast(
              tf.greater(sigmoid_output, 0.5), tf.float32)
          tf.summary.histogram('class_predictions', class_predictions)

      # Creates a network architecture for the adversarial regression task
      with tf.name_scope('adversary_NN'):
        with tf.variable_scope('adversary'):
          # Gets adversary features and features columns
          adversarial_features, adversary_feature_columns = self._get_adversary_features_and_feature_columns(features, labels)  # pylint: disable=line-too-long
          adv_input_layer = tf.feature_column.input_layer(
              adversarial_features, adversary_feature_columns)
          adv_h1 = tf.layers.Dense(self._adversary_hidden_units[0])(
              adv_input_layer)
          adv_output_layer = tf.layers.Dense(1, use_bias=True)(adv_h1)
          example_weights = tf.cond(
              tf.greater(current_step, self._pretrain_steps),
              true_fn=lambda: self._compute_example_weights(adv_output_layer),
              false_fn=lambda: tf.ones_like(class_labels))

      # Adds summary variables to tensorboard
      with tf.name_scope('example_weights'):
        tf.summary.histogram('example_weights', example_weights)
        tf.summary.histogram('label', class_labels)

      # Initializes Loss Functions
      primary_loss = self._primary_loss(class_labels, logits, example_weights)
      adversary_loss = self._adversary_loss(class_labels, logits, pos_weights,
                                            example_weights,
                                            self._adversary_loss_type)

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
              loss=primary_loss,
              eval_metric_ops=eval_metric_ops)

      # TRAIN Mode
      if mode == tf_estimator.ModeKeys.TRAIN:
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
        estimator_spec = tf_estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=primary_loss + adversary_loss,
            train_op=tf.cond(
                tf.greater(current_step, self._pretrain_steps),
                true_fn=lambda: tf.group([train_op_primary, train_op_adversary]),  # pylint: disable=line-too-long
                false_fn=lambda: tf.group([train_op_primary])))

      return estimator_spec

    return model_fn


class _AdversarialReweightingEstimator(tf_estimator.Estimator):
  """An estimator based on the core estimator."""

  def __init__(self, *args, **kwargs):
    """Initializes the estimator."""
    self.model = _AdversarialReweightingModel(*args, **kwargs)
    super(_AdversarialReweightingEstimator, self).__init__(
        model_fn=self.model._get_model_fn(),  # pylint: disable=protected-access
        model_dir=self.model._model_dir,  # pylint: disable=protected-access
        config=self.model._config  # pylint: disable=protected-access
    )


def get_estimator(*args, **kwargs):
  return _AdversarialReweightingEstimator(*args, **kwargs)
