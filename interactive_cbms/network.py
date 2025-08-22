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

"""Interactive Bottleneck Model."""

from typing import Dict, List, Optional, Tuple, Any
import tensorflow as tf

from interactive_cbms import enum_utils

tfk = tf.keras


class InteractiveBottleneckModel(tfk.Model):
  """Interactive Bottleneck Model class."""

  def __init__(self,
               arch = enum_utils.Arch.X_TO_C_TO_Y,
               n_concepts = 112,
               n_classes = 200,
               non_linear_ctoy = False):
    """Initialises an InteractiveBottleneckModel object.

    Args:
      arch: Architecture to use. Allowed values are 'XtoC', 'CtoY', 'XtoCtoY',
        'XtoCtoY_sigmoid' or 'XtoY'
      n_concepts: Number of binary concepts at the bottleneck.
      n_classes: Number of classes.
      non_linear_ctoy: Whether to use a non-linear CtoY model.
    """
    super().__init__()
    self.n_classes = n_classes
    self.n_concepts = n_concepts
    if arch is enum_utils.Arch.X_TO_C_TO_Y_SIGMOID:
      self.use_sigmoid = True
      arch = enum_utils.Arch.X_TO_C_TO_Y
    else:
      self.use_sigmoid = False
    self.arch = arch
    if arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_C_TO_Y,
                enum_utils.Arch.X_TO_Y]:
      self.base_model = tfk.applications.inception_v3.InceptionV3(
          weights='imagenet', include_top=False)
      self.gap = tfk.layers.GlobalAveragePooling2D()
      if arch is enum_utils.Arch.X_TO_Y:
        self.linear = tfk.layers.Dense(n_concepts, activation='relu')
      else:
        self.linear = tfk.layers.Dense(n_concepts, activation=None)

    if arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y,
                enum_utils.Arch.X_TO_Y]:
      if non_linear_ctoy:
        self.ctoy_module = tfk.Sequential([
            tfk.layers.Dense(128, activation='relu'),
            tfk.layers.Dense(128, activation='relu'),
            tfk.layers.Dense(n_classes, activation=None)])
      else:
        self.ctoy_module = tfk.layers.Dense(n_classes, activation=None)

  def compile(self,
              optimizer = None,
              loss_weights = None):
    """Sets up losses and metrics depending on the arch selected.

    This method overrides the compile() method of the parent tfk.Model.

    Args:
      optimizer: Optimizer to use.
      loss_weights: A list of lists containing the loss weights for the concept
        and class losses. As there is just one loss term each for concept and
        class predictions currently, this should look like
        [[<concept_loss_weight>], [<class_loss_weight>]]. However the nested
        lists would have more than one element if we want to have multiple
        losses for concept and class predictions.
    """
    super().compile(optimizer=optimizer)

    self.loss_weights = loss_weights
    self.custom_losses = []
    self.custom_loss_metrics = []
    self.custom_alt_metrics = []
    loss_weights = []

    # The order of appends below is important as it is used in compute_loss() to
    # match model outputs and losses/metrics.
    if self.arch in [enum_utils.Arch.X_TO_C, enum_utils.Arch.X_TO_C_TO_Y]:
      # Add losses and metrics for concept predictions.
      self.custom_losses.append(
          [tfk.losses.BinaryCrossentropy(from_logits=True)])
      self.custom_loss_metrics.append([tfk.metrics.Mean(name='concept_loss')])
      self.custom_alt_metrics.append([
          tfk.metrics.AUC(from_logits=True, name='concept_auroc'),
          tfk.metrics.BinaryAccuracy(threshold=0, name='concept_accuracy')
      ])
      loss_weights.append([1])
    if self.arch in [enum_utils.Arch.C_TO_Y, enum_utils.Arch.X_TO_C_TO_Y,
                     enum_utils.Arch.X_TO_Y]:
      # Add losses and metrics for class predictions.
      if self.n_classes == 1:
        self.custom_losses.append(
            [tfk.losses.BinaryCrossentropy(from_logits=True)])
        self.custom_alt_metrics.append([
            tfk.metrics.AUC(from_logits=True, name='class_auroc', curve='ROC'),
            tfk.metrics.AUC(from_logits=True, name='class_auprc', curve='PR'),
            tfk.metrics.BinaryAccuracy(threshold=0, name='class_accuracy')
        ])
      else:
        self.custom_losses.append(
            [tfk.losses.SparseCategoricalCrossentropy(from_logits=True)])
        self.custom_alt_metrics.append([
            tfk.metrics.SparseTopKCategoricalAccuracy(
                k=1, name='top-1_class_accuracy'),
            tfk.metrics.SparseTopKCategoricalAccuracy(
                k=2, name='top-2_class_accuracy'),
            tfk.metrics.SparseTopKCategoricalAccuracy(
                k=3, name='top-3_class_accuracy')
        ])
      self.custom_loss_metrics.append([tfk.metrics.Mean(name='class_loss')])
      loss_weights.append([1])

    if self.loss_weights is None:
      self.loss_weights = loss_weights

  @tf.function
  def compute_loss(self, y,
                   y_pred):
    """Computes the loss and updates the loss metrics.

    Args:
      y: A tuple containing true concepts and/or class labels. This could look
        like (<concept_labels>, <class_labels>),  (<concept_labels>,), or
        (<class_labels>).
      y_pred: A typle similar to y, but containing predicted concepts and/or
        class labels

    Returns:
      The loss.
    """
    losses = []
    for loss_fns_i, loss_weight_i, metric_fns_i, y_i, y_pred_i in zip(
        self.custom_losses, self.loss_weights, self.custom_loss_metrics, y,
        y_pred):
      for loss_fn, loss_weight, metric_fn in zip(loss_fns_i, loss_weight_i,
                                                 metric_fns_i):
        loss = loss_fn(y_i, y_pred_i)
        losses.append(loss_weight * loss)
        metric_fn.update_state(loss)
    return tf.add_n(losses)

  def update_metrics(self, y,
                     y_pred):
    """Updates alternative metrics like accuracy, AUC, etc.

    Args:
      y: A tuple containing true concepts and/or class labels. This could look
        like (<concept_labels>, <class_labels>),  (<concept_labels>,), or
        (<class_labels>).
      y_pred: A typle similar to y, but containing predicted concepts and/or
        class labels
    """
    for metric_fns_i, y_i, y_pred_i in zip(self.custom_alt_metrics, y, y_pred):
      for metric_fn in metric_fns_i:
        metric_fn.update_state(y_i, y_pred_i)

  @property
  def metrics(self):
    """Returns a list of all metrics in use."""
    metrics = []
    for metrics_i in self.custom_loss_metrics + self.custom_alt_metrics:
      metrics.extend(metrics_i)
    return metrics

  def call(self,
           inputs,
           training = False, mask = None):
    """Executes a forward pass through the network.

    This method overrides the call() method of the parent tfk.Model.

    Args:
      inputs: A tensor containing the input batch.
      training: Whether the model is training
      mask: A mask or list of masks. A mask can be either a boolean tensor
              or None (no mask).

    Returns:
      A tuple containing the model predictions. This could look like
      (<concept_preds>, <class_preds>),  (<concept_preds>,), or
      (<class_preds>).
    """
    if self.arch is enum_utils.Arch.X_TO_C:
      return (self.linear(
          self.gap(
              self.base_model(inputs, training=training), training=training),
          training=training),)
    elif self.arch is enum_utils.Arch.C_TO_Y:
      return (self.ctoy_module(inputs, training=training),)
    elif self.arch is enum_utils.Arch.X_TO_C_TO_Y:
      concept_logits = self.linear(
          self.gap(
              self.base_model(inputs, training=training), training=training),
          training=training)
      if self.use_sigmoid:
        class_logits = self.ctoy_module(
            tf.sigmoid(concept_logits), training=training)
      else:
        class_logits = self.ctoy_module(concept_logits, training=training)
      return (concept_logits, class_logits)
    else:  # When self.arch is enum_utils.Arch.X_TO_Y
      linear_out = self.linear(
          self.gap(
              self.base_model(inputs, training=training), training=training),
          training=training)
      class_logits = self.ctoy_module(linear_out, training=training)
      return (class_logits,)

  def get_x_y_from_data(
      self, data):
    """Unpacks the model inputs and ground truth outputs from a tuple containing the data batch.

    Args:
      data: A tuple of image, concept_label, class_label (and optionally, the
        concept uncertainty labels as well)

    Returns:
      x: the model input tensor.
      y: a tuple of ground truth output tensors.
    """
    image, concept_label, class_label = data[:3]
    if self.arch is enum_utils.Arch.X_TO_C:
      x = image
      y = (concept_label,)
    elif self.arch is enum_utils.Arch.C_TO_Y:
      x = concept_label
      y = (class_label,)
    elif self.arch is enum_utils.Arch.X_TO_C_TO_Y:
      x = image
      y = (concept_label, class_label)
    elif self.arch is enum_utils.Arch.X_TO_Y:
      x = image
      y = (class_label,)
    return x, y

  def train_step(self, data):
    """Implements the logic for one (custom) training step.

    This method overrides the train_step() method of the parent tfk.Model.

    Args:
      data: A tuple of image, concept_label, class_label (and optionally, the
        concept uncertainty labels as well)

    Returns:
      A dictionary of the format {<metric_name>: <metric_value>, ...} containing
      the metrics.
    """
    x, y = self.get_x_y_from_data(data)
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compute_loss(y, y_pred)

    gradients = tape.gradient(loss, self.trainable_variables)

    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.update_metrics(y, y_pred)

    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    """Implements the logic for one (custom) test step.

    This method overrides the test_step() method of the parent tfk.Model.

    Args:
      data: A tuple of image, concept_label, class_label (and optionally, the
        concept uncertainty labels as well)

    Returns:
      A dictionary of the format {<metric_name>: <metric_value>, ...} containing
      the metrics.
    """
    x, y = self.get_x_y_from_data(data)
    y_pred = self(x, training=False)
    self.compute_loss(y, y_pred)
    self.update_metrics(y, y_pred)

    return {m.name: m.result() for m in self.metrics}

  def predict_step(self, data):
    """Implements the logic for one (custom) predict step.

    This method overrides the predict_step() method of the parent tfk.Model.

    Args:
      data: A tuple of image, concept_label, class_label (and optionally, the
        concept uncertainty labels as well)

    Returns:
      A tuple containing the model predictions. This could look like
      (<concept_preds>, <class_preds>),  (<concept_preds>,), or
      (<class_preds>).
    """
    x, _ = self.get_x_y_from_data(data)
    return self(x, training=False)
