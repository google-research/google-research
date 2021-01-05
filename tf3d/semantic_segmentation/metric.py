# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Semantic segmentation metric."""
import gin
import gin.tf

import numpy as np
import tensorflow as tf

from tf3d import standard_fields
from object_detection.utils import label_map_util


@gin.configurable
class SemanticSegmentationMetric(tf.keras.metrics.Metric):
  """Semantic segmentation mean intersection over union metric."""

  def __init__(self,
               multi_label=False,
               num_classes=None,
               label_map=None,
               label_map_path=None,
               eval_prefix='eval',
               name='semantic_segmentation_metric'):
    """Semantic segmentation mean intersection over union metric.

    Args:
      multi_label: Boolean which denotes if pixels can be assigned multiple
        labels; classes are treated separately, logit > 0 is positive
        prediction.
      num_classes: Number of classes.
      label_map: A dictionary mapping label ids to label names.
      label_map_path: path to labelmap (could be None).
      eval_prefix: Prefix for eval name; separates scalar values in Tensorboard.
      name: class name.
    """
    super(SemanticSegmentationMetric, self).__init__(name=name)
    self.multi_label = multi_label
    self.num_classes = num_classes
    if label_map:
      self.label_map = label_map
    elif label_map_path:
      self.label_map = _get_label_map(label_map_path)
    else:
      self.label_map = None

    self.eval_prefix = eval_prefix
    if self.label_map is not None:
      self.class_range = self.label_map.keys()
    elif num_classes is not None:
      self.class_range = range(num_classes)
    else:
      raise ValueError('Both num_classes and label_map are None.')
    self.true_positive_metrics = {}
    self.false_positive_metrics = {}
    self.false_negative_metrics = {}
    for c in self.class_range:
      self.true_positive_metrics[c] = tf.keras.metrics.TruePositives(
          name=('%s_true_positive_%d' % (name, c)))
      self.false_positive_metrics[c] = tf.keras.metrics.FalsePositives(
          name=('%s_false_positive_%d' % (name, c)))
      self.false_negative_metrics[c] = tf.keras.metrics.FalseNegatives(
          name=('%s_false_negative_%d' % (name, c)))

  def update_state(self, inputs, outputs):
    """Function that updates the metric state at each example.

    Args:
      inputs: A dictionary containing input tensors.
      outputs: A dictionary containing output tensors.

    Returns:
      Update op.
    """
    # Prepare logits and labels
    logits = outputs[
        standard_fields.DetectionResultFields.object_semantic_points]
    labels = inputs[standard_fields.InputDataFields.object_class_points]
    weights = inputs[standard_fields.InputDataFields.point_loss_weights]
    num_valid_points = inputs[standard_fields.InputDataFields.num_valid_points]
    if len(logits.get_shape().as_list()) == 3:
      batch_size = logits.get_shape().as_list()[0]
      logits_list = []
      labels_list = []
      weights_list = []
      for i in range(batch_size):
        num_valid_points_i = num_valid_points[i]
        logits_list.append(logits[i, 0:num_valid_points_i, :])
        labels_list.append(labels[i, 0:num_valid_points_i, :])
        weights_list.append(weights[i, 0:num_valid_points_i, :])
      logits = tf.concat(logits_list, axis=0)
      labels = tf.concat(labels_list, axis=0)
      weights = tf.concat(weights_list, axis=0)
    if self.num_classes is None:
      num_classes = logits.get_shape().as_list()[-1]
    else:
      num_classes = self.num_classes
      if num_classes != logits.get_shape().as_list()[-1]:
        raise ValueError('num_classes do not match the logits dimensions.')

    class_labels, class_predictions = _get_class_labels_and_predictions(
        labels=labels,
        logits=logits,
        num_classes=self.num_classes,
        multi_label=self.multi_label)

    update_ops = []
    for c in self.class_range:
      update_op_tp_c = self.true_positive_metrics[c].update_state(
          y_true=class_labels[c],
          y_pred=class_predictions[c],
          sample_weight=weights)
      update_ops.append(update_op_tp_c)
      update_op_fp_c = self.false_positive_metrics[c].update_state(
          y_true=class_labels[c],
          y_pred=class_predictions[c],
          sample_weight=weights)
      update_ops.append(update_op_fp_c)
      update_op_fn_c = self.false_negative_metrics[c].update_state(
          y_true=class_labels[c],
          y_pred=class_predictions[c],
          sample_weight=weights)
      update_ops.append(update_op_fn_c)
    return tf.group(update_ops)

  def result(self):
    metrics_dict = self.get_metric_dictionary()
    return metrics_dict[self.eval_prefix + '_avg/mean_iou']

  def get_metric_dictionary(self):
    metrics_dict = {}
    class_recall_list = []  # used for calculating mean pixel accuracy.
    class_iou_list = []     # used for calculating mean iou.
    for c in self.class_range:
      tp = self.true_positive_metrics[c].result()
      fp = self.false_positive_metrics[c].result()
      fn = self.false_negative_metrics[c].result()
      class_recall = tp / (tp + fn)
      class_precision = tf.where(
          tf.greater(tp + fn, 0.0), _safe_div(tp, (tp + fp)),
          tf.constant(np.NaN))
      class_iou = tf.where(
          tf.greater(tp + fn, 0.0), tp / (tp + fn + fp), tf.constant(np.NaN))
      class_recall_list.append(class_recall)
      class_iou_list.append(class_iou)
      class_name = _get_class_name(class_id=c, label_map=self.label_map)
      metrics_dict[self.eval_prefix +
                   '_recall/{}'.format(class_name)] = class_recall
      metrics_dict[self.eval_prefix +
                   '_precision/{}'.format(class_name)] = class_precision
      metrics_dict[self.eval_prefix + '_iou/{}'.format(class_name)] = class_iou
    mean_pixel_accuracy = _non_nan_mean(class_recall_list)
    mean_iou = _non_nan_mean(class_iou_list)
    metrics_dict[self.eval_prefix +
                 '_avg/mean_pixel_accuracy'] = mean_pixel_accuracy
    metrics_dict[self.eval_prefix + '_avg/mean_iou'] = mean_iou
    return metrics_dict

  def reset_states(self):
    for _, value in self.true_positive_metrics.items():
      value.reset_states()
    for _, value in self.false_positive_metrics.items():
      value.reset_states()
    for _, value in self.false_negative_metrics.items():
      value.reset_states()


def _get_class_labels_and_predictions(labels, logits, num_classes, multi_label):
  """Returns list of per-class-labels and list of per-class-predictions.

  Args:
    labels: A `Tensor` of size [n, k]. In the
      multi-label case, values are either 0 or 1 and k = num_classes. Otherwise,
      k = 1 and values are in [0, num_classes).
    logits: A `Tensor` of size [n, `num_classes`]
      representing the logits of each pixel and semantic class.
    num_classes: Number of classes.
    multi_label: Boolean which defines if we are in a multi_label setting, where
      pixels can have multiple labels, or not.

  Returns:
    class_labels: List of size num_classes, where each entry is a `Tensor' of
      size [batch_size, height, width] of type float with values of 0 or 1
      representing the ground truth labels.
    class_predictions: List of size num_classes, each entry is a `Tensor' of
      size [batch_size, height, width] of type float with values of 0 or 1
      representing the predicted labels.
  """
  class_predictions = [None] * num_classes
  if multi_label:
    class_labels = tf.split(labels, num_or_size_splits=num_classes, axis=1)
    class_logits = tf.split(logits, num_or_size_splits=num_classes, axis=1)
    for c in range(num_classes):
      class_predictions[c] = tf.cast(
          tf.greater(class_logits[c], 0), dtype=tf.float32)
  else:
    class_predictions_flat = tf.argmax(logits, 1)
    class_labels = [None] * num_classes
    for c in range(num_classes):
      class_labels[c] = tf.cast(tf.equal(labels, c), dtype=tf.float32)
      class_predictions[c] = tf.cast(
          tf.equal(class_predictions_flat, c), dtype=tf.float32)
  return class_labels, class_predictions


def _get_class_name(class_id, label_map):
  """Gets class name from label dictionary."""
  if label_map and class_id in label_map:
    return label_map[class_id]
  else:
    return str(class_id)


def _non_nan_mean(tensor_list):
  """Calculates the mean of a list of tensors while ignoring nans."""
  tensor = tf.stack(tensor_list)
  not_nan = tf.logical_not(tf.math.is_nan(tensor))
  return tf.reduce_mean(tf.boolean_mask(tensor, not_nan))


def _safe_div(a, b):
  """Divides two numbers, returns 0 if denominator is (close to) 0."""
  return tf.where(tf.less(tf.abs(b), 1e-10), 0.0, a / b)


def _get_label_map(label_map_path):
  """Returns dictionary mapping label IDs to class-names."""
  if not label_map_path:
    return None

  label_map_proto = label_map_util.load_labelmap(label_map_path)
  label_map = {}
  for item in label_map_proto.item:
    if item.HasField('display_name'):
      label_map[item.id] = item.display_name
    elif item.HasField('name'):
      label_map[item.id] = item.name

  return label_map
