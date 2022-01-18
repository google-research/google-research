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

"""The metric functions for evaluating sequence tagging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import math_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import state_ops  # pylint: disable=g-direct-tensorflow-import


def streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
  """Calculate a streaming confusion matrix.

  Calculates a confusion matrix. For estimation over a stream of data,
  the function creates an  `update_op` operation.

  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can have.
      This value must be provided, since a confusion matrix of dimension =
      [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).

  Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
  """
  with tf.variable_scope(None, "streaming_confusion_matrix",
                         [predictions, labels]):

    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = slim.local_variable(
        tf.zeros([num_classes, num_classes], tf.float64),
        name="total_confusion_matrix")
    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.cast(predictions, tf.int64)
    labels = math_ops.cast(labels, tf.int64)
    num_classes = math_ops.cast(num_classes, tf.int64)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
      predictions = array_ops.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
      labels = array_ops.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
      weights = array_ops.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = tf.math.confusion_matrix(
        labels,
        predictions,
        tf.to_int64(num_classes),
        weights=weights,
        dtype=tf.float64,
        name="current_confusion_matrix")
    update_op = state_ops.assign_add(total_cm, current_cm)
    return total_cm, update_op


def precision(labels,
              predictions,
              num_classes,
              pos_indices=None,
              weights=None,
              average="micro"):
  """Multi-class precision metric for Tensorflow.

  Args:
    labels: Tensor of tf.int32 or tf.int64 The true labels
    predictions: Tensor of tf.int32 or tf.int64 The predictions, same shape as
      labels
    num_classes: int The number of classes
    pos_indices: list of int, optional The indices of the positive classes,
      default is all
    weights: Tensor of tf.int32, optional Mask, must be of compatible shape with
      labels
    average : str, optional
        "micro": counts the total number of true positives, false positives, and
          false negatives for the classes in `pos_indices` and infer the metric
          from it.
        "macro": will compute the metric separately for each class in
          `pos_indices` and average. Will not account for class imbalance.
        "weighted": will compute the metric separately for each class in
          `pos_indices` and perform a weighted average by the total number of
          true labels for each class.

  Returns:
    tuple of (scalar float Tensor, update_op)
  """
  cm, op = streaming_confusion_matrix(labels, predictions, num_classes, weights)
  pr, _, _ = metrics_from_confusion_matrix(cm, pos_indices, average=average)
  op, _, _ = metrics_from_confusion_matrix(op, pos_indices, average=average)
  return (pr, op)


def recall(labels,
           predictions,
           num_classes,
           pos_indices=None,
           weights=None,
           average="micro"):
  """Multi-class recall metric for Tensorflow.

  Args:
    labels : Tensor of tf.int32 or tf.int64 The true labels
    predictions : Tensor of tf.int32 or tf.int64 The predictions, same shape as
      labels
    num_classes : int The number of classes
    pos_indices : list of int, optional The indices of the positive classes,
      default is all
    weights : Tensor of tf.int32, optional Mask, must be of compatible shape
      with labels
    average : str, optional
        "micro": counts the total number of true positives, false positives, and
          false negatives for the classes in `pos_indices` and infer the metric
          from it.
        "macro": will compute the metric separately for each class in
          `pos_indices` and average. Will not account for class imbalance.
        "weighted": will compute the metric separately for each class in
          `pos_indices` and perform a weighted average by the total number of
          true labels for each class.

  Returns:
    tuple of (scalar float Tensor, update_op)
  """
  cm, op = streaming_confusion_matrix(labels, predictions, num_classes, weights)
  _, re, _ = metrics_from_confusion_matrix(cm, pos_indices, average=average)
  _, op, _ = metrics_from_confusion_matrix(op, pos_indices, average=average)
  return (re, op)


def f1(labels,
       predictions,
       num_classes,
       pos_indices=None,
       weights=None,
       average="micro"):
  return fbeta(labels, predictions, num_classes, pos_indices, weights, average)


def fbeta(labels,
          predictions,
          num_classes,
          pos_indices=None,
          weights=None,
          average="micro",
          beta=1):
  """Multi-class fbeta metric for Tensorflow.

  Args:
    labels : Tensor of tf.int32 or tf.int64 The true labels
    predictions : Tensor of tf.int32 or tf.int64 The predictions, same shape as
      labels
    num_classes : int The number of classes
    pos_indices : list of int, optional The indices of the positive classes,
      default is all
    weights : Tensor of tf.int32, optional Mask, must be of compatible shape
      with labels
    average : str, optional
        "micro": counts the total number of true positives, false positives, and
          false negatives for the classes in `pos_indices` and infer the metric
          from it.
        "macro": will compute the metric separately for each class in
          `pos_indices` and average. Will not account for class imbalance.
        "weighted": will compute the metric separately for each class in
          `pos_indices` and perform a weighted average by the total number of
          true labels for each class.
    beta : int, optional Weight of precision in harmonic mean

  Returns:
    tuple of (scalar float Tensor, update_op)
  """
  cm, op = streaming_confusion_matrix(labels, predictions, num_classes, weights)
  _, _, fbeta_score = metrics_from_confusion_matrix(
      cm, pos_indices, average=average, beta=beta)
  _, _, op = metrics_from_confusion_matrix(
      op, pos_indices, average=average, beta=beta)
  return (fbeta_score, op)


def safe_div(numerator, denominator):
  """Does safe division, returns 0 if denominator is 0."""
  numerator, denominator = tf.to_float(numerator), tf.to_float(denominator)
  zeros = tf.zeros_like(numerator, dtype=numerator.dtype)
  denominator_is_zero = tf.equal(denominator, zeros)
  return tf.where(denominator_is_zero, zeros, numerator / denominator)


def pr_re_fbeta(cm, pos_indices, beta=1):
  """Uses a confusion matrix to compute precision, recall and fbeta."""
  num_classes = cm.shape[0]
  neg_indices = [i for i in range(num_classes) if i not in pos_indices]
  cm_mask = np.ones([num_classes, num_classes])
  cm_mask[neg_indices, neg_indices] = 0
  diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))

  cm_mask = np.ones([num_classes, num_classes])
  cm_mask[:, neg_indices] = 0
  tot_pred = tf.reduce_sum(cm * cm_mask)

  cm_mask = np.ones([num_classes, num_classes])
  cm_mask[neg_indices, :] = 0
  tot_gold = tf.reduce_sum(cm * cm_mask)

  pr = safe_div(diag_sum, tot_pred)
  re = safe_div(diag_sum, tot_gold)
  fbeta_score = safe_div((1. + beta**2) * pr * re, beta**2 * pr + re)

  return pr, re, fbeta_score


def metrics_from_confusion_matrix(cm,
                                  pos_indices=None,
                                  average="micro",
                                  beta=1):
  """Generates Precision, Recall and F1 from the confusion matrix.

  Args:
    cm: tf.Tensor of type tf.int32, of shape (num_classes, num_classes) The
      streaming confusion matrix.
    pos_indices: list of int, optional The indices of the positive classes
    average: str, optional "micro", "macro" or "weighted"
    beta: int, optional Weight of precision in harmonic mean

  Returns:
    the confusion_matrix
  """
  num_classes = cm.shape[0]
  if pos_indices is None:
    pos_indices = [i for i in range(num_classes)]

  if average == "micro":
    return pr_re_fbeta(cm, pos_indices, beta)
  elif average in {"macro", "weighted"}:
    precisions, recalls, fbetas, n_golds = [], [], [], []
    for idx in pos_indices:
      pr, re, fbeta_score = pr_re_fbeta(cm, [idx], beta)
      precisions.append(pr)
      recalls.append(re)
      fbetas.append(fbeta_score)
      cm_mask = np.zeros([num_classes, num_classes])
      cm_mask[idx, :] = 1
      n_golds.append(tf.to_float(tf.reduce_sum(cm * cm_mask)))

    if average == "macro":
      pr = tf.reduce_mean(precisions)
      re = tf.reduce_mean(recalls)
      fbeta_score = tf.reduce_mean(fbetas)
      return pr, re, fbeta_score
    if average == "weighted":
      n_gold = tf.reduce_sum(n_golds)
      pr_sum = sum(p * n for p, n in zip(precisions, n_golds))
      pr = safe_div(pr_sum, n_gold)
      re_sum = sum(r * n for r, n in zip(recalls, n_golds))
      re = safe_div(re_sum, n_gold)
      fbeta_sum = sum(f * n for f, n in zip(fbetas, n_golds))
      fbeta_score = safe_div(fbeta_sum, n_gold)
      return pr, re, fbeta_score

  else:
    raise NotImplementedError()
