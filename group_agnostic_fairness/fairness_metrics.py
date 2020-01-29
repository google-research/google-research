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
"""Implements evaluation metrics for ML fairness."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib import metrics as contrib_metrics


class RobustFairnessMetrics():
  """Implements evaluation metrics for measuring robustness performance."""

  def __init__(self,
               label_column_name,
               protected_groups,
               subgroups,
               print_dir=None):
    """Initializes RobustFairnessMetrics class.

    Args:
      label_column_name: (string) name of the target variable.
      protected_groups: string list of protected features. For example,
        ["sex,"race"]. Currently, we assume protected features to take binary
        values 0 or 1. For example, Male vs Female, Black vs White.
      subgroups: int list enumerating subgroup in the dataset, given
      by the caretisian product of protected features. For example, if the
      dataset has two protected features ["race","sex"] that take values
      ["White", "Black", "Male", "Female"]. We call their catesian product as
      subgroups ["Black Male", Black Female", "White Male", "White Female"],
      which are enumerate as [0, 1, 2, 3].
      print_dir: (string) path to the directory in which we want to print
        tensorflow variables of interest. Currently, we save "example_weights",
        "label", "predictions", and protected groups. If set to None, no
        variables are saved to directory.
        This argument is passed to the "output_steam" option of tf.print".
    """
    self._label_column_name = label_column_name
    self._subgroups = subgroups
    self._protected_groups = protected_groups
    self._print_dir = print_dir

    self.metrics_fn = {
        "accuracy": tf.metrics.accuracy,
        "recall": tf.metrics.recall,
        "precision": tf.metrics.precision,
        "tp": tf.metrics.true_positives,
        "fp": tf.metrics.false_positives,
        "tn": tf.metrics.true_negatives,
        "fn": tf.metrics.false_negatives,
        "fpr": contrib_metrics.streaming_false_positive_rate,
        "fnr": contrib_metrics.streaming_false_negative_rate
    }

    self.metrics_fn_th = {
        "recall_th": tf.metrics.recall_at_thresholds,
        "precision_th": tf.metrics.precision_at_thresholds,
        "tp_th": tf.metrics.true_positives_at_thresholds,
        "fp_th": tf.metrics.false_positives_at_thresholds,
        "tn_th": tf.metrics.true_negatives_at_thresholds,
        "fn_th": tf.metrics.false_negatives_at_thresholds
    }

  def _get_control_dependencies_for_print(self, labels, predictions):
    """Returns a list of control dependencies for tf.print.

    Instantiates tf.print ops, and sets the output_stream option of tf.print to
    print tf variables to print_dir. We print following TensorFlow variables:
    ["example_weights","labels","predictions","subgroups",<protected_groups>]
    Output is a text file at path: "<print_dir>/<tf_variable_name>.csv".
    The number of lines is same as number of evaluation steps.
    Each line is of format: "<variable_name>,<tensor_of_batch_size>".
    For example, "example_weight,[1.0,1.0,....]"

    Args:
      labels: A `dict` of `Tensor` Objects. Expects to have a key/value pair
          for the strings in self._label_column_name, self._protected_groups,
          and "subgroups".
      predictions: A `dict` of `Tensor` Objects. Expects to have a
          key/value pair for keys in (self._label_column_name, "class_ids"),
          and "example_weights".

    Returns:
      control_dependencies_for_print: A list of tf.print control_dependency ops.
    """
    if self._print_dir:
      predictions_ids_print_op = tf.print(
          "predictions",
          predictions[(self._label_column_name, "class_ids")],
          summarize=-1,
          sep=",",
          output_stream="file:///{}/predictions.csv".format(
              self._print_dir))
      label_print_op = tf.print(
          "labels",
          tf.reshape(labels[self._label_column_name], [-1]),
          summarize=-1,
          sep=",",
          output_stream="file:///{}/labels.csv".format(self._print_dir))
      subgroup_print_op = tf.print(
          "subgroups",
          tf.reshape(labels["subgroup"], [-1]),
          summarize=-1,
          sep=",",
          output_stream="file:///{}/subgroups.csv".format(self._print_dir))
      example_weights_print_op = tf.print(
          "example_weights",
          predictions[("example_weights")],
          summarize=-1,
          sep=",",
          output_stream="file:///{}/example_weights.csv".format(
              self._print_dir))
      protected_col_0_print_op = tf.print(
          self._protected_groups[0],
          tf.reshape(labels[self._protected_groups[0]], [-1]),
          summarize=-1,
          sep=",",
          output_stream="file:///{}/{}.csv".format(self._print_dir,
                                                   self._protected_groups[0]))
      protected_col_1_print_op = tf.print(
          self._protected_groups[1],
          tf.reshape(labels[self._protected_groups[1]], [-1]),
          summarize=-1,
          sep=",",
          output_stream="file:///{}/{}.csv".format(self._print_dir,
                                                   self._protected_groups[1]))

      control_dependencies_for_print = [
          predictions_ids_print_op,
          label_print_op,
          subgroup_print_op,
          example_weights_print_op,
          protected_col_0_print_op,
          protected_col_1_print_op
      ]
    else:
      control_dependencies_for_print = None
    return control_dependencies_for_print

  def _get_protected_group_masks(self, labels):
    """Initialize tensors with (binary) masks for protected groups (e.g., sex and race) and the corresponding intersectional subgroups formed by sex times race.

    This method assumes protected groups to take binary values 0 or 1.
    For example, Male vs Female, Black vs White. Group_0 and Group_1 here refer
    to these binary values. The list of protected groups (e.g.,["sex", "race"])
    is configurable via self._protected_groups.

    Args:
     labels: A `dict` of `Tensor` Objects. Expects to have a key/value pair
          for the strings in self.target_variable_name, self.protected_groups,
          and "subgroups".

    Returns:
     masks_protected_group_0: A `dict` of `Tensor` Objects.
     masks_protected_group_1: A `dict` of `Tensor` Objects.
     masks_subgroup: A `dict` of `Tensor` Objects.
    """
    # Assumes (binary) protected features, which take values 0 or 1.
    masks_protected_group_0 = {}
    masks_protected_group_1 = {}
    for sensitive_group in self._protected_groups:
      masks_protected_group_0[sensitive_group] = tf.cast(
          tf.equal(labels[sensitive_group], 0), tf.float32)
      masks_protected_group_1[sensitive_group] = tf.cast(
          tf.equal(labels[sensitive_group], 1), tf.float32)

    masks_subgroup = {}
    for subgroup in self._subgroups:
      masks_subgroup[subgroup] = tf.cast(tf.equal(labels["subgroup"], subgroup),
                                         tf.float32)

    return masks_protected_group_0, masks_protected_group_1, masks_subgroup

  def create_fairness_metrics_fn(self, num_thresholds=10):
    """Creates a metric function for tf.core Estimators."""

    def fairness_metrics_fn(features,  # pylint: disable=unused-argument
                            labels,
                            predictions):
      """Creates a metric function for tf.core Estimators.

      The metric_fn can be used to add additional metrics to the estimator. For
      instance using tf.estimator.add_metrics().

      Args:
        features: a dict from feature name to feature tensors.
        labels: labels: A `dict` of `Tensor` Objects. Expects to have a
          key/value pair for the keys in self._label_column_name,
          self._protected_groups, and "subgroup".
        predictions: A `dict` of `Tensor` Objects. Expects to have a
          key/value pair for keys in (self._label_column_name, "class_ids"),
          (self._label_column_name, "logistic"), and "example_weights".

      Returns:
        A dictionary of computed eval metrics.


      Compute metrics follow this naming convention:
        -- metrics computed on the entire data:
            "label $METRIC"
        -- metrics computed for each sensitive group in the data:
            "$GROUP_NAME sensitive class $CLASS_NUMBER $METRIC"
        -- metrics computed at a number of thresholds (given N_THRESHOLD):
            "label $METRIC"
            "$GROUP_NAME sensitive class $CLASS_NUMBER $METRIC th"

        $METRIC takes value in [accuracy, recall, precision, -- perf metrics
                                    tp, tn, fp, fn, --confusion matrix
                                    fpr, -- false positive  rate
                                    tpr, -- true positive  rate
                                    tnr, -- true negative  rate
                                    ]
        $GROUP takes value in  [sex,race]
        $CLASS_NUMBER takes value in [0,1]
      """
      # If print_dir is not None, initializes tf.print for following tensors:
      # # example_weights, label, prediction, subgroup and protected_groups
      control_dependencies_for_print = self._get_control_dependencies_for_print(labels, predictions)  # pylint: disable=line-too-long

      with tf.control_dependencies(control_dependencies_for_print):
        masks_protected_group_0, masks_protected_group_1, masks_subgroup = self._get_protected_group_masks(labels)  # pylint: disable=line-too-long

        metrics = {}

        # Adds metrics that rely on predicted binary class id
        class_id_kwargs = {
            "labels": labels[self._label_column_name],
            "predictions": predictions[(self._label_column_name, "class_ids")]
        }
        for metric, metric_fn in self.metrics_fn.items():
          metrics["{}".format(metric)] = metric_fn(**class_id_kwargs)
          for group in self._protected_groups:
            metrics["{} {} group 0".format(metric, group)] = metric_fn(
                weights=masks_protected_group_0[group],
                **class_id_kwargs)
            metrics["{} {} group 1".format(metric, group)] = metric_fn(
                weights=masks_protected_group_1[group],
                **class_id_kwargs)
          for subgroup in self._subgroups:
            metrics["{} subgroup {}".format(metric, subgroup)] = metric_fn(
                weights=masks_subgroup[subgroup],
                **class_id_kwargs)

        # Adds metrics that rely on predicted probability
        logistics_kwargs = {
            "labels": labels[self._label_column_name],
            "predictions": predictions[(self._label_column_name, "logistic")]
        }
        metrics["auc"] = tf.metrics.auc(
            num_thresholds=num_thresholds, **logistics_kwargs)
        for group in self._protected_groups:
          metrics["{} {} group 0".format("auc", group)] = tf.metrics.auc(
              weights=masks_protected_group_0[group],
              num_thresholds=num_thresholds,
              **logistics_kwargs)
          metrics["{} {} group 1".format("auc", group)] = tf.metrics.auc(
              weights=masks_protected_group_1[group],
              num_thresholds=num_thresholds,
              **logistics_kwargs)
        for subgroup in self._subgroups:
          metrics["{} subgroup {}".format("auc", subgroup)] = tf.metrics.auc(
              weights=masks_subgroup[subgroup],
              num_thresholds=num_thresholds,
              **logistics_kwargs)

        # Adds metrics that are computed at set thresholds
        thresholds = list(
            [float(th) / float(num_thresholds) for th in range(num_thresholds)])
        for metric, metric_fn in self.metrics_fn_th.items():
          metrics["{}".format(metric)] = metric_fn(
              thresholds=thresholds, **logistics_kwargs)
          for group in self._protected_groups:
            metrics["{} {} group 0".format(metric, group)] = metric_fn(
                weights=masks_protected_group_0[group],
                thresholds=thresholds,
                **logistics_kwargs)
            metrics["{} {} group 1".format(metric, group)] = metric_fn(
                weights=masks_protected_group_1[group],
                thresholds=thresholds,
                **logistics_kwargs)
          for subgroup in self._subgroups:
            metrics["{} subgroup {}".format(metric, subgroup)] = metric_fn(
                weights=masks_subgroup[subgroup],
                thresholds=thresholds,
                **logistics_kwargs)

      return metrics

    return fairness_metrics_fn
