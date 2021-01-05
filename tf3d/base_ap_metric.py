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

"""General average precision metric.

The update_state function is not implemented, and needs to be implemented in the
child class.
"""
import tensorflow as tf

from object_detection.utils import label_map_util


class BaseAPMetric(tf.keras.metrics.Metric):
  """General average precision metric."""

  def __init__(self,
               iou_threshold=0.5,
               num_classes=None,
               label_map=None,
               label_map_path=None,
               eval_prefix='eval',
               name='ap_metric'):
    """Base AP metric class.

    Args:
      iou_threshold: Intersection-over-union threshold at which the average
        precision is computed.
      num_classes: Number of classes.
      label_map: A dictionary mapping label ids to label names.
      label_map_path: path to labelmap (could be None).
      eval_prefix: Prefix for eval name; separates scalar values in Tensorboard.
      name: class name.
    """
    super(BaseAPMetric, self).__init__(name=name)
    self.iou_threshold = iou_threshold
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
    self.tp = {}
    self.fp = {}
    self.scores = {}
    self.num_gt = {}
    for c in self.class_range:
      self.tp[c] = tf.constant([], dtype=tf.int32)
      self.scores[c] = tf.constant([], dtype=tf.float32)
      self.num_gt[c] = tf.constant(0, dtype=tf.int32)

  def result(self):
    metrics_dict = self.get_metric_dictionary()
    return metrics_dict[self.eval_prefix +
                        '_avg/mean_AP_IOU{}'.format(self.iou_threshold)]

  def get_metric_dictionary(self):
    metrics_dict = {}
    ap_list = []
    for c in self.class_range:
      num_detections_c = tf.shape(self.scores[c])[0]
      _, sorted_detection_indices = tf.math.top_k(
          self.scores[c], k=num_detections_c)
      tp_c = tf.cast(
          tf.gather(self.tp[c], sorted_detection_indices), dtype=tf.float32)
      fp_c = 1.0 - tp_c
      num_gt_c = tf.cast(self.num_gt[c], dtype=tf.float32)
      if num_gt_c > 0 and num_detections_c > 0:
        cumsum_tp_c = tf.math.cumsum(tp_c, axis=0)
        cumsum_fp_c = tf.math.cumsum(fp_c, axis=0)
        precision = cumsum_tp_c / (cumsum_tp_c + cumsum_fp_c)
        recall = cumsum_tp_c / num_gt_c
        recall_mask = tf.math.logical_and(
            tf.greater_equal(recall, 0.0), tf.less_equal(recall, 1.0))
        precision = tf.boolean_mask(precision, recall_mask)
        recall = tf.boolean_mask(recall, recall_mask)
        recall = tf.concat([
            tf.constant([0], dtype=tf.float32),
            recall,
            tf.constant([1], dtype=tf.float32)
        ],
                           axis=0)
        precision = tf.concat([
            tf.constant([0], dtype=tf.float32),
            precision,
            tf.constant([0], dtype=tf.float32)
        ],
                              axis=0)
        num_values = tf.shape(precision)[0]
        for i in tf.range(num_values - 2, -1, -1):
          precision = tf.maximum(
              precision,
              tf.one_hot(i, depth=num_values, dtype=tf.float32) *
              precision[i + 1])
        ap_c = tf.reduce_sum((recall[1:] - recall[:-1]) * precision[:-1])
      else:
        ap_c = tf.constant(0.0, dtype=tf.float32)
      class_name = _get_class_name(class_id=c, label_map=self.label_map)
      metrics_dict[
          self.eval_prefix +
          '_IOU{}_AP/{}'.format(self.iou_threshold, class_name)] = ap_c
      ap_list.append(ap_c)
    mean_ap = _non_nan_mean(ap_list)
    metrics_dict[
        self.eval_prefix +
        '_avg/mean_AP_IOU{}'.format(self.iou_threshold)] = mean_ap
    return metrics_dict

  def reset_states(self):
    self.tp = {}
    self.fp = {}
    self.scores = {}
    self.num_gt = {}
    for c in self.class_range:
      self.tp[c] = tf.constant([], dtype=tf.int32)
      self.scores[c] = tf.constant([], dtype=tf.float32)
      self.num_gt[c] = tf.constant(0, dtype=tf.int32)


def _get_class_name(class_id, label_map):
  """Gets class name from label dictionary."""
  if label_map and class_id in label_map:
    return label_map[class_id]
  else:
    return str(class_id)


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


def _non_nan_mean(tensor_list):
  """Calculates the mean of a list of tensors while ignoring nans."""
  tensor = tf.stack(tensor_list)
  not_nan = tf.logical_not(tf.math.is_nan(tensor))
  return tf.reduce_mean(tf.boolean_mask(tensor, not_nan))
