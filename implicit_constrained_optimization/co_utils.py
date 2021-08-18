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

r"""Utils for constrained optimization."""

import tensorflow as tf


def fpr_func(threshold, preds, labels):
  """False acceptance rate or False positive rate."""
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds, threshold), tf.float32),
          1 - labels)) / tf.reduce_sum(1 - labels)


def fpr_func_multi(threshold, preds, labels):
  """False acceptance rate or False positive rate."""
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds, threshold), tf.float32), 1 - labels),
      axis=0) / tf.reduce_sum(
          1 - labels, axis=0)


def fpr_func_multi_th(threshold, preds, labels):
  """False acceptance rate or False positive rate."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds_exp, threshold_exp), tf.float32),
          1 - labels_exp),
      axis=0) / tf.reduce_sum(
          1 - labels_exp, axis=0)


def tpr_func_multi(threshold, preds, labels):
  """True positives."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds_exp, threshold_exp), tf.float32),
          labels_exp),
      axis=0) / tf.reduce_sum(
          labels_exp, axis=0)


def fnr_func(threshold, preds, labels):
  """False rejection rate or False negative rate."""
  return tf.reduce_sum(
      tf.multiply(tf.cast(tf.less_equal(preds, threshold), tf.float32),
                  labels)) / tf.reduce_sum(labels)


def fnr_func_multi(threshold, preds, labels):
  """False rejection rate or False negative rate."""
  return tf.reduce_sum(
      tf.multiply(tf.cast(tf.less_equal(preds, threshold), tf.float32), labels),
      axis=0) / tf.reduce_sum(
          labels, axis=0)


def fpr_sigmoid_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False acceptance rate using Sigmoid."""
  return tf.reduce_sum(
      tf.multiply(tf.sigmoid(temperature * (preds - threshold)),
                  (1 - labels))) / tf.reduce_sum(1 - labels)


def fpr_sigmoid_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of False acceptance rate using Sigmoid."""
  fpr_per_label = tf.reduce_sum(
      tf.multiply(tf.sigmoid(temperature * (preds - threshold)), (1 - labels)),
      axis=0) / tf.reduce_sum(
          1 - labels, axis=0)
  return fpr_per_label


def fnr_sigmoid_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False rejection rate using Sigmoid."""
  return tf.reduce_sum(
      tf.multiply(tf.sigmoid(-1 * temperature * (preds - threshold)),
                  labels)) / tf.reduce_sum(labels)


def fnr_sigmoid_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of False negative rate using Sigmoid."""
  fnr_per_label = tf.reduce_sum(
      tf.multiply(tf.sigmoid(-1 * temperature * (preds - threshold)), labels),
      axis=0) / tf.reduce_sum(
          labels, axis=0)
  return fnr_per_label


def fnr_sigmoid_proxy_func_multi_th(threshold, preds, labels, temperature=1.):
  """Approximation of FNR using Sigmoid."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_negatives = tf.sigmoid(
      -1 * temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  fnr_per_label = tf.reduce_sum(
      tf.multiply(pred_negatives, labels_exp), axis=0) / tf.reduce_sum(
          labels_exp, axis=0)  # classes x thresholds_per_class
  return fnr_per_label


def fp_sigmoid_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False acceptance rate using Sigmoid."""
  return tf.reduce_sum(
      tf.multiply(tf.sigmoid(temperature * (preds - threshold)), (1 - labels)))


def fn_sigmoid_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False rejection rate using Sigmoid."""
  return tf.reduce_sum(
      tf.multiply(tf.sigmoid(-1. * temperature * (preds - threshold)), labels))


def fpr_softplus_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False acceptance rate using Softplus."""
  return tf.reduce_sum(
      tf.multiply(
          # tf.log(1 + tf.exp(temperature * (preds - threshold))),
          tf.math.softplus(temperature * (preds - threshold)),
          (1 - labels))) / tf.reduce_sum(1 - labels)


def fpr_softplus_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of False acceptance rate using Softplus."""
  fpr_per_label = tf.reduce_sum(
      tf.multiply(
          # tf.log(1 + tf.exp(temperature * (preds - threshold))),
          tf.math.softplus(temperature * (preds - threshold)),
          (1 - labels)),
      axis=0) / tf.reduce_sum(
          1 - labels, axis=0)
  return fpr_per_label


def fnr_softplus_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False rejection rate using Softplus."""
  return tf.reduce_sum(
      tf.multiply(
          # tf.log(1 + tf.exp(-1 * temperature * (preds - threshold))),
          tf.math.softplus(-1 * temperature * (preds - threshold)),
          labels)) / tf.reduce_sum(labels)


def fnr_softplus_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of False rejection rate using Softplus."""
  fnr_per_label = tf.reduce_sum(
      tf.multiply(
          tf.math.softplus(-1 * temperature * (preds - threshold)), labels),
      axis=0) / tf.reduce_sum(
          labels, axis=0)
  return fnr_per_label


def fnr_softplus_proxy_func_multi_th(threshold, preds, labels, temperature=1.):
  """Approximation of FNR using Softplus."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_negatives = tf.math.softplus(
      -1 * temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  fnr_per_label = tf.reduce_sum(
      tf.multiply(pred_negatives, labels_exp), axis=0) / tf.reduce_sum(
          labels_exp, axis=0)  # classes x thresholds_per_class
  return fnr_per_label


def tpr_softplus_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of Recall using Softplus."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.math.softplus(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  recall_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, labels_exp), axis=0) / tf.reduce_sum(
          labels_exp, axis=0)  # classes x thresholds_per_class
  return recall_per_label


def tpr_sigmoid_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of Recall using Sigmoid."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.sigmoid(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  recall_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, labels_exp), axis=0) / tf.reduce_sum(
          labels_exp, axis=0)  # classes x thresholds_per_class
  return recall_per_label


def fpr_softplus_proxy_func_multi_th(threshold, preds, labels, temperature=1.):
  """Approximation of FPR using Softplus."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.math.softplus(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  fpr_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, 1 - labels_exp), axis=0) / tf.reduce_sum(
          1 - labels_exp, axis=0)  # classes x thresholds_per_class
  return fpr_per_label


def fpr_sigmoid_proxy_func_multi_th(threshold, preds, labels, temperature=1.):
  """Approximation of FPR using Sigmoid."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.sigmoid(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  fpr_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, 1 - labels_exp), axis=0) / tf.reduce_sum(
          1 - labels_exp, axis=0)  # classes x thresholds_per_class
  return fpr_per_label


def tp_softplus_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of true positives using Softplus."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.math.softplus(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  tp_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, labels_exp),
      axis=0)  # classes x thresholds_per_class
  return tp_per_label


def tp_sigmoid_proxy_func_multi(threshold, preds, labels, temperature=1.):
  """Approximation of true positives using Sigmoid."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.sigmoid(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  tp_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, labels_exp),
      axis=0)  # classes x thresholds_per_class
  return tp_per_label


def fp_softplus_proxy_func_multi_th(threshold, preds, labels, temperature=1.):
  """Approximation of false positives using Softplus."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.math.softplus(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  fp_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, 1 - labels_exp),
      axis=0)  # classes x thresholds_per_class
  return fp_per_label


def fp_sigmoid_proxy_func_multi_th(threshold, preds, labels, temperature=1.):
  """Approximation of false positives using Sigmoid."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  pred_positives = tf.sigmoid(
      temperature *
      (preds_exp - threshold_exp))  # batchsize x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  fp_per_label = tf.reduce_sum(
      tf.multiply(pred_positives, 1 - labels_exp),
      axis=0)  # classes x thresholds_per_class
  return fp_per_label


def fp_softplus_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False positives using Softplus."""
  return tf.reduce_sum(
      tf.multiply(
          # tf.log(1 + tf.exp(temperature * (preds - threshold))),
          tf.math.softplus(temperature * (preds - threshold)),
          (1 - labels)))


def fn_softplus_proxy_func(threshold, preds, labels, temperature=1.):
  """Approximation of False negatives using Softplus."""
  return tf.reduce_sum(
      tf.multiply(
          # tf.log(1 + tf.exp(-1. * temperature * (preds - threshold))),
          tf.math.softplus(-1 * temperature * (preds - threshold)),
          labels))


def fp_func(threshold, preds, labels):
  """False accepts or False positives."""
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds, threshold), tf.float32), 1 - labels))


def tp_func_multi(threshold, preds, labels):
  """True positives."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds_exp, threshold_exp), tf.float32),
          labels_exp),
      axis=0)


def fp_func_multi(threshold, preds, labels):
  """False accepts or False positives."""
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds, threshold), tf.float32), 1 - labels),
      axis=0)


def fp_func_multi_th(threshold, preds, labels):
  """False accepts or False positives."""
  preds_exp = tf.expand_dims(preds, axis=-1)  # batchsize x classes x 1
  threshold_exp = tf.expand_dims(
      threshold, axis=0)  # 1 x classes x thresholds_per_class
  labels_exp = tf.expand_dims(labels, axis=-1)  # batchsize x classes x 1
  return tf.reduce_sum(
      tf.multiply(
          tf.cast(tf.greater(preds_exp, threshold_exp), tf.float32),
          1 - labels_exp),
      axis=0)


def fn_func(threshold, preds, labels):
  """False rejects or False negatives."""
  return tf.reduce_sum(
      tf.multiply(tf.cast(tf.less_equal(preds, threshold), tf.float32), labels))
