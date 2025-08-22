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

"""Custom metrics."""

import json
from typing import Mapping, Optional, Sequence

import gin
import tensorflow as tf


@gin.configurable
class PearsonCorrelation(tf.metrics.Metric):
  """Implements Pearson correlation as tf.metrics.Metric class."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._y_true_mean = tf.metrics.Mean()
    self._y_pred_mean = tf.metrics.Mean()
    self._y_true_sq_mean = tf.metrics.Mean()
    self._y_pred_sq_mean = tf.metrics.Mean()
    self._y_true_dot_y_pred_mean = tf.metrics.Mean()

  def update_state(self, y_true, y_pred, sample_weight=None):
    self._y_true_mean.update_state(y_true, sample_weight)
    self._y_pred_mean.update_state(y_pred, sample_weight)
    self._y_true_sq_mean.update_state(y_true ** 2, sample_weight)
    self._y_pred_sq_mean.update_state(y_pred ** 2, sample_weight)
    self._y_true_dot_y_pred_mean.update_state(y_true * y_pred, sample_weight)

  def result(self):
    y_true_var = self._y_true_sq_mean.result() - self._y_true_mean.result() ** 2
    y_pred_var = self._y_pred_sq_mean.result() - self._y_pred_mean.result() ** 2
    cov = (self._y_true_dot_y_pred_mean.result()
           - self._y_true_mean.result() * self._y_pred_mean.result())
    return cov / tf.sqrt(y_true_var) / tf.sqrt(y_pred_var)

  def reset_states(self):
    self._y_true_mean.reset_states()
    self._y_pred_mean.reset_states()
    self._y_true_sq_mean.reset_states()
    self._y_pred_sq_mean.reset_states()
    self._y_true_dot_y_pred_mean.reset_states()


@gin.configurable
class Perplexity(tf.metrics.SparseCategoricalCrossentropy):
  """Implements perplexity as tf.metrics.Metric class."""

  def __init__(self, from_logits=True, name='perplexity', **kwargs):
    super().__init__(from_logits=from_logits, name=name, **kwargs)

  def result(self):
    return tf.exp(super().result())


class DoubleMean(tf.keras.metrics.Metric):
  """The means of predictions and ground truth for a given metrics."""

  def __init__(self, mean_metric_cls, **kwargs):
    self._predicted = mean_metric_cls()
    self._expected = mean_metric_cls()
    super().__init__(name=self._expected.name)

  def update_state(self, y_true, y_pred, sample_weight=None):
    self._predicted.update_state(y_pred, sample_weight)
    self._expected.update_state(y_true, sample_weight)

  def reset_states(self):
    self._predicted.reset_states()
    self._expected.reset_states()

  def result(self):
    return {
        f'{self.name}/true': self._expected.result(),
        f'{self.name}/pred': self._predicted.result()
    }


@gin.configurable
class SparseLiftedClanAccuracy(tf.metrics.Accuracy):
  """Evaluates SparseCategoricalAccuracy at the lifted clan level."""

  def __init__(
      self, filename, name = 'lifted_clan_accuracy', **kwargs):
    super().__init__(name=name, **kwargs)
    # Precomputes a 1D Tensor cla_from_fam such that cla_from_fam[fam_key]
    # contains the label cla_key of the clan to which the family indexed by
    # fam_key belongs.
    self._filename = filename  # A json file.
    cla_key_from_fam_key = self._load_mapping()
    keys = list(cla_key_from_fam_key.keys())
    values = list(cla_key_from_fam_key.values())
    indices = sorted(range(len(keys)), key=lambda i: keys[i])
    self._cla_from_fam = tf.convert_to_tensor(
        [values[i] for i in indices], tf.int64)

  def _load_mapping(self):
    """Prepares family to clan key mapping from JSON file."""
    with tf.io.gfile.GFile(self._filename, 'r') as f:
      cla_id_from_fam_id = json.load(f)
    # "Translates" the mapping between IDs to a mapping between integer keys.
    idx_from_fam, idx_from_cla = {}, {}
    for fam, cla in cla_id_from_fam_id.items():
      if fam not in idx_from_fam:
        idx_from_fam[fam] = len(idx_from_fam)
      if cla not in idx_from_cla:
        idx_from_cla[cla] = len(idx_from_cla)
    return {idx_from_fam[k]: idx_from_cla[v]
            for k, v in cla_id_from_fam_id.items()}

  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight = None,
                   metadata = ()):
    # Ignores family labels, assumes metadata always contains clan labels.
    y_true = metadata[0]
    # Computes predicted family labels from probabilities / logits. Then, maps
    # these to clan labels.
    y_pred = tf.gather(self._cla_from_fam, tf.math.argmax(y_pred, axis=-1))
    super().update_state(y_true, y_pred, sample_weight=sample_weight)


@gin.configurable
class ContactPrecisionRecallFixedK(tf.metrics.Metric):
  """Implements basic PR metrics for residue-residue contact prediction."""

  def __init__(self,
               name = 'contact_pr',
               range_low = 12,
               range_high = 23,
               at_k = 50,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._range_low = range_low
    self._range_high = range_high
    self._at_k = at_k  # TODO(qberthet): allow at_k to be list to clean up gin.

    self._precision = tf.metrics.Mean()
    self._recall = tf.metrics.Mean()
    self._f1score = tf.metrics.Mean()
    self._auprc = tf.metrics.AUC(
        num_thresholds=1000, curve='PR', from_logits=False)

  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight):
    batch = tf.shape(y_pred)[0]
    num_embs = tf.shape(y_pred)[1]
    proba_pred = tf.nn.sigmoid(y_pred)
    if sample_weight is not None:
      proba_pred *= sample_weight[Ellipsis, None]

    weights_range = tf.range(num_embs, dtype=tf.float32)
    weights_range_square = tf.abs(weights_range[:, tf.newaxis] -
                                  weights_range[tf.newaxis, :])
    indic_range_fun = lambda x, a, b: tf.logical_and(x >= a, x <= b)
    weights_square = indic_range_fun(weights_range_square,
                                     self._range_low,
                                     self._range_high)
    weights_square = tf.cast(weights_square[None, Ellipsis, None], dtype=tf.float32)

    proba_pred_filter = proba_pred * weights_square
    flat_proba_pred_filter = tf.reshape(proba_pred_filter, (batch, -1))
    y_true_filter = y_true * weights_square
    flat_y_true_filter = tf.reshape(y_true_filter, (batch, -1))

    _, indices = tf.math.top_k(flat_proba_pred_filter, k=self._at_k)
    flat_y_pred_filter = tf.cast(flat_proba_pred_filter > 0.5, tf.float32)

    true_in_top = tf.gather(flat_y_true_filter, indices, batch_dims=-1)
    pred_in_top = tf.gather(flat_y_pred_filter, indices, batch_dims=-1)
    true_pred_in_top = tf.gather(
        flat_y_true_filter * flat_y_pred_filter, indices, batch_dims=-1)

    number_true = tf.maximum(tf.reduce_sum(true_in_top, axis=-1), 1e-6)
    number_preds = tf.maximum(tf.reduce_sum(pred_in_top, axis=-1), 1e-6)
    number_true_preds = tf.reduce_sum(true_pred_in_top, axis=-1)

    precision = tf.maximum(number_true_preds / number_preds, 1e-6)
    recall = tf.maximum(number_true_preds / number_true, 1e-6)

    self._precision.update_state(precision)
    self._recall.update_state(recall)
    self._f1score.update_state(
        2 * (precision * recall) / (precision + recall))

    # TODO(qberthet): double-check.
    auprc_weight = sample_weight * weights_square[Ellipsis, 0]
    self._auprc.update_state(
        y_true_filter, proba_pred_filter, sample_weight=auprc_weight)

  def result(self):
    return {
        f'{self.name}/precision': self._precision.result(),
        f'{self.name}/recall': self._recall.result(),
        f'{self.name}/f1': self._f1score.result(),
        f'{self.name}/auprc': self._auprc.result(),
    }

  def reset_states(self):
    self._precision.reset_states()
    self._recall.reset_states()
    self._f1score.reset_states()
    self._auprc.reset_states()
