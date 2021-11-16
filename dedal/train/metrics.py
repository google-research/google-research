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
