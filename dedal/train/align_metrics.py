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

"""Custom metrics for sequence alignment.

This module defines the following types, which serve as inputs to all metrics
implemented here:

+ GroundTruthAlignment is A tf.Tensor<int>[batch, 3, align_len] that can be
  written as tf.stack([pos_x, pos_y, enc_trans], 1) such that
    (pos_x[b][i], pos_y[b][i], enc_trans[b][i]) represents the i-th transition
  in the ground-truth alignment for example b in the minibatch.
  Both pos_x and pos_y are assumed to use one-based indexing and enc_trans
  follows the (categorical) 9-state encoding of edge types used throughout
  `learning/brain/research/combini/diff_opt/alignment/tf_ops.py`.

+ SWParams is a tuple (sim_mat, gap_open, gap_extend) parameterizing the
  Smith-Waterman LP such that
  + sim_mat is a tf.Tensor<float>[batch, len1, len2] (len1 <= len2) with the
    substitution values for pairs of sequences.
  + gap_open is a tf.Tensor<float>[batch, len1, len2] (len1 <= len2) or
    tf.Tensor<float>[batch] with the penalties for opening a gap. Must agree
    in rank with gap_extend.
  + gap_extend is a tf.Tensor<float>[batch, len1, len2] (len1 <= len2) or
    tf.Tensor<float>[batch] with the penalties for extending a gap. Must agree
    in rank with gap_open.

+ AlignmentOutput is a tuple (solution_values, solution_paths, sw_params) such
  that
  + 'solution_values' contains a tf.Tensor<float>[batch] with the (soft) optimal
    Smith-Waterman scores for the batch.
  + 'solution_paths' contains a tf.Tensor<float>[batch, len1, len2, 9] that
    describes the optimal soft alignments.
  + 'sw_params' is a SWParams tuple as described above.
"""


import functools
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

import gin
import tensorflow as tf

from dedal import alignment


GroundTruthAlignment = tf.Tensor
PredictedPaths = tf.Tensor
SWParams = Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]
AlignmentOutput = Tuple[tf.Tensor, Optional[PredictedPaths], SWParams]
NaiveAlignmentOutput = Tuple[tf.Tensor, tf.Tensor, SWParams]


def _confusion_matrix(
    alignments_true,
    sol_paths_pred):
  """Computes true, predicted and actual positives for a batch of alignments."""
  batch_size = tf.shape(alignments_true)[0]

  # Computes the number of true positives per example as an (sparse) inner
  # product of two binary tensors of shape (batch_size, len_x, len_y) via
  # indexing. Entirely avoids materializing one of the two tensors explicitly.
  match_indices_true = alignment.alignments_to_state_indices(
      alignments_true, 'match')  # [n_aligned_chars_true, 3]
  match_indicators_pred = alignment.paths_to_state_indicators(
      sol_paths_pred, 'match')  # [batch, len_x, len_y]
  batch_indicators = match_indices_true[:, 0]  # [n_aligned_chars_true]
  matches_flat = tf.gather_nd(
      match_indicators_pred, match_indices_true)  # [n_aligned_chars_true]
  true_positives = tf.math.unsorted_segment_sum(
      matches_flat, batch_indicators, batch_size)  # [batch]

  # Compute number of predicted and ground-truth positives per example.
  pred_positives = tf.reduce_sum(match_indicators_pred, axis=[1, 2])
  # Note(fllinares): tf.math.bincount unsupported in TPU :(
  cond_positives = tf.math.unsorted_segment_sum(
      tf.ones_like(batch_indicators, tf.float32),
      batch_indicators,
      batch_size)  # [batch]
  return true_positives, pred_positives, cond_positives


@gin.configurable
class AlignmentPrecisionRecall(tf.metrics.Metric):
  """Implements precision and recall metrics for sequence alignment."""

  def __init__(self,
               name = 'alignment_pr',
               threshold = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._threshold = threshold
    self._true_positives = tf.metrics.Mean()  # TP
    self._pred_positives = tf.metrics.Mean()  # TP + FP
    self._cond_positives = tf.metrics.Mean()  # TP + FN

  def update_state(
      self,
      alignments_true,
      alignments_pred,
      sample_weight = None):
    """Updates TP, TP + FP and TP + FN for a batch of true, pred alignments."""
    if alignments_pred[1] is None:
      return

    sol_paths_pred = alignments_pred[1]
    if self._threshold is not None:  # Otherwise, we assume already binarized.
      sol_paths_pred = tf.cast(sol_paths_pred >= self._threshold, tf.float32)

    true_positives, pred_positives, cond_positives = _confusion_matrix(
        alignments_true, sol_paths_pred)

    self._true_positives.update_state(true_positives, sample_weight)
    self._pred_positives.update_state(pred_positives, sample_weight)
    self._cond_positives.update_state(cond_positives, sample_weight)

  def result(self):
    true_positives = self._true_positives.result()
    pred_positives = self._pred_positives.result()
    cond_positives = self._cond_positives.result()
    precision = tf.where(
        true_positives > 0.0, true_positives / pred_positives, 0.0)
    recall = tf.where(
        true_positives > 0.0, true_positives / cond_positives, 0.0)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return {
        f'{self.name}/precision': precision,
        f'{self.name}/recall': recall,
        f'{self.name}/f1': f1,
    }

  def reset_states(self):
    self._true_positives.reset_states()
    self._pred_positives.reset_states()
    self._cond_positives.reset_states()


@gin.configurable
class NaiveAlignmentPrecisionRecall(tf.metrics.Metric):
  """Implements precision and recall metrics for (naive) sequence alignment."""

  def __init__(self,
               name = 'naive_alignment_pr',
               threshold = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._precision = tf.metrics.Precision(thresholds=threshold)
    self._recall = tf.metrics.Recall(thresholds=threshold)

  def update_state(
      self,
      alignments_true,
      alignments_pred,
      sample_weight = None):
    """Updates precision, recall for a batch of true, pred alignments."""
    if alignments_pred[1] is None:
      return

    _, match_indicators_pred, sw_params = alignments_pred
    sim_mat, _, _ = sw_params
    shape, dtype = sim_mat.shape, match_indicators_pred.dtype

    match_indices_true = alignment.alignments_to_state_indices(
        alignments_true, 'match')
    updates_true = tf.ones([tf.shape(match_indices_true)[0]], dtype=dtype)
    match_indicators_true = tf.scatter_nd(
        match_indices_true, updates_true, shape=shape)

    batch = tf.shape(sample_weight)[0]
    sample_weight = tf.reshape(sample_weight, [batch, 1, 1])
    mask = alignment.mask_from_similarities(sim_mat, dtype=dtype)

    self._precision.update_state(
        match_indicators_true, match_indicators_pred, sample_weight * mask)
    self._recall.update_state(
        match_indicators_true, match_indicators_pred, sample_weight * mask)

  def result(self):
    precision, recall = self._precision.result(), self._recall.result()
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return {
        f'{self.name}/precision': precision,
        f'{self.name}/recall': recall,
        f'{self.name}/f1': f1,
    }

  def reset_states(self):
    self._precision.reset_states()
    self._recall.reset_states()


@gin.configurable
class AlignmentMSE(tf.metrics.Mean):
  """Implements mean squared error metric for sequence alignment."""

  def __init__(self, name = 'alignment_mse', **kwargs):
    super().__init__(name=name, **kwargs)

  def update_state(
      self,
      alignments_true,
      alignments_pred,
      sample_weight = None):
    """Updates mean squared error for a batch of true vs pred alignments."""
    if alignments_pred[1] is None:
      return

    sol_paths_pred = alignments_pred[1]
    len_x, len_y = tf.shape(sol_paths_pred)[1], tf.shape(sol_paths_pred)[2]
    sol_paths_true = alignment.alignments_to_paths(
        alignments_true, len_x, len_y)
    mse = tf.reduce_sum((sol_paths_pred - sol_paths_true) ** 2, axis=[1, 2, 3])
    super().update_state(mse, sample_weight)


@gin.configurable
class MeanList(tf.metrics.Metric):
  """Means over ground-truth and predictions for positive and negative pairs."""

  def __init__(self,
               positive_keys=('true', 'pred_pos'),
               negative_keys=('pred_neg',),
               **kwargs):
    super().__init__(**kwargs)
    self._keys = positive_keys + negative_keys
    self._process_negatives = bool(len(negative_keys))
    self._means = {}

  def _split(self,
             inputs,
             return_neg = True):
    if not self._process_negatives:
      return (inputs,)
    pos = tf.nest.map_structure(lambda t: t[:tf.shape(t)[0] // 2], inputs)
    if return_neg:
      neg = tf.nest.map_structure(lambda t: t[tf.shape(t)[0] // 2:], inputs)
    return (pos, neg) if return_neg else (pos,)

  def result(self):
    return {f'{self.name}/{k}': m.result() for k, m in self._means.items()}

  def reset_states(self):
    for mean in self._means.values():
      mean.reset_states()


@gin.configurable
class AlignmentStats(MeanList):
  """Tracks alignment length, number of matches and number of gaps."""
  STATS = ('length', 'n_match', 'n_gap')

  def __init__(self,
               name = 'alignment_stats',
               process_negatives = True,
               **kwargs):
    negative_keys = ('pred_neg',) if process_negatives else ()
    super().__init__(name=name, negative_keys=negative_keys, **kwargs)
    for stat in self.STATS:
      self._means.update({f'{stat}/{k}': tf.metrics.Mean() for k in self._keys})
    self._stat_fn = {
        'length': alignment.length,
        'n_match': functools.partial(alignment.state_count, states='match'),
        'n_gap': functools.partial(alignment.state_count, states='gap_open'),
    }

  def update_state(
      self,
      alignments_true,
      alignments_pred,
      sample_weight = None):
    """Updates alignment stats for a batch of true and predicted alignments."""
    del sample_weight  # Logic in this metric controlled by process_negatives.
    if alignments_pred[1] is None:
      return

    vals = self._split(alignments_true, False) + self._split(alignments_pred[1])
    for stat in self.STATS:
      for k, tensor in zip(self._keys, vals):
        self._means[f'{stat}/{k}'].update_state(self._stat_fn[stat](tensor))


@gin.configurable
class AlignmentScore(MeanList):
  """Tracks alignment score / solution value."""

  def __init__(self,
               name = 'alignment_score',
               process_negatives = True,
               **kwargs):
    negative_keys = ('pred_neg',) if process_negatives else ()
    super().__init__(name=name, negative_keys=negative_keys, **kwargs)
    self._means.update({k: tf.metrics.Mean() for k in self._keys})

  def update_state(
      self,
      alignments_true,
      alignments_pred,
      sample_weight = None):
    """Updates alignment scores for a batch of true and predicted alignments."""
    del sample_weight  # Logic in this metric controlled by process_negatives.

    vals_true = (self._split(alignments_pred[2], False) +
                 self._split(alignments_true, False))
    self._means[self._keys[0]].update_state(alignment.sw_score(*vals_true))

    vals_pred = self._split(alignments_pred[0])
    for k, tensor in zip(self._keys[1:], vals_pred):
      self._means[k].update_state(tensor)


@gin.configurable
class SWParamsStats(MeanList):
  """Tracks Smith-Waterman substitution costs and gap penalties."""
  PARAMS = ('sim_mat', 'gap_open', 'gap_extend')

  def __init__(self,
               name = 'sw_params_stats',
               process_negatives = True,
               **kwargs):
    positive_keys = ('pred_pos',)
    negative_keys = ('pred_neg',) if process_negatives else ()
    super().__init__(name=name,
                     positive_keys=positive_keys,
                     negative_keys=negative_keys,
                     **kwargs)
    for p in self.PARAMS:
      self._means.update({f'{p}/{k}': tf.metrics.Mean() for k in self._keys})

  def update_state(
      self,
      alignments_true,
      alignments_pred,
      sample_weight = None):
    """Updates SW param stats for a batch of true and predicted alignments."""
    del alignments_true  # Present for compatibility with SeqAlign.
    del sample_weight  # Logic in this metric controlled by process_negatives.

    vals = self._split(alignments_pred[2])
    for k, sw_params in zip(self._keys, vals):
      for p, t in zip(self.PARAMS, sw_params):
        # Prevents entries corresponding to padding from being tracked.
        mask = alignment.mask_from_similarities(t)
        self._means[f'{p}/{k}'].update_state(t, sample_weight=mask)


@gin.configurable
class StratifyByPID(tf.metrics.Metric):
  """Wraps Keras metric, accounting only for examples in given PID bins."""

  def __init__(self,
               metric_cls,
               lower = None,
               upper = None,
               step = None,
               pid_definition = '3',
               **kwargs):
    self._lower = lower if lower is not None else 0.0
    if isinstance(step, Sequence):
      self._upper = self._lower + sum(step)  # Ignores arg. Not used, remove?
      self._steps = step
    else:
      self._upper = upper if upper is not None else 1.0
      step = step if step is not None else self._upper - self._lower
      self._steps = (step,)

    self._stratified_metrics = []
    lower = self._lower
    for step in self._steps:
      upper = lower + step
      self._stratified_metrics.append((metric_cls(), lower, upper))
      lower = upper

    self._pid_definition = pid_definition

    stratify_by_pid_str = f'stratify_by_pid{self._pid_definition}'
    super().__init__(
        name=f'{stratify_by_pid_str}/{self._stratified_metrics[0][0].name}',
        **kwargs)

  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight,
                   metadata):
    pid = metadata[0]
    no_pid_info = pid == -1
    for metric, lower, upper in self._stratified_metrics:
      in_bin = tf.logical_and(
          tf.logical_or(pid == self._lower, pid > lower), pid <= upper)
      keep_mask = tf.logical_or(in_bin, no_pid_info)
      metric.update_state(
          y_true, y_pred, sample_weight=tf.where(keep_mask, sample_weight, 0.0))

  def result(self):
    res = {}
    for metric, lower, upper in self._stratified_metrics:
      res_i = metric.result()
      suffix = f'PID{self._pid_definition}:{lower:.2f}-{upper:.2f}'
      if isinstance(res_i, Mapping):
        res.update({f'{k}/{suffix}': v for k, v in res_i.items()})
      else:
        res[f'{self._stratified_metrics[0][0].name}/{suffix}'] = res_i
    return res

  def reset_states(self):
    for metric, _, _ in self._stratified_metrics:
      metric.reset_states()
