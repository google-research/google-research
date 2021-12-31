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

"""Implements custom losses."""

from typing import NamedTuple, Optional, Tuple, Union

import gin
import tensorflow as tf

from dedal import alignment
from dedal import multi_task
from dedal import pairs


@gin.configurable
class WeightedLoss(NamedTuple):
  weight: float
  loss: tf.keras.losses.Loss


MaybeWeightedLoss = Union[WeightedLoss, tf.keras.losses.Loss]
NestedWeights = multi_task.Backbone[Optional[tf.Tensor]]
SWParams = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
AlignmentOutput = Tuple[tf.Tensor,  # Solution values.
                        Optional[tf.Tensor],  # Solution paths.
                        SWParams,  # DP parameters.
                       ]
NaiveAlignmentOutput = Tuple[tf.Tensor, tf.Tensor, SWParams]


@gin.configurable
class SmithWatermanLoss(tf.losses.Loss):
  """Implements a loss for differentiable local sequence alignment."""

  def __init__(self,
               name = 'smith_waterman_loss',
               reduction = tf.losses.Reduction.AUTO):
    super().__init__(name=name, reduction=reduction)

  def call(self, true_alignments_or_paths,
           alignment_output):
    """Computes a loss associated with the Smith-Waterman DP.

    Args:
      true_alignments_or_paths: The ground-truth alignments for the batch. Both
        sparse and dense representations of the alignments are allowed. For the
        sparse case, true_alignments_or_paths is expected to be a
        tf.Tensor<int>[batch, 3, align_len] = tf.stack([pos_x, pos_y,
        enc_trans], 1) such that (pos_x[b][i], pos_y[b][i], enc_trans[b][i])
        represents the i-th transition in the ground-truth alignment for example
        b in the minibatch. Both pos_x and pos_y are assumed to use one-based
        indexing and enc_trans follows the (categorical) 9-state encoding of
        edge types used throughout alignment.py. For the dense case,
        true_alignments_or_paths is instead expected to be a
        tf.Tensor<float>[batch, len_x, len_y, 9] with binary entries,
        representing the trajectory of the indices along the predicted alignment
        paths, by having a one along the taken edges, with nine possible edges
        for each i,j.
      alignment_output: An AlignmentOutput, which is a tuple (solution_values,
        solution_paths, sw_params) such that + 'solution_values' contains a
        tf.Tensor<float>[batch] with the (soft) optimal Smith-Waterman scores
        for the batch. + 'solution_paths', which is not used by the loss,
        optionally contains a tf.Tensor<float>[batch, len1, len2, 9] that
        describes the optimal soft alignments, being None otherwise. +
        'sw_params' contains a tuple (sim_mat, gap_open, gap_extend) of
        tf.Tensor objects parameterizing the Smith-Waterman LP such that +
        sim_mat is a tf.Tensor<float>[batch, len1, len2] (len1 <= len2) with the
        substitution values for pairs of sequences. + gap_open is a
        tf.Tensor<float>[], tf.Tensor<float>[batch] or tf.Tensor<float>[batch,
        len1, len2] (len1 <= len2) with the penalties for opening a gap. Must
        agree in rank with gap_extend.
          + gap_extend: a tf.Tensor<float>[], tf.Tensor<float>[batch] or
            tf.Tensor<float>[batch, len1, len2] (len1 <= len2) with the
            penalties for with the penalties for extending a gap. Must agree in
            rank with gap_open.

    Returns:
      The loss value for each example in the batch.
    """
    solution_values, _, sw_params = alignment_output
    return (solution_values -
            alignment.sw_score(sw_params, true_alignments_or_paths))


@gin.configurable
class BCEAlignmentLoss(tf.losses.Loss):
  """Implements a brute-force BCE loss for pairwise sequence alignment."""

  def __init__(self,
               name = 'bce_alignment_loss',
               reduction = tf.losses.Reduction.AUTO,
               pad_penalty = 1e8):
    super().__init__(name=name, reduction=reduction)
    self._pad_penalty = pad_penalty

  def call(self, true_alignments,
           alignment_output):
    """Computes a brute-force BCE loss for pairwise sequence alignment.

    Args:
      true_alignments: The ground-truth alignments for the batch, given by a
        expected tf.Tensor<int>[batch, 3, align_len] = tf.stack([pos_x, pos_y,
        enc_trans], 1) such that (pos_x[b][i], pos_y[b][i], enc_trans[b][i])
        represents the i-th transition in the ground-truth alignment for example
        b in the minibatch. Both pos_x and pos_y are assumed to use one-based
        indexing and enc_trans follows the (categorical) 9-state encoding of
        edge types used throughout alignment.py.
      alignment_output: A NaiveAlignmentOutput, which is a 3-tuple made of:
        + The alignment scores: tf.Tensor<float>[batch].
        + The pairwise match probabilities: tf.Tensor<int>[batch, len, len].
        + A 3-tuple containing the Smith-Waterman parameters: similarities, gap
          open and gap extend. Similaries is tf.Tensor<float>[batch, len, len],
          the gap penalties can be either tf.Tensor<float>[batch] or
          tf.Tensor<float>[batch, len, len].

    Returns:
      The loss value for each example in the batch.
    """
    _, match_indicators_pred, sw_params = alignment_output
    sim_mat, _, _ = sw_params
    shape, dtype = sim_mat.shape, match_indicators_pred.dtype

    match_indices_true = alignment.alignments_to_state_indices(
        true_alignments, 'match')
    updates_true = tf.ones([tf.shape(match_indices_true)[0]], dtype=dtype)
    match_indicators_true = tf.scatter_nd(
        match_indices_true, updates_true, shape=shape)

    raw_losses = tf.losses.binary_crossentropy(
        match_indicators_true[Ellipsis, tf.newaxis],
        match_indicators_pred[Ellipsis, tf.newaxis])

    mask = alignment.mask_from_similarities(
        sim_mat, dtype=dtype, pad_penalty=self._pad_penalty)
    return tf.reduce_sum(mask * raw_losses, axis=[1, 2])


@gin.configurable
class ProcrustesLoss(tf.losses.Loss):
  """Implements a loss for embeddings, up to a rigid transformation."""

  def __init__(self,
               name = 'procrustes_loss',
               reduction = tf.losses.Reduction.AUTO):
    super().__init__(name=name, reduction=reduction)

  def call(self, embs_true, embs_pred):
    """Computes the Procrustes loss between two (batches of) sets of vectors.

    Args:
      embs_true: a tf.Tensor<float>[batch_size, num_embs, dims] batch of
        'num_embs' embeddings in dimension 'dim'.
      embs_pred: a tf.Tensor<float>[batch_size, num_embs, dims] batch of
        'num_embs' embeddings in dimension 'dim'.

    Returns:
      The Procrustes loss value between each pair of embeddings in the batch.
    """
    embs_true_bar = embs_true - tf.reduce_mean(embs_true, axis=1, keepdims=True)
    embs_pred_bar = embs_pred - tf.reduce_mean(embs_pred, axis=1, keepdims=True)
    prod = tf.matmul(embs_true_bar, embs_pred_bar, transpose_a=True)
    _, u_left, v_right = tf.linalg.svd(prod, full_matrices=True)
    rotation_opt = tf.matmul(u_left, v_right, transpose_b=True)
    return tf.linalg.norm(
        tf.matmul(embs_true_bar, rotation_opt) - embs_pred_bar, axis=(1, 2))


@gin.configurable
class ContactLoss(tf.losses.Loss):
  """Implements a loss for contact matrices."""

  def __init__(self,
               name = 'contact',
               reduction = tf.losses.Reduction.AUTO,
               no_contact_fun=lambda x: tf.math.exp(-x),
               contact_fun=tf.math.sqrt,
               weights_fun=tf.identity,
               from_embs=False):
    """Loss for predicted positions, based on ground truth contact information.

    Args:
      name: the name of the loss
      reduction: how the loss is computed from element-wise losses.
      no_contact_fun: function used in the loss when there is no contact in the
        ground truth. (see below)
      contact_fun: function used in the loss when there is contact in the ground
        truth. (see below)
      weights_fun: a weight function, applied on |i-j|, where i, j are the
        matrix indices. (see below)
      from_embs: whether the loss is computed from predicted embeddings (True)
        or directly a predicted pairwise distance matrix (False, by default).

    Returns:
      A loss function
    """
    self._no_contact_fun = no_contact_fun
    self._contact_fun = contact_fun
    self._weights_fun = weights_fun
    self._from_embs = from_embs
    super().__init__(name=name, reduction=reduction)

  def call(self, contact_true, pred):
    """Computes the Contact loss between contact / distance matrices.

    Args:
      contact_true: a tf.Tensor<float>[batch_size, num_embs, num_embs], a batch
        of binary contact matrices for 'num_embs' embeddings.
      pred: a tf.Tensor<float> of shape either + [batch_size, num_embs, dims] if
        'from_embs' is True (embeddings case) a batch of 'num_embs' embeddings
        in dimension 'dim'. + [batch_size, num_embs, num_embs] if 'from_embs' is
        False (matrix case) a batch of pairwise distances for 'num_embs'
        embeddings.

    Returns:
      The contact loss values between the contact matrices and predictions
      in the batch. This is computed for an instance matrix in the batch as:
        loss(y, p) = sum_ij w_|i-j| fun(y_ij, p_ij),
      where y is the ground truth contact matrix and p is the predicted
      pairwise distance matrix.
        + fun(y_ij, _) is no_contact_fun if y_ij = 0, contact_fun if y_ij = 1.
        + w_|i-j| is weights_fun(|i-j|), and just |i-j| if None.
      If from_embs is true, the predicted matrix is the pairwise distance of the
      predicted embeddings.
    """
    if self._from_embs:
      pairw_dist_pred = pairs.square_distances(embs_1=pred, embs_2=pred)
    else:
      pairw_dist_pred = pred
    num_embs = tf.shape(pairw_dist_pred)[1]
    weights_range = tf.range(num_embs, dtype=tf.float32)
    weights_range_square = tf.abs(weights_range[tf.newaxis, :, tf.newaxis] -
                                  weights_range[tf.newaxis, tf.newaxis, :])
    weights_batch_square = self._weights_fun(weights_range_square)
    contact_true = tf.cast(contact_true, dtype=pred.dtype)
    mat_losses = contact_true * self._contact_fun(pairw_dist_pred) + (
        1 - contact_true) * self._no_contact_fun(pairw_dist_pred)
    return weights_batch_square * mat_losses


@gin.configurable
class MultiTaskLoss:
  """A loss to combine multiple ones for a model that outputs a Dict."""

  def __init__(self, losses):
    self._losses = losses
    # Make sure every loss has a weight.
    for level in self._losses.levels:
      for i in range(len(level)):
        if isinstance(level[i], tf.keras.losses.Loss):
          level[i] = (1.0, level[i])

  def _compute_weight_correction(self, labels, weights=None, epsilon=1e-9):
    """Account for weight sums for a specific head/loss."""
    replica_ctx = tf.distribute.get_replica_context()
    per_replica = (
        tf.shape(labels)[0] if weights is None else tf.math.reduce_sum(weights))
    total = replica_ctx.all_reduce('sum', per_replica)
    return 1.0 / (tf.cast(total, tf.float32) + epsilon)

  def __call__(self,
               y_true,
               y_pred,
               weights = None):
    # TODO(oliviert): Should we unflatten?
    y_true = multi_task.Backbone.unflatten(y_true)
    weights = multi_task.Backbone.unflatten(weights)
    if y_pred.shape != self._losses.shape:
      raise ValueError(
          f'The SeqAlign MultiTaskLoss shape {self._losses.shape} is not '
          f'matching the predictions shape {y_pred.shape}')

    total_loss = 0.0
    individual_losses = {}
    for weighted_loss, label, pred, batch_w in zip(self._losses, y_true, y_pred,
                                                   weights):
      loss_w, loss_fn = weighted_loss
      if loss_fn is None:
        continue
      loss_w *= self._compute_weight_correction(label, batch_w)
      loss = loss_w * tf.math.reduce_sum(
          loss_fn(label, pred, sample_weight=batch_w))
      total_loss += loss
      individual_losses[loss_fn.name] = loss
    return total_loss, individual_losses
