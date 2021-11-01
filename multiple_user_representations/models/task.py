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

"""Defines the task class that computes the loss and metrics for retrieval task."""

from typing import Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs


class MultiShotRetrievalTask(tfrs.tasks.Retrieval):
  """Extends the tfrs retrieval task to support multiple user representation (MUR).

  This class modifies the call function to support MUR. See base class for more
  details. For details on MUR see http://shortn/_PO6OdvUuAs.
  """

  def call(self,
           user_embeddings,
           candidate_embeddings,
           sample_weight = None,
           candidate_sampling_probability = None,
           eval_candidate_embeddings = None,
           compute_metrics = True,
           is_head_item = None):
    """Computes the loss function for next-item prediction.

    While computing the loss, the next-item's embedding is used as positive,
    while remaining items in the batch are negatives (in-batch negatives).

    Args:
      user_embeddings: User Embeddings [B, H, D] from the user tower, where H
        corresponds to the number of user representations.
      candidate_embeddings: The candidate embeddings corresponding to ground
        truth items that match the users. Shape: [B, D].
      sample_weight: Tensor of sample weights [B].
      candidate_sampling_probability: Optional tensor of candidate sampling
        probabilities. When given will be be used to correct the logits to
        reflect the sampling probability of in-batch negatives.
      eval_candidate_embeddings: Candidate Embeddings [B, N, D] from the item
        tower, where N corresponds to the number of candidates. The
        eval_candidate_embeddings is only used to evaluate the metrics. See
        evaluation (http://shortn/_PO6OdvUuAs#heading=h.tghsu9ag8g7x) with
        negative samples. When not given, all candidates are considered for
        evaluating metrics.
      compute_metrics: Whether to compute metrics or not.
      is_head_item: 1 if the positive candidate is a head item else 0. This is
        used to compute head/tail metrics. Shape: [B, 1].

    Returns:
      loss: Loss value tensor.
    """

    scores = tf.linalg.matmul(
        user_embeddings, candidate_embeddings, transpose_b=True)
    scores = tf.math.reduce_max(scores, axis=1)
    batch_size = tf.shape(scores)[0]
    num_candidates = tf.shape(scores)[-1]
    labels = tf.eye(batch_size, num_candidates)

    if self._temperature is not None:
      scores = scores / self._temperature

    if candidate_sampling_probability is not None:
      candidate_sampling_probability = tf.cast(candidate_sampling_probability,
                                               tf.float32)
      scores = tfrs.layers.loss.SamplingProbablityCorrection()(
          scores, candidate_sampling_probability)

    loss = self._loss(y_true=labels, y_pred=scores, sample_weight=sample_weight)

    if not compute_metrics:
      return loss

    if not self._factorized_metrics:
      return loss

    if eval_candidate_embeddings is None:
      update_op = self._factorized_metrics.update_state(
          user_embeddings, candidate_embeddings, is_head_item=is_head_item)
    else:
      update_op = self._factorized_metrics.update_state_with_negatives(
          user_embeddings,
          candidate_embeddings,
          eval_candidate_embeddings,
          is_head_item=is_head_item)

    with tf.control_dependencies([update_op]):
      return tf.identity(loss)

  def compute_cosine_disagreement_loss(self,
                                       query_head):
    """Computes the cosine disagreement loss given the query_head.

    Args:
      query_head: Tensor of shape [1, H, D].

    Returns:
      disagreement_loss: The cosine similarity computed across axis=1 for
        different queries in the query_head.
    """

    norm = tf.linalg.norm(query_head, axis=-1, keepdims=True)
    disagreement_queries = query_head / tf.stop_gradient(norm)
    query_embedding_score = tf.linalg.matmul(
        disagreement_queries, disagreement_queries, transpose_b=True)
    disagreement_loss = tf.reduce_sum(query_embedding_score)
    disagreement_loss -= tf.reduce_sum(tf.linalg.trace(query_embedding_score))
    return disagreement_loss


class MultiQueryStreaming(tfrs.layers.factorized_top_k.Streaming):
  """A wrapper for item candidates to efficiently retrieve top K scores.

  This class extends the tfrs factorized_top_k.Streaming class, which only
  supports a single user representation for retrieval. The class overrides the
  _compute_score to extend the functionality to multiple representations.
  See base class.
  """

  def _compute_score(self, queries,
                     candidates):
    """Computes multi-query score by overriding _compute_score from parent.

    Args:
      queries: Multiple queries.
      candidates: Candidate tensor.

    Returns:
      scores: Max score from multiple queries.
    """

    scores = tf.matmul(queries, candidates, transpose_b=True)
    scores = tf.reduce_max(scores, axis=1)
    return scores


class MultiQueryFactorizedTopK(tfrs.metrics.FactorizedTopK):
  """Computes metrics for multi user representations across top K candidates.

  We reuse the functionality from base class, while only modifying the
  update_state function to support multiple representations.
  See base class for details.
  """

  def update_state(self,
                   query_embeddings,
                   true_candidate_embeddings,
                   is_head_item = None):
    """Updates the state of the FactorizedTopK Metric.

    See the base class method `update_state` for details. This method extends
    the functionality to multiple user representation case, i.e. when
    query_embeddings is of shape [B, H, D].

    Args:
      query_embeddings: The query embeddings used to retrieve candidates.
      true_candidate_embeddings: The positive candidate embeddings.
      is_head_item: 1 if the positive candidate is a head item else 0. This is
        used to compute head/tail metrics. Shape: [B, 1].

    Returns:
      update_ops: The metric update op. Used for tf.v1 functionality.
    """

    # true_candidate_embeddings: [B, d]
    true_candidate_embeddings = tf.expand_dims(
        true_candidate_embeddings, axis=1)

    # positive_scores: B x H x 1
    positive_scores = tf.reduce_sum(
        query_embeddings * true_candidate_embeddings, axis=2, keepdims=True)

    # positive_scores: B x 1
    positive_scores = tf.reduce_max(positive_scores, axis=1)

    top_k_predictions, _ = self._candidates(query_embeddings, k=self._k)

    y_true = tf.concat(
        [tf.ones(tf.shape(positive_scores)),
         tf.zeros_like(top_k_predictions)],
        axis=1)
    y_pred = tf.concat([positive_scores, top_k_predictions], axis=1)

    update_ops = []
    for metric in self._top_k_metrics:
      if metric.name.startswith(
          ("head_", "Head_")) and is_head_item is not None:
        update_ops.append(
            metric.update_state(
                y_true=y_true, y_pred=y_pred, sample_weight=is_head_item))
      elif metric.name.startswith(
          ("tail_", "Tail_")) and is_head_item is not None:
        # If item is not head, then the item is considered to be a tail item.
        is_tail_item = 1.0 - is_head_item
        update_ops.append(
            metric.update_state(
                y_true=y_true, y_pred=y_pred, sample_weight=is_tail_item))
      else:
        update_ops.append(metric.update_state(y_true=y_true, y_pred=y_pred))
    return tf.group(update_ops)

  def update_state_with_negatives(
      self,
      query_embeddings,
      candidate_embeddings,
      neg_candidate_embeddings,
      is_head_item = None):
    """Updates the state of the FactorizedTopK Metric wrt the negative samples.

    The function computes the metrics only wrt the given negative samples.
    Unlike the update_state fn, this fn assumes that the number of candidate
    samples are fewer while finding topK candidates.

    Args:
      query_embeddings: The query embeddings used for retrieval. Shape: [B,H,D].
      candidate_embeddings: The candidate embeddings corresponding to ground
        truth items that match the query. Shape: [B,D].
      neg_candidate_embeddings: The negative candidate embeddings against which
        we evaluate. Shape: [B,N,D].
      is_head_item: 1 if the positive candidate is a head item else 0. This is
        used to compute head/tail metrics. Shape: [B, 1].

    Returns:
      update_ops: The metric op used for tf.v1 functionality.
    """

    pos_candidate_embeddings = tf.expand_dims(candidate_embeddings, axis=-1)
    positive_scores = tf.matmul(query_embeddings, pos_candidate_embeddings)
    positive_scores = tf.reduce_max(positive_scores, axis=1)

    negative_scores = tf.matmul(
        query_embeddings, neg_candidate_embeddings, transpose_b=True)
    negative_scores = tf.reduce_max(negative_scores, axis=1)

    y_true = tf.concat(
        [tf.ones_like(positive_scores),
         tf.zeros_like(negative_scores)],
        axis=1)
    y_pred = tf.concat([positive_scores, negative_scores], axis=1)

    update_ops = []
    for metric in self._top_k_metrics:
      if metric.name.startswith(
          ("head_", "Head_")) and is_head_item is not None:
        update_ops.append(
            metric.update_state(
                y_true=y_true, y_pred=y_pred, sample_weight=is_head_item))
      elif metric.name.startswith(
          ("tail_", "Tail_")) and is_head_item is not None:
        # If item is not head, then the item is considered to be a tail item.
        is_tail_item = 1.0 - is_head_item
        update_ops.append(
            metric.update_state(
                y_true=y_true, y_pred=y_pred, sample_weight=is_tail_item))
      else:
        update_ops.append(metric.update_state(y_true=y_true, y_pred=y_pred))

    return tf.group(update_ops)
