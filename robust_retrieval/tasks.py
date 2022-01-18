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

"""A factorized retrieval task extended for distributionally-robust learning."""

import sys
from typing import List, Optional, Text, Union

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


class RobustRetrieval(tf.keras.layers.Layer):
  """A factorized retrieval task for distributionally-robust learning.

  Recommender systems are often composed of two components:
  - a retrieval model, retrieving O(thousands) candidates from a corpus of
    O(millions) candidates.
  - a ranker model, scoring the candidates retrieved by the retrieval model to
    return a ranked shortlist of a few dozen candidates.

  This task defines models that facilitate efficient retrieval of candidates
  from large corpora by maintaining a two-tower, factorized structure: separate
  query and candidate representation towers, joined at the top via a lightweight
  scoring function.
  We implemented a robust learning technique based on Distributionally-Robust
  Optimization (DRO), which aims to improve worse-case subgroup performance.

  We further propose several improvements for the robust retrieval model.
  - Streaming-loss DRO: keep streaming estimations of group loss.
  - Metric-based DRO: replace loss-based re-weighting with metric-based
  re-weighting.
  - KL-regularization on DRO: penalize weight distributions that diverge from
  empirical distributions.

  Reference:
  Sagawa S, Koh P W, et al. Distributionally robust neural networks for group
  shifts: On the importance of regularization for worst-case generalization.
  https://arxiv.org/abs/1911.08731.
  """

  def __init__(self,
               group_labels: Union[List[Text], List[int]],
               group_loss_init: List[float],
               group_metric_init: List[float],
               group_weight_init: List[float],
               group_reweight_strategy: Text,
               loss: Optional[tf.keras.losses.Loss] = None,
               metrics: Optional[tfrs.metrics.FactorizedTopK] = None,
               topk: Optional[int] = 100,
               candidates: Optional[tf.data.Dataset] = None,
               temperature: Optional[float] = None,
               num_hard_negatives: Optional[int] = None,
               dro_temperature: Optional[float] = 0.1,
               streaming_group_loss: Optional[bool] = False,
               streaming_group_loss_lr: Optional[float] = 0.01,
               streaming_group_metric_lr: Optional[float] = 0.01,
               group_metric_with_decay: Optional[bool] = False,
               metric_update_freq: Optional[int] = 1,
               name: Optional[Text] = "robust_retrieval_task") -> None:
    """Initializes the task.

    Args:
      group_labels: A list of integers or strings as group identity labels. Used
        to define subgroups for optimizing robust loss.
      group_loss_init: A list of [num_groups] floats for group loss
        initialization, e.g. [1.0, 2.0, 3.0].
      group_metric_init: A list of [num_groups] floats for group metric
        initialization, e.g. [0.0, 0.0, 0.0].
      group_weight_init: A list of [num_groups] floats for group weight
        initialization that add up to 1, e.g. [0.3, 0.2, 0.5].
      group_reweight_strategy: Group reweighting strategy. Shall be one of
        ["group-dro", "uniform"].
      loss: Loss function. Defaults to
        `tf.keras.losses.CategoricalCrossentropy`.
      metrics: Object for evaluating top-K metrics over a corpus of candidates.
        These metrics measure how good the model is at picking the true
        candidate out of all possible candidates in the system. Note, because
        the metrics range over the entire candidate set, they are usually much
        slower to compute. Consider set `compute_metrics=False` during training
        to save the time in computing the metrics.
      topk: Number of top scoring candidates to retrieve for metric evaluation.
      candidates: A set of candidate items.
      temperature: Temperature of the softmax.
      num_hard_negatives: If positive, the `num_hard_negatives` negative
        examples with largest logits are kept when computing cross-entropy loss.
        If larger than batch size or non-positive, all the negative examples are
        kept.
      dro_temperature: A float, temperature of the group re-weighting in DRO. A
        suggested range is between [0.001,0.1].
      streaming_group_loss: if `True` will use streaming loss estimations.
      streaming_group_loss_lr: between [0,1], larger value will let the
        estimations of group loss focus more on the current batch.
      streaming_group_metric_lr: between [0,1], larger value will let the
        estimations of group metric focus more on the current batch.
      group_metric_with_decay: if `True` will use decay for group metric update.
      metric_update_freq: group metric updates every after n batch.
      name: Optional task name.
    """
    super().__init__(name=name)
    # Robust training settings.
    self._group_labels = group_labels
    self._group_labels_matrix = tf.reshape(np.array(group_labels), [-1, 1])
    self._num_groups = len(group_labels)
    self._group_reweight_strategy = group_reweight_strategy
    self._dro_temperature = dro_temperature
    self._streaming_group_loss = streaming_group_loss
    self._streaming_group_loss_lr = streaming_group_loss_lr
    self._streaming_group_metric_lr = streaming_group_metric_lr
    self._metric_update_freq = metric_update_freq
    self._group_metric_with_decay = group_metric_with_decay

    # Initialization of group weights.
    self._group_weights = tf.Variable(
        initial_value=tf.convert_to_tensor(group_weight_init), trainable=False)

    # Initialization of group loss.
    self._group_loss = tf.Variable(
        initial_value=tf.convert_to_tensor(group_loss_init), trainable=False)

    self._sample_loss = (
        loss if loss is not None else tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE))

    self._topk = topk
    self._factorized_metrics = metrics
    if isinstance(candidates, tf.data.Dataset):
      candidates = tfrs.layers.factorized_top_k.Streaming(
          k=self._topk).index(candidates)

    # Initialization of group metric.
    self._group_metric_estimates = tf.Variable(
        initial_value=tf.convert_to_tensor(group_metric_init), trainable=False)

    self._group_metrics = []
    for x in range(self._num_groups):
      self._group_metrics.append(
          tf.keras.metrics.TopKCategoricalAccuracy(
              k=self._topk,
              name=f"group_{self._group_labels[x]}_top{self._topk}_accuracy",
          ))

    self._candidates = candidates
    self._temperature = temperature
    self._num_hard_negatives = num_hard_negatives
    self._name = name

  @property
  def factorized_metrics(self) -> Optional[tfrs.metrics.FactorizedTopK]:
    """The metrics object used to compute retrieval metrics."""

    return self._factorized_metrics

  @factorized_metrics.setter
  def factorized_metrics(self,
                         value: Optional[tfrs.metrics.FactorizedTopK]) -> None:
    """Sets factorized metrics."""

    self._factorized_metrics = value

  def call(self,
           query_embeddings: tf.Tensor,
           candidate_embeddings: tf.Tensor,
           group_identity: tf.Tensor,
           step_count: tf.Tensor,
           sample_weight: Optional[tf.Tensor] = None,
           candidate_sampling_probability: Optional[tf.Tensor] = None,
           candidate_ids: Optional[tf.Tensor] = None,
           compute_metrics: bool = True) -> tf.Tensor:
    """Computes the task loss and metrics.

    The main argument are pairs of query and candidate embeddings: the first row
    of query_embeddings denotes a query for which the candidate from the first
    row of candidate embeddings was selected by the user.

    The task will try to maximize the affinity of these query, candidate pairs
    while minimizing the affinity between the query and candidates belonging
    to other queries in the batch.

    Args:
      query_embeddings: [num_queries, embedding_dim] tensor of query
        representations.
      candidate_embeddings: [num_queries, embedding_dim] tensor of candidate
        representations.
      group_identity: [num_queries] tensor of query group identity.
      step_count: Number of training steps.
      sample_weight: [num_queries] tensor of sample weights.
      candidate_sampling_probability: Optional tensor of candidate sampling
        probabilities. When given will be be used to correct the logits to
        reflect the sampling probability of negative candidates.
      candidate_ids: Optional tensor containing candidate ids. When given
        enables removing accidental hits of examples used as negatives. An
        accidental hit is defined as an candidate that is used as an in-batch
        negative but has the same id with the positive candidate.
      compute_metrics: If true, metrics will be computed. Because evaluating
        metrics may be slow, consider disabling this in training.

    Returns:
      loss: Tensor of loss values.
    """

    scores = tf.linalg.matmul(
        query_embeddings, candidate_embeddings, transpose_b=True)

    num_queries = tf.shape(scores)[0]
    num_candidates = tf.shape(scores)[1]

    labels = tf.eye(num_queries, num_candidates)

    if candidate_sampling_probability is not None:
      scores = tfrs.layers.loss.SamplingProbablityCorrection()(
          scores, candidate_sampling_probability)

    if candidate_ids is not None:
      scores = tfrs.layers.loss.RemoveAccidentalHits()(labels, scores,
                                                       candidate_ids)

    if self._num_hard_negatives is not None:
      scores, labels = tfrs.layers.loss.HardNegativeMining(
          self._num_hard_negatives)(scores, labels)

    if self._temperature is not None:
      scores = scores / self._temperature

    sample_loss = self._sample_loss(y_true=labels, y_pred=scores)

    # group_mask: [num_groups, num_queries], cur_group_loss: [num_groups]
    cur_group_loss, group_mask = self._compute_group_loss(
        sample_loss, group_identity)

    # Set default DRO update ops.
    group_loss_update = tf.no_op()
    group_metric_update = tf.no_op()
    group_weights_update = tf.no_op()

    # Note: only update loss/metric estimations when subgroup exists in a batch.
    # group_exist_in_batch: [num_groups], bool
    group_exist_in_batch = tf.math.reduce_sum(group_mask, axis=1) > 1e-16

    if self._streaming_group_loss:
      # Perform streaming estimation of group loss.
      stream_group_loss = (
          1 - tf.cast(group_exist_in_batch, "float32") *
          self._streaming_group_loss_lr
      ) * self._group_loss + self._streaming_group_loss_lr * cur_group_loss
      group_loss_update = self._group_loss.assign(
          stream_group_loss, read_value=False)
    else:
      group_loss_update = self._group_loss.assign(
          cur_group_loss, read_value=False)

    if self._group_reweight_strategy == "loss-dro":
      # Perform loss-based group weight updates.
      with tf.control_dependencies([group_loss_update]):
        group_weights_update = self._update_group_weights(self._group_loss)

    elif self._group_reweight_strategy == "metric-dro":
      # Perform metric-based group weight updates.
      # Note: only update when subgroup exists in a batch.
      # Assuming only update weights at every `_metric_update_freq` epochs
      # TODO(xinyang,tyao,jiaxit): change to sampled metric for effiency.
      if (step_count % self._metric_update_freq) == 0:
        batch_group_metric_update = self._update_group_metrics(
            query_embeddings, candidate_embeddings, group_mask)
        with tf.control_dependencies([batch_group_metric_update]):
          if self._group_metric_with_decay:
            stream_group_metric_lr = tf.cast(
                group_exist_in_batch,
                "float32") * self._streaming_group_metric_lr
            stream_group_metrics = (
                1 - stream_group_metric_lr
            ) * self._group_metric_estimates + stream_group_metric_lr * self.get_group_metrics(
            )
            group_metric_update = self._group_metric_estimates.assign(
                stream_group_metrics, read_value=False)
            group_weights_update = self._update_group_weights(
                1 - stream_group_metrics)
          else:
            group_weights_update = self._update_group_weights(
                1 - self.get_group_metrics())

    update_ops = [group_loss_update, group_metric_update, group_weights_update]

    if compute_metrics and (self._factorized_metrics is not None):
      update_ops.append(
          self._factorized_metrics.update_state(query_embeddings,
                                                candidate_embeddings))

    with tf.control_dependencies([tf.group(update_ops)]):
      # Add group log for analysis and debuggging.
      self._add_group_logs(cur_group_loss, step_count)
      return tf.reduce_sum(
          tf.stop_gradient(self._group_weights) * cur_group_loss) * tf.cast(
              num_queries, dtype="float32")

  def _compute_group_loss(self, sample_loss, group_identity):
    """Calculate subgroup losses.

    Args:
       sample_loss: Tensor of [num_queries] representing loss for each query.
       group_identity: Tensor of [num_queries] representing the group identity
         for each query.

    Returns:
       group_loss: Tensor of group loss values.
       group_mask: Tensor of [num_groups, num_queries].
    """
    # Shape of group_mask: [num_groups, num_queries].
    group_mask = tf.cast(
        tf.equal(group_identity, self._group_labels_matrix), dtype="float32")
    group_cnts = tf.reduce_sum(group_mask, axis=1)
    # Avoid divide by zero.
    group_cnts += tf.cast(group_cnts == 0, dtype="float32")
    # group loss shape: [num_groups]
    group_loss = tf.divide(
        tf.reduce_sum(group_mask * sample_loss, axis=1), group_cnts)
    return group_loss, group_mask

  def _update_group_metrics(self, query_embeddings, true_candidate_embeddings,
                            group_mask):
    """Perform group metric updates."""
    # [batch_size, 1]
    positive_scores = tf.reduce_sum(
        query_embeddings * true_candidate_embeddings, axis=1, keepdims=True)

    # [batch_size, k]
    top_k_predictions, _ = self._candidates(query_embeddings, k=self._topk)

    y_true = tf.concat(
        [tf.ones_like(positive_scores),
         tf.zeros_like(top_k_predictions)],
        axis=1)
    y_pred = tf.concat([positive_scores, top_k_predictions], axis=1)

    update_ops = []
    for group_id, metric in enumerate(self._group_metrics):
      if self._group_metric_with_decay:
        # Reset states to get batch-wise metrics.
        metric.reset_states()
      update_ops.append(
          metric.update_state(
              y_true=y_true, y_pred=y_pred, sample_weight=group_mask[group_id]))

    return tf.group(update_ops)

  def _update_group_weights(self, group_hardness, read_value=False):
    """Compute subgroup weights.

    Args:
      group_hardness: Tensor of [num_groups] representing hardness for each
        subgroup, for example, group loss or group metric.
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op. See
        tf.Variable.assign.

    Returns:
      group_weights_assign_op: Op of group weights assignment.
    """
    new_group_weights = tf.nn.softmax(
        tf.math.log(self._group_weights) +
        self._dro_temperature * group_hardness,
        axis=-1)
    group_weights_assign_op = self._group_weights.assign(
        new_group_weights, read_value=read_value)
    return group_weights_assign_op

  def _get_group_metrics(self):
    """Return the latest subgroup metrics."""
    return tf.convert_to_tensor(
        [metric.result() for metric in self._group_metrics])

  def _add_group_logs(self, cur_group_loss, step_count):
    """Add to group loss and weights to tensorboard."""
    tf.print("step_count:", step_count, output_stream=sys.stdout)
    tf.print("group loss:", cur_group_loss, output_stream=sys.stdout)
    tf.print("group est. loss:", self._group_loss, output_stream=sys.stdout)
    tf.print("group weights:", self._group_weights, output_stream=sys.stdout)
    group_summary = {}
    group_loss_summary = {
        f"Batch_GroupLoss_{self._group_labels[i]}": cur_group_loss[i]
        for i in range(self._num_groups)
    }
    group_loss_est_summary = {
        f"Est_GroupLoss_{self._group_labels[i]}": self._group_loss[i]
        for i in range(self._num_groups)
    }
    group_weight_summary = {
        f"GroupWeights_{self._group_labels[i]}": self._group_weights[i]
        for i in range(self._num_groups)
    }
    group_summary.update(group_loss_summary)
    group_summary.update(group_loss_est_summary)
    group_summary.update(group_weight_summary)
    self._add_tf_scalar_summary(group_summary, step_count)

  def _add_tf_histogram_summary(self, tensor_dict, step):
    for key, tensor in tensor_dict.items():
      if tensor is not None:
        if self._name is not None:
          tf.summary.histogram(f"{self._name}/{key}", tensor, step=step)
        else:
          tf.summary.histogram(key, tensor, step=step)

  def _add_tf_scalar_summary(self, tensor_dict, step):
    for key, tensor in tensor_dict.items():
      if tensor is not None:
        if self._name is not None:
          tf.summary.scalar(f"{self._name}/{key}", tensor, step=step)
        else:
          tf.summary.scalar(key, tensor, step=step)
