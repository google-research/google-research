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

"""Metrics modules to be used in dowstream train/evaluation."""

import collections

from jax import numpy as jnp
import numpy as np

from imp.max.core import utils
from imp.max.utils import typing


def construct_metrics_dictionary_from_metrics_stack(
    metrics_stack,
    prediction,
    target,
    mask = None,
    ):
  """Calculates metric values using a sequence of metric functions."""

  metrics = {}
  for metric_fn in metrics_stack:
    for key, value in metric_fn(prediction, target, mask).items():
      metrics[key] = value

  return metrics


class BaseMetric(object):
  """The base metric class."""

  def __init__(self, name = 'base_metric'):
    self.name = name
    self.np = np
    self.using_jax_mode = False

  def enable_jax_mode(self):
    self.np = jnp
    self.using_jax_mode = True
    return self

  def enable_numpy_mode(self):
    self.np = np
    return self

  def __call__(self, *args, **kwargs):
    raise NotImplementedError(
        'This is an abstract method. Please inherit from this class and '
        'implement the __call__ method.')


class Accuracy(BaseMetric):
  """Calculates an accuracy metric."""

  def __init__(self,
               top = 1,
               average_logits = True,
               one_hot_targets = True,
               name = 'accuracy'):
    """Initializes the accuracy metric.

    Args:
      top: An integer n selecting the top n predictions, or sequence of
        integers for multiple top predictions.
      average_logits: If True, averages the logits before running the
        accuracy calculation.
      one_hot_targets: Whether the target (labels) are one-hot vectors.
      name: The name of this metric object.
    """
    super().__init__(name=name)
    if isinstance(top, int):
      top = (top,)
    self.top = top
    self.average_logits = average_logits
    self.one_hot_targets = one_hot_targets

  def __call__(
      self,
      pred,
      true,
      mask = None,
  ):
    """Calculates the accuracy with respect to the predictions.

    Args:
      pred: the predicted logits.
      true: the ground-truth labels.
      mask: an optional mask applied to each sample. This mask should have the
        same shape as pred.shape[:-1].

    Returns:
      the accuracy of predictions.
    """

    if self.one_hot_targets:
      # true one-hot to index
      true = self.np.argmax(true, axis=-1)  # (B, ...)

    # Expand along the last dimension (for easier comparison w/ the preds)
    true = true[Ellipsis, self.np.newaxis]  # (B, ..., 1)

    if self.average_logits:
      # average logits if multiple-instance
      if len(pred.shape) == 3:
        pred = pred.mean(axis=1, keepdims=True)  # (B, 1, C)

    if len(true.shape) != len(pred.shape):
      raise ValueError('Rank mismatch between predictions and labels!'
                       f'predictions: {pred.shape}, labels: {true.shape}')

    if mask is not None and mask.shape != pred.shape[:-1]:
      raise ValueError('Shape mismatch between prediction samples and mask!'
                       f'predictions: {pred.shape[:-1]}, mask: {mask.shape}')

    # sort based on logits activation (B, ..., C)
    max_top = max(self.top)
    if self.using_jax_mode:
      # argsort is inefficient during training, so we use a custom util.
      pred = utils.top_k(pred, k=max_top)
    else:
      # Negate preds to sort descending
      pred = self.np.argsort(-pred, axis=-1)[Ellipsis, :max_top]

    metrics = {}
    for top in self.top:
      top_x = pred[Ellipsis, :top]  # (B, ..., top)
      match_x = self.np.max(top_x == true, axis=-1)
      if mask is None:
        top_x_acc = self.np.mean(match_x)
      else:
        # double check the top-k rank shape to avoid wrong broadcasting
        if mask.shape != match_x.shape:
          raise ValueError('Shape mismatch between score samples and mask!'
                           f'scores: {match_x.shape}, mask: {mask.shape}')
        epsilon = 1e-20  # Use epsilon to handle when the mask is all 0.
        top_x_acc = self.np.sum(match_x * mask) / (self.np.sum(mask) + epsilon)

      metrics[f'top_{top}'] = top_x_acc

    return metrics


class RetrievalRecall(BaseMetric):
  """Calculate retrieval Recall at multiple levels."""

  def __init__(self,
               at,
               return_median_rank = True,
               instance_selection_method = 'boundary',
               name = 'recall'):
    """Constructs a Retrieval Recall object operating on query-key inputs."""

    super().__init__(name=name)
    self.at = at
    if isinstance(self.at, int):
      self.at = tuple(self.at)
    self.return_median_rank = return_median_rank
    self.instance_selection_method = instance_selection_method

    if self.instance_selection_method == 'best':
      self.calculate_ranks = self.best_instance_ranks
    elif self.instance_selection_method == 'boundary':
      self.calculate_ranks = self.boundary_ranks
    else:
      raise ValueError(
          '`instance_selection_method` could only be one of `best` or '
          f'`boundary`. Instead, received {self.instance_selection_method!r}.')

  def l2_normalize(self,
                   inputs,
                   axis = -1):
    """Normalizes an ND-Array based on L2-norm along a given axis."""
    return inputs / (
        self.np.linalg.norm(inputs, axis=axis, keepdims=True) + 1e-6)

  def calculate_cross_similarity(
      self,
      modality_1,
      modality_2,
  ):
    """Calculate Modality_1 vs. Modality_2 pair-wise similarities."""

    # normalize embeddings
    modality_1 = self.l2_normalize(modality_1, axis=-1)  # (B1, N1, D)
    modality_2 = self.l2_normalize(modality_2, axis=-1)  # (B1, N2, D)

    # calculate cross-modal similarities for all pairs of samples (per instance)
    # resulting in -> (B1, N1, B2, N2)
    m1_vs_m2 = self.np.einsum('bmd,cnd->bmcn', modality_1, modality_2)

    return m1_vs_m2

  def best_instance_ranks(self, scores):
    """Calculates query-to-key retrieval ranks by choosing the best instance."""

    # calculate where the maximum score is assigned for B=B1=B2 queries
    ranks_q_to_k = self.np.argsort(-scores, axis=2)  # (B1, N1, B2, N2)

    # rank indices for B1 queries to compare with actual ranks -> (B1, 1, 1, 1)
    ## this is broadcastable, which means that any of N1 * B2 * N2 ranks is
    ## compared to a single index
    b1 = scores.shape[0]
    rank_ids_q_to_k = self.np.arange(b1)[:, None, None, None]

    # check where the ranks are equal to the indices
    matched_ranks_q_to_k = self.np.where(  # -> (B1, N1, B2, N2)
        ranks_q_to_k == rank_ids_q_to_k, rank_ids_q_to_k, -self.np.inf)

    # the resulting output of the last line has one number in range [0, B) and
    # -inf for the rest B-1 elements. this is because all ranks are unique
    # numbers in [0, B) and only one of them is equal to rank_ids.
    # hence, we can get the location where that element exists by taking the
    # argmax of it along the axis=2 which results in -> (B1, N1)
    matched_ranks_q_to_k = self.np.argmax(matched_ranks_q_to_k, axis=2)

    # the best rank among the N2 instances is chosen -> (B1, N1)
    matched_ranks_q_to_k = self.np.min(matched_ranks_q_to_k, axis=2)

    # flatten the resulting matrix to (B1 * N1)
    matched_ranks_q_to_k = matched_ranks_q_to_k.reshape(-1)

    return matched_ranks_q_to_k

  def boundary_ranks(self, scores):
    """Calculates query-to-key retrieval ranks by using a boundary."""

    b1, n1, _, n2 = scores.shape
    scores = scores.reshape(b1, n1, -1)
    # calculate where the maximum score is assigned for B1 queries over
    # B2*N2 keys. each query has n1 instances, which are treated
    # independently. however, in this instance selection method we sort
    # the indices over all B2*N2 keys
    ranks_q_to_k = self.np.argsort(-scores, axis=2)  # (B1, N1, B2 * N2)

    # rank indices for B1 queries to compare with actual ranks -> (B1, 1, 1)
    ## this is broadcastable, which means that any of N1 * B2 * N2 ranks is
    ## compared to a single index
    rank_ids_q_to_k = self.np.arange(b1)[:, None, None]

    # check where the ranks are withtin the valid lower and upper bounds
    hit = self.np.logical_and(
        n2 * rank_ids_q_to_k <= ranks_q_to_k,
        ranks_q_to_k < n2*(1 + rank_ids_q_to_k))
    matched_ranks_q_to_k = self.np.where(  # -> (B1, N1, B2 * N2)
        hit, rank_ids_q_to_k, -self.np.inf)

    # similar to the 'best' instance selection method we can get the
    # location where the best ranked element exists by taking the argmax
    # all filtered ranks along the axis=2 which results in -> (B1, N1)
    matched_ranks_q_to_k = self.np.argmax(matched_ranks_q_to_k, axis=2)

    # flatten the resulting matrix to (B1 * N1)
    matched_ranks_q_to_k = matched_ranks_q_to_k.reshape(-1)

    return matched_ranks_q_to_k

  def __call__(
      self,
      modality_1,
      modality_2,
      mask = None,
  ):

    if mask is not None:
      raise NotImplementedError

    # modality_1: (B1, N1, D), modality_2: (B2, N2, D)
    b1 = modality_1.shape[0]
    b2 = modality_2.shape[0]
    if b1 != b2:
      raise ValueError('n_batch should be equal for both inputs')

    rank_1 = len(modality_1.shape)
    rank_2 = len(modality_2.shape)

    if rank_1 != rank_2:
      raise ValueError('Input ranks must be equal and exactly 3. '
                       f'{rank_1} and {rank_2} were given.')

    scores = self.calculate_cross_similarity(modality_1, modality_2)
    matched_ranks_m1_to_m2 = self.calculate_ranks(scores)
    scores = self.np.transpose(scores, [2, 3, 0, 1])
    matched_ranks_m2_to_m1 = self.calculate_ranks(scores)

    metrics_m1_to_m2 = {}
    metrics_m2_to_m1 = {}
    # calculate R at different levels
    for at in self.at:
      metrics_m1_to_m2[f'R{at}'] = self.np.mean(matched_ranks_m1_to_m2 < at)
      metrics_m2_to_m1[f'R{at}'] = self.np.mean(matched_ranks_m2_to_m1 < at)

    # calculate median rank
    if self.return_median_rank:
      metrics_m1_to_m2['MedianRank'] = self.np.median(matched_ranks_m1_to_m2+1)
      metrics_m2_to_m1['MedianRank'] = self.np.median(matched_ranks_m2_to_m1+1)

    return {'m1_vs_m2': metrics_m1_to_m2,
            'm2_vs_m1': metrics_m2_to_m1}


METRIC_BANK = {
    'accuracy': Accuracy,
    'retrieval_recall': RetrievalRecall,
}


def create_metric_stack(config):
  config = config.as_dict()
  metrics = collections.defaultdict(tuple)
  for serving in config:
    for metric_config in config[serving]:
      metric_class = METRIC_BANK[metric_config['name']](**metric_config)
      metrics[serving] += (metric_class,)

  return metrics
