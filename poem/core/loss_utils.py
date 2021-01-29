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

"""Loss computation utility functions."""

import functools

import tensorflow as tf
import tensorflow_probability as tfp

from poem.core import common
from poem.core import data_utils
from poem.core import distance_utils
from poem.core import keypoint_utils


def create_sample_distance_fn(
    pair_type=common.DISTANCE_PAIR_TYPE_ALL_PAIRS,
    distance_kernel=common.DISTANCE_KERNEL_SQUARED_L2,
    pairwise_reduction=common.DISTANCE_REDUCTION_MEAN,
    componentwise_reduction=common.DISTANCE_REDUCTION_MEAN,
    **distance_kernel_kwargs):
  """Creates sample distance function.

  Args:
    pair_type: An enum string (see `common`) for type of pairs to use.
    distance_kernel: An enum string (see `common`) or a function handle for
      point distance kernel to use.
    pairwise_reduction: An enum string (see `common`) or a function handle for
      pairwise distance reducer to use. If not a supported enum string, uses it
      directly as a function handle.
    componentwise_reduction: An enum string (see `common`) or a function handle
      for component-wise distance reducer to use. If not a supported enum
      string, uses it directly as a function handle.
    **distance_kernel_kwargs: A dictionary for additional arguments to be passed
      to the distance kernel. The keys are in the format
      `${distance_kernel_name}_${argument_name}`.

  Returns:
    A function handle for computing sample group distances that takes two
      tensors of shape [..., num_components, num_embeddings, embedding_dim] as
      input.
  """

  def get_distance_matrix_fn():
    """Selects point distance matrix function."""
    if pair_type == common.DISTANCE_PAIR_TYPE_ALL_PAIRS:
      l2_distance_computer = distance_utils.compute_all_pair_l2_distances
    elif pair_type == common.DISTANCE_PAIR_TYPE_CORRESPONDING_PAIRS:
      l2_distance_computer = distance_utils.compute_corresponding_pair_l2_distances

    if distance_kernel == common.DISTANCE_KERNEL_SQUARED_L2:
      return functools.partial(l2_distance_computer, squared=True)

    if distance_kernel == common.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB:

      def compute_l2_sigmoid_matching_distances(lhs, rhs):
        """Computes L2 sigmoid matching probability distances."""
        inner_distances = l2_distance_computer(lhs, rhs, squared=False)
        return distance_utils.compute_sigmoid_matching_probabilities(
            inner_distances,
            raw_a_initializer=distance_kernel_kwargs.get(
                (distance_kernel + '_raw_a_initializer'), None),
            b_initializer=distance_kernel_kwargs.get(
                (distance_kernel + '_b_initializer'), None),
            name=distance_kernel_kwargs.get((distance_kernel + '_name'),
                                            'MatchingSigmoid'))

      return compute_l2_sigmoid_matching_distances

    if distance_kernel == common.DISTANCE_KERNEL_EXPECTED_LIKELIHOOD:

      def compute_gaussian_likelihoods(lhs, rhs):
        """Computes sample likelihoods."""
        num_lhs_samples = lhs.shape.as_list()[-2] - 2
        num_rhs_samples = rhs.shape.as_list()[-2] - 2
        lhs_means, lhs_stddevs, lhs_samples = tf.split(
            lhs, [1, 1, num_lhs_samples], axis=-2)
        rhs_means, rhs_stddevs, rhs_samples = tf.split(
            rhs, [1, 1, num_rhs_samples], axis=-2)
        rhs_likelihoods = distance_utils.compute_gaussian_likelihoods(
            lhs_means,
            lhs_stddevs,
            rhs_samples,
            min_stddev=distance_kernel_kwargs.get(
                distance_kernel + '_min_stddev', None),
            max_squared_mahalanobis_distance=distance_kernel_kwargs.get(
                distance_kernel + '_max_squared_mahalanobis_distance', None),
            smoothing=distance_kernel_kwargs.get(distance_kernel + '_smoothing',
                                                 None))
        lhs_likelihoods = distance_utils.compute_gaussian_likelihoods(
            rhs_means,
            rhs_stddevs,
            lhs_samples,
            l2_distance_computer=l2_distance_computer,
            min_stddev=distance_kernel_kwargs.get(
                distance_kernel + '_min_stddev', None),
            max_squared_mahalanobis_distance=distance_kernel_kwargs.get(
                distance_kernel + '_max_squared_mahalanobis_distance', None),
            smoothing=distance_kernel_kwargs.get(distance_kernel + '_smoothing',
                                                 None))
        return (rhs_likelihoods + lhs_likelihoods) / 2.0

      return compute_gaussian_likelihoods

    raise ValueError('Unsupported distance kernel: `%s`.' %
                     str(distance_kernel))

  def get_pairwise_distance_reduction_fn():
    """Selects pairwise distance reduction function."""
    if pairwise_reduction == common.DISTANCE_REDUCTION_MEAN:
      return functools.partial(tf.math.reduce_mean, axis=[-2, -1])
    if pairwise_reduction == common.DISTANCE_REDUCTION_LOWER_HALF_MEAN:
      return functools.partial(
          data_utils.compute_lower_percentile_means, axis=[-2, -1], q=50)
    if pairwise_reduction == common.DISTANCE_REDUCTION_NEG_LOG_MEAN:
      return lambda x: -tf.math.log(tf.math.reduce_mean(x, axis=[-2, -1]))

    if pairwise_reduction == common.DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN:

      def compute_lower_half_negative_log_mean(x):
        return -tf.math.log(
            data_utils.compute_lower_percentile_means(x, axis=[-2, -1], q=50))

      return compute_lower_half_negative_log_mean

    if pairwise_reduction == common.DISTANCE_REDUCTION_ONE_MINUS_MEAN:
      return lambda x: 1.0 - tf.math.reduce_mean(x, axis=[-2, -1])

    return pairwise_reduction

  def get_componentwise_distance_reduction_fn():
    """Selects component-wise distance reduction function."""
    if componentwise_reduction == common.DISTANCE_REDUCTION_MEAN:
      return functools.partial(tf.math.reduce_mean, axis=[-1])

    return componentwise_reduction

  def sample_distance_fn(lhs, rhs):
    """Computes sample distances."""
    distances = get_distance_matrix_fn()(lhs, rhs)
    distances = get_pairwise_distance_reduction_fn()(distances)
    distances = get_componentwise_distance_reduction_fn()(distances)
    return distances

  return sample_distance_fn


def compute_negative_indicator_matrix(anchor_points,
                                      match_points,
                                      distance_fn,
                                      min_negative_distance,
                                      anchor_point_masks=None,
                                      match_point_masks=None):
  """Computes all-pair negative match indicator matrix.

  Args:
    anchor_points: A tensor for anchor points. Shape = [num_anchors, ...,
      point_dim].
    match_points: A tensor for match points. Shape = [num_matches, ...,
      point_dim].
    distance_fn: A function handle for computing distance matrix.
    min_negative_distance: A float for the minimum negative distance threshold.
    anchor_point_masks: A tensor for anchor point masks. Shape = [num_anchors,
      ...]. Ignored if None.
    match_point_masks: A tensor for match point masks. Shape = [num_matches,
      ...]. Ignored if None.

  Returns:
    A boolean tensor for negative indicator matrix. Shape = [num_anchors,
      num_matches].
  """
  distance_matrix = distance_utils.compute_distance_matrix(
      anchor_points,
      match_points,
      distance_fn=distance_fn,
      start_point_masks=anchor_point_masks,
      end_point_masks=match_point_masks)
  return distance_matrix >= min_negative_distance


def compute_hard_negative_distances(anchor_match_distance_matrix,
                                    negative_indicator_matrix,
                                    use_semi_hard=False,
                                    anchor_positive_mining_distances=None,
                                    anchor_match_mining_distance_matrix=None):
  """Computes (semi-)hard negative distances.

  Args:
    anchor_match_distance_matrix: A tensor for anchor/match distance matrix.
      Shape = [num_anchors, num_matches].
    negative_indicator_matrix: A tensor for anchor/match negative indicator
      matrix. Shape = [num_anchors, num_matches].
    use_semi_hard: A boolean for whether to compute semi-hard negative distances
      instead of hard negative distances.
    anchor_positive_mining_distances: A tensor for positive distances of each
      anchor for (semi-)hard negative mining. Only used if `use_semi_hard` is
      True. Shape = [num_anchors].
    anchor_match_mining_distance_matrix: A tensor for an alternative
      anchor/match distance matrix to use for (semi-)hard negative mining. Use
      None to ignore and use `anchor_match_distance_matrix` instead. If
      specified, must be of the same shape as `anchor_match_distance_matrix`.

  Returns:
    hard_negative_distances: A tensor for (semi-)hard negative distances. Shape
      = [num_amchors]. If an anchor has no (semi-)hard negative match, its
      negative distance will be assigned as the maximum value of
      anchor_match_distance_matrix.dtype.
    hard_negative_mining_distances: A tensor for (semi-)hard negative mining
      distances. Shape = [num_amchors]. If an anchor has no (semi-)hard negative
      match, its negative distance will be assigned as the maximum value of
      anchor_match_distance_matrix.dtype.

  Raises:
    ValueError: If `use_semi_hard` is True, but
      `anchor_positive_mining_distances` is not specified.
  """
  indicators = negative_indicator_matrix
  if anchor_match_mining_distance_matrix is None:
    anchor_match_mining_distance_matrix = anchor_match_distance_matrix

  if use_semi_hard:
    if anchor_positive_mining_distances is None:
      raise ValueError('Positive match embeddings must be specified to compute '
                       'semi-hard distances.')
    anchor_positive_mining_distances = tf.expand_dims(
        anchor_positive_mining_distances, axis=-1)
    indicators &= (
        anchor_match_mining_distance_matrix > anchor_positive_mining_distances)

  def find_hard_distances(distance_matrix, indicator_matrix):
    distance_matrix = tf.where(
        tf.stop_gradient(indicator_matrix), distance_matrix,
        tf.fill(tf.shape(distance_matrix), distance_matrix.dtype.max))
    hard_distances = tf.math.reduce_min(distance_matrix, axis=-1)
    return hard_distances

  hard_negative_mining_distances = find_hard_distances(
      anchor_match_mining_distance_matrix, indicators)

  indicators &= tf.math.equal(
      anchor_match_mining_distance_matrix,
      tf.expand_dims(hard_negative_mining_distances, axis=-1))

  hard_negative_distances = find_hard_distances(anchor_match_distance_matrix,
                                                indicators)

  return hard_negative_distances, hard_negative_mining_distances


def compute_hard_negative_triplet_loss(
    anchor_positive_distances,
    anchor_match_distance_matrix,
    anchor_match_negative_indicator_matrix,
    margin,
    use_semi_hard,
    anchor_positive_mining_distances=None,
    anchor_match_mining_distance_matrix=None):
  """Computes triplet loss with (semi-)hard negative mining.

  Args:
    anchor_positive_distances: A tensor for anchor/positive distances. Shape =
      [num_anchors].
    anchor_match_distance_matrix: A tensor for anchor/match distance matrix.
      Shape = [num_anchors, num_matches].
    anchor_match_negative_indicator_matrix: A tensor for anchor/match negative
      indicator matrix. Shape = [num_anchors, num_matches].
    margin: A float for triplet loss margin.
    use_semi_hard: A boolean for whether to compute semi-hard negative distances
      instead of hard negative distances.
    anchor_positive_mining_distances: A tensor for positive distances of each
      anchor for (semi-)hard negative mining. Only used if `use_semi_hard` is
      True. Shape = [num_anchors].
    anchor_match_mining_distance_matrix: A tensor for an alternative
      anchor/match distance matrix to use for (semi-)hard negative mining. Use
      None to ignore and use `anchor_match_distance_matrix` instead. If
      specified, must be of the same shape as `anchor_match_distance_matrix`.

  Returns:
    loss: A tensor for loss. Shape = [].
    num_active_triplets: A tensor for number of active triplets. Shape = [].
    anchor_negative_distances: A tensor for anchor/negative distances. Shape =
      [num_amchors]. If an anchor has no (semi-)hard negative match, its
      negative distance will be assigned as the maximum value of
      anchor_match_distance_matrix.dtype.
    mining_loss: A tensor for loss based on mining distances. Shape = [].
    num_active_mining_triplets: A tensor for number of active triplets based on
      mining distances. Shape = [].
    anchor_negative_mining_distances: A tensor for anchor/negative mining
      distances. Shape = [num_amchors]. If an anchor has no (semi-)hard negative
      match, its negative distance will be assigned as the maximum value of
      anchor_match_mining_distance_matrix.dtype.
  """
  if anchor_positive_mining_distances is None:
    anchor_positive_mining_distances = anchor_positive_distances
  if anchor_match_mining_distance_matrix is None:
    anchor_match_mining_distance_matrix = anchor_match_distance_matrix

  anchor_negative_distances, anchor_negative_mining_distances = (
      compute_hard_negative_distances(
          anchor_match_distance_matrix,
          anchor_match_negative_indicator_matrix,
          use_semi_hard=use_semi_hard,
          anchor_positive_mining_distances=anchor_positive_mining_distances,
          anchor_match_mining_distance_matrix=(
              anchor_match_mining_distance_matrix)))

  def compute_triplet_loss(positive_distances, negative_distances):
    losses = tf.nn.relu(positive_distances + margin - negative_distances)
    losses = tf.where(
        tf.stop_gradient(losses < losses.dtype.max), losses,
        tf.zeros_like(losses))
    num_nonzero_losses = tf.math.count_nonzero(losses)
    loss = tf.math.reduce_mean(losses)
    return loss, num_nonzero_losses

  loss, num_active_triplets = compute_triplet_loss(anchor_positive_distances,
                                                   anchor_negative_distances)
  mining_loss, num_active_mining_triplets = compute_triplet_loss(
      anchor_positive_mining_distances, anchor_negative_mining_distances)

  return (loss, num_active_triplets, anchor_negative_distances, mining_loss,
          num_active_mining_triplets, anchor_negative_mining_distances)


def compute_keypoint_triplet_losses(
    anchor_embeddings,
    positive_embeddings,
    match_embeddings,
    anchor_keypoints,
    match_keypoints,
    margin,
    min_negative_keypoint_distance,
    use_semi_hard,
    exclude_inactive_triplet_loss,
    anchor_keypoint_masks=None,
    match_keypoint_masks=None,
    embedding_sample_distance_fn=create_sample_distance_fn(),
    keypoint_distance_fn=keypoint_utils.compute_procrustes_aligned_mpjpes,
    anchor_mining_embeddings=None,
    positive_mining_embeddings=None,
    match_mining_embeddings=None,
    summarize_percentiles=True):
  """Computes triplet losses with both hard and semi-hard negatives.

  Args:
    anchor_embeddings: A tensor for anchor embeddings. Shape = [num_anchors,
      embedding_dim] or [num_anchors, num_samples, embedding_dim].
    positive_embeddings: A tensor for positive match embeddings. Shape =
      [num_anchors, embedding_dim] or [num_anchors, num_samples, embedding_dim].
    match_embeddings: A tensor for candidate negative match embeddings. Shape =
      [num_anchors, embedding_dim] or [num_matches, num_samples, embedding_dim].
    anchor_keypoints: A tensor for anchor keypoints for computing pair labels.
      Shape = [num_anchors, ..., num_keypoints, keypoint_dim].
    match_keypoints: A tensor for match keypoints for computing pair labels.
      Shape = [num_anchors, ..., num_keypoints, keypoint_dim].
    margin: A float for triplet loss margin.
    min_negative_keypoint_distance: A float for the minimum negative distance
      threshold. If negative, uses all other samples as negative matches. In
      this case, `num_anchors` and `num_matches` are assumed to be equal. Note
      that this option is for saving negative match computation. To support
      different `num_anchors` and `num_matches`, setting this to 0 (without
      saving computation).
    use_semi_hard: A boolean for whether to use semi-hard negative triplet loss
      as the final loss.
    exclude_inactive_triplet_loss: A boolean for whether to exclude inactive
      triplets in the final loss computation.
    anchor_keypoint_masks: A tensor for anchor keypoint masks for computing pair
      labels. Shape = [num_anchors, ..., num_keypoints]. Ignored if None.
    match_keypoint_masks: A tensor for match keypoint masks for computing pair
      labels. Shape = [num_anchors, ..., num_keypoints]. Ignored if None.
    embedding_sample_distance_fn: A function handle for computing sample
      embedding distances, which takes two embedding tensors of shape [...,
      num_samples, embedding_dim] and returns a distance tensor of shape [...].
    keypoint_distance_fn: A function handle for computing keypoint distance
      matrix, which takes two matrix tensors and returns an element-wise
      distance matrix tensor.
    anchor_mining_embeddings: A tensor for anchor embeddings for triplet mining.
      Shape = [num_anchors, embedding_dim] or [num_anchors, num_samples,
      embedding_dim]. Use None to ignore and use `anchor_embeddings` instead.
    positive_mining_embeddings: A tensor for positive match embeddings for
      triplet mining. Shape = [num_anchors, embedding_dim] or [num_anchors,
      num_samples, embedding_dim]. Use None to ignore and use
      `positive_embeddings` instead.
    match_mining_embeddings: A tensor for candidate negative match embeddings
      for triplet mining. Shape = [num_anchors, embedding_dim] or [num_matches,
      num_samples, embedding_dim]. Use None to ignore and use `match_embeddings`
      instead.
    summarize_percentiles: A boolean for whether to summarize percentiles of
      certain variables, e.g., embedding distances in triplet loss. Consider
      turning this off in case tensorflow_probability percentile computation
      causes failures at random due to empty tensor.

  Returns:
    loss: A tensor for triplet loss. Shape = [].
    summaries: A dictionary for loss and batch statistics summaries.
  """

  def maybe_expand_sample_dim(embeddings):
    if len(embeddings.shape.as_list()) == 2:
      return tf.expand_dims(embeddings, axis=-2)
    return embeddings

  anchor_embeddings = maybe_expand_sample_dim(anchor_embeddings)
  positive_embeddings = maybe_expand_sample_dim(positive_embeddings)
  match_embeddings = maybe_expand_sample_dim(match_embeddings)

  if min_negative_keypoint_distance >= 0.0:
    anchor_match_negative_indicator_matrix = (
        compute_negative_indicator_matrix(
            anchor_points=anchor_keypoints,
            match_points=match_keypoints,
            distance_fn=keypoint_distance_fn,
            min_negative_distance=min_negative_keypoint_distance,
            anchor_point_masks=anchor_keypoint_masks,
            match_point_masks=match_keypoint_masks))
  else:
    num_anchors = tf.shape(anchor_keypoints)[0]
    anchor_match_negative_indicator_matrix = tf.math.logical_not(
        tf.eye(num_anchors, dtype=tf.bool))

  anchor_positive_distances = embedding_sample_distance_fn(
      anchor_embeddings, positive_embeddings)

  if anchor_mining_embeddings is None and positive_mining_embeddings is None:
    anchor_positive_mining_distances = anchor_positive_distances
  else:
    anchor_positive_mining_distances = embedding_sample_distance_fn(
        anchor_embeddings if anchor_mining_embeddings is None else
        maybe_expand_sample_dim(anchor_mining_embeddings),
        positive_embeddings if positive_mining_embeddings is None else
        maybe_expand_sample_dim(positive_mining_embeddings))

  anchor_match_distance_matrix = distance_utils.compute_distance_matrix(
      anchor_embeddings,
      match_embeddings,
      distance_fn=embedding_sample_distance_fn)

  if anchor_mining_embeddings is None and match_mining_embeddings is None:
    anchor_match_mining_distance_matrix = anchor_match_distance_matrix
  else:
    anchor_match_mining_distance_matrix = distance_utils.compute_distance_matrix(
        anchor_embeddings if anchor_mining_embeddings is None else
        maybe_expand_sample_dim(anchor_mining_embeddings),
        match_embeddings if match_mining_embeddings is None else
        maybe_expand_sample_dim(match_mining_embeddings),
        distance_fn=embedding_sample_distance_fn)

  num_total_triplets = tf.cast(tf.shape(anchor_embeddings)[0], dtype=tf.float32)

  def compute_loss_and_create_summaries(use_semi_hard):
    """Computes loss and creates summaries."""
    (loss, num_active_triplets, negative_distances, mining_loss,
     num_active_mining_triplets, negative_mining_distances) = (
         compute_hard_negative_triplet_loss(
             anchor_positive_distances,
             anchor_match_distance_matrix,
             anchor_match_negative_indicator_matrix,
             margin=margin,
             use_semi_hard=use_semi_hard,
             anchor_positive_mining_distances=anchor_positive_mining_distances,
             anchor_match_mining_distance_matrix=(
                 anchor_match_mining_distance_matrix)))
    negative_distances = tf.boolean_mask(
        negative_distances,
        mask=negative_distances < negative_distances.dtype.max)
    negative_mining_distances = tf.boolean_mask(
        negative_mining_distances,
        mask=negative_distances < negative_distances.dtype.max)

    active_triplet_ratio = (
        tf.cast(num_active_triplets, dtype=tf.float32) / num_total_triplets)
    active_mining_triplet_ratio = (
        tf.cast(num_active_mining_triplets, dtype=tf.float32) /
        num_total_triplets)

    active_loss = (
        loss / tf.math.maximum(1e-12, tf.stop_gradient(active_triplet_ratio)))
    active_mining_loss = (
        mining_loss /
        tf.math.maximum(1e-12, tf.stop_gradient(active_mining_triplet_ratio)))

    tag = 'SemiHardNegative' if use_semi_hard else 'HardNegative'
    summaries = {
        # Summaries related to triplet loss computation.
        'triplet_loss/Anchor/%s/Distance/Mean' % tag:
            tf.math.reduce_mean(negative_distances),
        'triplet_loss/%s/Loss/All' % tag:
            loss,
        'triplet_loss/%s/Loss/Active' % tag:
            active_loss,
        'triplet_loss/%s/ActiveTripletNum' % tag:
            num_active_triplets,
        'triplet_loss/%s/ActiveTripletRatio' % tag:
            active_triplet_ratio,

        # Summaries related to triplet mining.
        'triplet_mining/Anchor/%s/Distance/Mean' % tag:
            tf.math.reduce_mean(negative_mining_distances),
        'triplet_mining/%s/Loss/All' % tag:
            mining_loss,
        'triplet_mining/%s/Loss/Active' % tag:
            active_mining_loss,
        'triplet_mining/%s/ActiveTripletNum' % tag:
            num_active_mining_triplets,
        'triplet_mining/%s/ActiveTripletRatio' % tag:
            active_mining_triplet_ratio,
    }
    if summarize_percentiles:
      summaries.update({
          'triplet_loss/Anchor/%s/Distance/Median' % tag:
              tfp.stats.percentile(negative_distances, q=50),
          'triplet_mining/Anchor/%s/Distance/Median' % tag:
              tfp.stats.percentile(negative_mining_distances, q=50),
      })

    return loss, active_loss, summaries

  hard_negative_loss, hard_negative_active_loss, hard_negative_summaries = (
      compute_loss_and_create_summaries(use_semi_hard=False))
  (semi_hard_negative_loss, semi_hard_negative_active_loss,
   semi_hard_negative_summaries) = (
       compute_loss_and_create_summaries(use_semi_hard=True))

  summaries = {
      'triplet_loss/Margin':
          tf.constant(margin),
      'triplet_loss/Anchor/Positive/Distance/Mean':
          tf.math.reduce_mean(anchor_positive_distances),
      'triplet_mining/Anchor/Positive/Distance/Mean':
          tf.math.reduce_mean(anchor_positive_mining_distances),
  }
  if summarize_percentiles:
    summaries.update({
        'triplet_loss/Anchor/Positive/Distance/Median':
            tfp.stats.percentile(anchor_positive_distances, q=50),
        'triplet_mining/Anchor/Positive/Distance/Median':
            tfp.stats.percentile(anchor_positive_mining_distances, q=50),
    })
  summaries.update(hard_negative_summaries)
  summaries.update(semi_hard_negative_summaries)

  if use_semi_hard:
    if exclude_inactive_triplet_loss:
      loss = semi_hard_negative_active_loss
    else:
      loss = semi_hard_negative_loss
  else:
    if exclude_inactive_triplet_loss:
      loss = hard_negative_active_loss
    else:
      loss = hard_negative_loss

  return loss, summaries


def compute_kl_regularization_loss(means,
                                   stddevs,
                                   loss_weight,
                                   prior_mean=0.0,
                                   prior_stddev=1.0):
  """Computes KL divergence regularization loss for multivariate Gaussian.

  Args:
    means: A tensor for distribution means. Shape = [..., dim].
    stddevs: A tensor for distribution standard deviations. Shape = [..., dim].
    loss_weight: A float for loss weight.
    prior_mean: A float for prior distribution mean.
    prior_stddev: A float for prior distribution standard deviation.

  Returns:
    loss: A tensor for weighted regularization loss. Shape = [].
    summaries: A dictionary for loss summaries.
  """
  loss = tf.math.reduce_mean(
      distance_utils.compute_gaussian_kl_divergence(
          means, stddevs, rhs_means=prior_mean, rhs_stddevs=prior_stddev))
  weighted_loss = loss_weight * loss
  summaries = {
      'regularization_loss/KL/PriorMean/Mean':
          tf.math.reduce_mean(tf.constant(prior_mean)),
      'regularization_loss/KL/PriorVar/Mean':
          tf.math.reduce_mean(tf.constant(prior_stddev)**2),
      'regularization_loss/KL/Loss/Original':
          loss,
      'regularization_loss/KL/Loss/Weighted':
          weighted_loss,
      'regularization_loss/KL/Loss/Weight':
          tf.constant(loss_weight),
  }
  return weighted_loss, summaries


def compute_positive_pairwise_loss(anchor_embeddings,
                                   positive_embeddings,
                                   loss_weight,
                                   distance_fn=functools.partial(
                                       distance_utils.compute_l2_distances,
                                       squared=True)):
  """Computes anchor/positive pairwise (squared L2) loss.

  Args:
    anchor_embeddings: A tensor for anchor embeddings. Shape = [...,
      embedding_dim].
    positive_embeddings: A tensor for positive embeddings. Shape = [...,
      embedding_dim].
    loss_weight: A float for loss weight.
    distance_fn: A function handle for computing embedding distances, which
      takes two embedding tensors of shape [..., embedding_dim] and returns a
      distance tensor of shape [...].

  Returns:
    loss: A tensor for weighted positive pairwise loss. Shape = [].
    summaries: A dictionary for loss summaries.
  """
  loss = tf.math.reduce_mean(
      distance_fn(anchor_embeddings, positive_embeddings))
  weighted_loss = loss_weight * loss
  summaries = {
      'pairwise_loss/PositivePair/Loss/Original': loss,
      'pairwise_loss/PositivePair/Loss/Weighted': weighted_loss,
      'pairwise_loss/PositivePair/Loss/Weight': tf.constant(loss_weight),
  }
  return weighted_loss, summaries
