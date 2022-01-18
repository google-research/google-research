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

"""Implements contrastive losses over features from multiple views of data."""

import six

import tensorflow.compat.v2 as tf
from tensorflow.keras import backend as K
from supcon import enums
from supcon import utils


# Derived wholesale from (unexported) LossFunctionWrapper in keras losses.py
class LossFunctionWrapper(tf.keras.losses.Loss):
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               fn,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None,
               **kwargs):
    """Initializes `LossFunctionWrapper` class.

    Args:
      fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial]
        (https://www.tensorflow.org/tutorials/distribute/custom_training) for
          more details.
      name: (Optional) name for the loss.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    return self.fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if K.is_tensor or K.is_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ContrastiveLoss(LossFunctionWrapper):
  """Contrastive Loss for keras models.

  Attributes:
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of
      built-in training loops such as `tf.keras` `compile` and `fit`, using
      `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see this
      custom training [tutorial]
      (https://www.tensorflow.org/tutorials/distribute/custom_training) for more
        details.
    name: Optional name for the op.
    **kwargs: Extra key-word arguments for the loss. See `contrastive_loss` for
      valid arguments for **kwargs
  """

  def __init__(self,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='contrastive_loss',
               **kwargs):
    super(ContrastiveLoss, self).__init__(
        fn=self.contrastive_loss_function,
        name=name,
        reduction=reduction,
        **kwargs)

  def contrastive_loss_function(self, y_true, y_pred, **kwargs):
    return contrastive_loss(y_pred, labels=y_true, **kwargs)


def _validate_contrastive_loss_inputs(features, labels, contrast_mode,
                                      summation_location, denominator_mode,
                                      positives_cap):
  r"""Validates inputs for contrastive_loss().

  Args:
    features: Tensor of rank at least 3, where the first 2 dimensions are
      batch_size and num_views, and the remaining dimensions are the feature
      shape.
    labels: One-hot labels tensor of shape [batch_size, num_labels] with numeric
      dtype.
    contrast_mode: LossContrastMode specifying which views get used as anchors.
    summation_location: LossSummationLocation specifying location of positives
      summation. See documentation above for more details.
    denominator_mode: LossDenominatorMode specifying which positives to include
      in contrastive denominator. See documentation above for more details.
    positives_cap: Integer maximum number of positives *other* than
      augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
      Including augmentations, a maximum of (positives_cap + num_views - 1)
      positives is possible. This parameter modifies the contrastive numerator
      by selecting which positives are present in the summation, and which
      positives contribure to the denominator if denominator_mode ==
      enums.LossDenominatorMode.ALL.

  Returns:
    Tuple containing batch_size and num_views values.

  Raises:
    ValueError if any of the inputs are invalid.
  """
  if features.shape.rank < 3:
    raise ValueError(
        f'Invalid features rank ( = {features.shape.rank}). Should have rank '
        '>= 3 with shape [batch_size, num_views] + `feature shape.`')

  batch_size = tf.compat.dimension_at_index(features.shape, 0).value
  if batch_size is None:
    raise ValueError('features has unknown batch_size dimension.')
  num_views = tf.compat.dimension_at_index(features.shape, 1).value
  if num_views is None:
    raise ValueError('features has unknown num_views dimension.')

  if labels is not None:
    # Check that |labels| are shaped like a one_hot vector.
    if labels.shape.rank != 2 or labels.shape[0] != batch_size:
      raise ValueError(
          f'Invalid labels shape (= {labels.shape}). Should have shape '
          f'(batch_size = {batch_size}, num_labels).')

  if not isinstance(contrast_mode, enums.LossContrastMode):
    raise ValueError(
        f'Invalid contrast_mode (= {contrast_mode}). Should be an instance of '
        'LossContrastMode.')
  if not isinstance(summation_location, enums.LossSummationLocation):
    raise ValueError(
        f'Invalid summation_location (= {summation_location}). Should be an '
        'instance of LossSummationLocation.')
  if not isinstance(denominator_mode, enums.LossDenominatorMode):
    raise ValueError(
        f'Invalid denominator_mode (= {denominator_mode}). Should be an '
        'instance of LossDenominatorMode.')
  if positives_cap > -1 and positives_cap % num_views != 0:
    raise ValueError(
        f'positives_cap (= {positives_cap}) must be a multiple of the '
        f'num_views (= {num_views}).')

  return batch_size, num_views


def _cap_positives_mask(untiled_mask, diagonal_mask, num_views, positives_cap):
  r"""Cap positives in the provided untiled_mask.

      'positives_cap' specifies the maximum number of positives *other* than
      augmentations of the anchor. Positives will be evenly sampled from all
      views.

  Args:
    untiled_mask: Tensor of shape [local_batch_size, global_batch_size] that has
      entry (r, c) == 1 if feature entries in rows r and c are from the same
      class. Else (r, c) == 0.
    diagonal_mask: Tensor with the same shape as `untiled_mask`. When
      local_batch_size == global_batch_size this is just an identity matrix.
      Otherwise, it is an identity matrix of size `local_batch_size` that is
      padded with 0's in the 2nd dimension to match the target shape. This is
      used to indicate where the anchor views exist in the global batch of
      views.
    num_views: Integer number of total views.
    positives_cap: Integer maximum number of positives *other* than
      augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
      Including augmentations, a maximum of (positives_cap + num_views - 1)
      positives is possible. This parameter modifies the contrastive numerator
      by selecting which positives are present in the summation, and which
      positives contribure to the denominator if denominator_mode ==
      enums.LossDenominatorMode.ALL.

  Returns:
    A tf.Tensor with the modified `untiled_mask`.
  """
  untiled_mask_no_diagonal = tf.math.minimum(untiled_mask, 1. - diagonal_mask)
  untiled_positives_per_anchor = positives_cap // num_views

  # Grabs top-k positives from each row in the mask. Can end up with negatives
  # incorrectly marked as positives if fewer than `untiled_positives_per_anchor`
  # exist in any row of `untiled_mask_no_diagonal`. However, these false
  # positives wil be masked out before the function returns.
  _, top_k_col_idx = tf.math.top_k(untiled_mask_no_diagonal,
                                   untiled_positives_per_anchor)
  top_k_row_idx = tf.expand_dims(tf.range(tf.shape(untiled_mask)[0]), axis=1)

  # Construct |top_k_idx|, a tensor of shape
  # [untiled_positives_per_anchor * local_batch_size, 2]. Each row represents
  # the 2D index in a
  # [local_batch_size * num_anchor_views, global_batch_size * num_views] size
  # tensor which holds a positive; all others are negatives.
  top_k_idx = tf.reshape(
      tf.stack([
          tf.tile(top_k_row_idx,
                  (1, untiled_positives_per_anchor)), top_k_col_idx
      ],
               axis=-1), (-1, 2))

  # Construct |untiled_mask|. Sets positives to 1 according to top_k_idx
  # above.
  untiled_mask_capped = tf.scatter_nd(
      top_k_idx,
      tf.ones(
          shape=tf.shape(top_k_idx)[0], dtype=untiled_mask_no_diagonal.dtype),
      untiled_mask_no_diagonal.shape)
  untiled_mask_capped = tf.math.maximum(untiled_mask_capped, diagonal_mask)
  return untiled_mask * untiled_mask_capped


def _create_tiled_masks(untiled_mask, diagonal_mask, num_views,
                        num_anchor_views, positives_cap):
  r"""Creates tiled versions of untiled mask.

  Tiles `untiled_mask`, which has shape [local_batch_size, global_batch_size]
  by factors of [num_anchor_views, num_views], and then generates two masks from
  it. In both cases, the mask dimensions are ordered by view and then by sample,
  so if there was a batch size of 3 with 2 views the order would be
  [b1v1, b2v1, b3v1, b1v2, b2v2, b3v2]:
    positives_mask: Entry (row = i, col = j) is 1 if
      untiled_mask[i % local_batch_size, j % global_batch_size] == 1 and
      i // local_batch_size != j // global_batch_size. This results in a mask
      that is 1 for all pairs that are the same class but are not the exact same
      view. An exception to this is if positives_cap > -1, in which case there
      is a maximum of (positives_cap) 1-values per row, not including the
      entries that correspond to other views of the anchor. That is,
      positives_cap does nothing if there is only a single 1-valued entry per
      row in `untiled_mask`.
    negatives_mask: Entry (row = i, col = j) is 1 if features i and j are
      different classes. Otherwise the entry is 0.

  Args:
    untiled_mask: Tensor of shape [local_batch_size, global_batch_size], where
      local_batch_size <= global_batch_size, that has entry (r, c) == 1 if
      feature entries in rows r and c are from the same class. Else (r, c) == 0.
      In the self-supervised case, where the only positives are other views of
      the same sample, `untiled_mask` and `diagonal_mask` should be the same.
    diagonal_mask: Tensor with the same shape as `untiled_mask`. When
      local_batch_size == global_batch_size this is just an identity matrix.
      Otherwise, it is a slice of a [global_batch_size, global_batch_size]
      identity matrix that indicates where in the global batch the local batch
      is located.
    num_views: Integer number of total views.
    num_anchor_views: Integer number of anchor views.
    positives_cap: Integer maximum number of positives *other* than
      augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
      Including augmentations, a maximum of (positives_cap + num_views - 1)
      positives is possible. This parameter modifies the contrastive numerator
      by selecting which positives are present in the summation, and which
      positives contribure to the denominator if denominator_mode ==
      enums.LossDenominatorMode.ALL.

  Returns:
    Tuple containing positives_mask and negatives_mask tensors.
  """
  global_batch_size = tf.shape(untiled_mask)[1]
  # Generate |all_but_diagonal_mask|, a tensor of shape
  # [local_batch_size * num_anchor_views, global_batch_size * num_views] where
  # entry (row = i, column = j) is 0 for cases where the anchor view
  # corresponding to row i is the same as the view corresponding to column j. In
  # the case where local_batch_size == global_batch_size and
  # num_anchor_views == num_views, this is just 1 - identity_matrix.
  labels = tf.argmax(diagonal_mask, axis=-1)
  tiled_labels = []
  for i in range(num_anchor_views):
    tiled_labels.append(labels + tf.cast(global_batch_size, labels.dtype) * i)
  tiled_labels = tf.concat(tiled_labels, axis=0)
  tiled_diagonal_mask = tf.one_hot(tiled_labels, global_batch_size * num_views)
  all_but_diagonal_mask = 1. - tiled_diagonal_mask

  # Construct |negatives_mask| and |uncapped_positives_mask|, both tensors of
  # shape [local_batch_size * num_anchor_views, global_batch_size * num_views].
  # |uncapped_positives_mask| are all positive candidates, including the
  # diagonal representing the anchor view itself, before the capping procedure.
  # Any element that is not an `uncapped` positive is a negative.
  uncapped_positives_mask = tf.tile(untiled_mask, [num_anchor_views, num_views])

  negatives_mask = 1. - uncapped_positives_mask

  # Select only 'positives_cap' positives by selecting top-k values of 0/1 mask
  # and scattering ones into those indices. This capping is done on only
  # non-diagonal positives.
  if positives_cap > -1:
    untiled_mask = _cap_positives_mask(untiled_mask, diagonal_mask, num_views,
                                       positives_cap)
    # Construct |positives_mask|, a tensor of shape
    # [local_batch_size * num_anchor_views, global_batch_size * num_views].
    # Entry (r,c) is 1 iff
    # untiled_mask[r % local_batch_size, c % global_batch_size] == 1 *and*
    # r != c.
    # Else it is zero.
    positives_mask = tf.tile(untiled_mask, [num_anchor_views, num_views])
  else:
    positives_mask = uncapped_positives_mask

  positives_mask = positives_mask * all_but_diagonal_mask  # Zero the diagonal.

  return positives_mask, negatives_mask


def contrastive_loss(features,
                     labels=None,
                     temperature=1.0,
                     contrast_mode=enums.LossContrastMode.ALL_VIEWS,
                     summation_location=enums.LossSummationLocation.OUTSIDE,
                     denominator_mode=enums.LossDenominatorMode.ALL,
                     positives_cap=-1,
                     scale_by_temperature=True):
  r"""Contrastive loss over features.

  Implemented as described in: https://arxiv.org/abs/2004.11362, Equation 2.

  Given `num_views` different views of each of `batch_size` samples, let `f_i`
  (i \in [1, 2 ... (num_views * batch_size)]) denote each respective feature
  vector. The contrastive loss then takes the following form:

    L = \sum_{i} L_i

  where each L_i is computed as:

    L_i = -\tau * \sum_{k \in P(i)} \log(p_{ik})    (1)

  where P(i) is the set of positives for entry i (distinct from i) and where:

                       \exp(f_i^T f_k / \tau)
    p_{ik} = ----------------------------------------                        (2)
             \sum_{j \in A(i)} \exp(f_i^T f_j / \tau)

  where A(i) is the set of all positives or negatives (distinct from i). `i` is
  the anchor, and \tau is the temperature.

  This maximizes the likelihood of a given (anchor, positive) pair with
  respect to all possible pairs where the first member is the anchor and the
  second member is a positive or a negative.

  A typical way to define a positive is to define samples from the
  same class (but not the anchor itself) regardless of what view they are from.
  Similarly, a typical way to define a negative is for it to be any view of a
  sample from a different class.

  There are two ways to define which feature pairs should be treated as
  positives and negatives. All views of the same sample are always treated as
  positives. You can declare other samples to be positives by providing `labels`
  such that all samples with the same label will be positives for each other.

  If `labels` is not provided then we default to every sample belonging to its
  own unique class. Therefore, the only positive used is another view of the
  anchor itself. This implements the loss as described in:

    https://arxiv.org/pdf/2002.05709.pdf
    A Simple Framework for Contrastive Learning of Visual Representations
    Chen T., Kornblith S., Norouzi M., Hinton G.

  It is recommended to use features whose L_2 norm is 1. since that ensures
  that the loss does not return NaN values without changing the intended
  behaviour of the loss function.

  In (1) above, note that the summation over positives is located outside of the
  \log(). However, one can permute these two operations. The result is Eq. 3 in
  https://arxiv.org/abs/2004.11362. Users can specify the location of the
  summation relative to the \log() via the `summation_location' argmument:
   - 'out': Eq. 2 in https://arxiv.org/abs/2004.11362.
   - 'in' : Eq. 3 in https://arxiv.org/abs/2004.11362.

  Additionally, in (2) above, note that the denominator sums over *all* entries
  distinct from i. One can change which terms are included in the denominator
  via the `denominator_mode` argument:
   - LossDenominatorMode.ALL : All entries (i.e., all negatives and all
             positives) distinct from i are included.
   - LossDenominatorMode.ONE_POSITIVE : All negatives are included but only the
             single positive in the numerator of (2) is included. Any other
             positives are excluded.
   - LossDenominatorMode.ONLY_NEGATIVES: All negatives are included but no
             positives are, not even the single positive in the numerator of
             (2).

  On TPUs, this method will internally perform the cross-replica operations that
  enable using the samples from all cores in computing the loss. The inputs to
  this function should be the features and labels from a single core and each
  core will compute the loss using just these features as anchors, but will use
  positives and negatives from the full global batch. Since the loss for each
  anchor is only computed on one TPU core, it's still necessary to have a
  cross-replica reduction in the final loss computation.

  Also, though it is not applicable to multiview contrastive learning, this
  function will work if |features| contains only 1 view. In the high batch size
  limit, the implemented contrastive loss with only 1 view, positives_cap = 1,
  and temperature = 1.0 is equivalent to the N-pairs loss
  (https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)

  Args:
    features: A Tensor of rank at least 3, where the first 2 dimensions are
      batch_size and num_views, and the remaining dimensions are the feature
      shape. Note that when running on TPU, batch_size is the per-core batch
      size.
    labels: One-hot labels to be used to construct the supervised contrastive
      loss. Samples with the same labels are used as positives for each other.
      Labels must have shape [batch_size, num_labels] with numeric dtype and be
      0-1 valued. Note that when running on TPU, batch_size is the per-core
      batch size.
    temperature: Temperature at which softmax evaluation is done. Temperature
      must be a python scalar or scalar Tensor of numeric dtype.
    contrast_mode: LossContrastMode specifying which views get used as anchors
      (f_i in the expression above)
      'ALL_VIEWS': All the views of all samples are used as anchors (f_i in the
        expression above).
      'ONE_VIEW': Just the first view of each sample is used as an anchor (f_i
        in the expression above). This view is called the `core` view against
        which other views are contrasted.
    summation_location: LossSummationLocation specifying location of positives
      summation. See documentation above for more details.
    denominator_mode: LossDenominatorMode specifying which positives to include
      in contrastive denominator. See documentation above for more details.
    positives_cap: Integer maximum number of positives *other* than
      augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
      Including augmentations, a maximum of (positives_cap + num_views - 1)
      positives is possible. This parameter modifies the contrastive numerator
      by selecting which positives are present in the summation, and which
      positives contribure to the denominator if denominator_mode ==
      enums.LossDenominatorMode.ALL.
    scale_by_temperature: Boolean. Whether to scale the loss by `temperature`.
      The loss gradient naturally has a 1/temperature scaling factor, so this
      counteracts it.

  Returns:
    Scalar tensor with contrastive loss value with shape [batch_size] and dtype
    tf.float32. The loss for each batch element is the mean over all views.

  Raises:
    ValueError if the shapes of any of the Tensors are unexpected, or if both
    `labels` and `mask` are not `None`.
  """
  features = tf.convert_to_tensor(features)
  labels = tf.convert_to_tensor(labels) if labels is not None else None

  local_batch_size, num_views = _validate_contrastive_loss_inputs(
      features, labels, contrast_mode, summation_location, denominator_mode,
      positives_cap)

  # Flatten `features` to a single dimension per view per sample so it has shape
  # [local_batch_size, num_views, num_features].
  if features.shape.rank > 3:
    features = tf.reshape(features,
                          tf.concat([tf.shape(features)[:2], [-1]], axis=0),
                          'flattened_features')
  if features.dtype != tf.float32:
    features = tf.cast(features, tf.float32)

  # Grab the features from all TPU cores. We use the local batch as anchors and
  # the full global batch as contrastives. If not on TPU, global_features is the
  # same as features.
  global_features = utils.cross_replica_concat(features)
  global_batch_size = tf.compat.dimension_at_index(global_features.shape,
                                                   0).value
  local_replica_id = utils.local_tpu_replica_id()

  # Generate the [local_batch_size, global_batch_size] slice of the
  # [global_batch_size, global_batch_size] identity matrix that corresponds to
  # the current replica.
  diagonal_mask = tf.one_hot(
      tf.range(local_batch_size) + (local_replica_id * local_batch_size),
      global_batch_size)

  # Generate `mask` with shape [local_batch_size, global_batch_size] that
  # indicates which samples should be considered positives for each other.
  if labels is None:
    # Defaults to every sample belonging to its own unique class, containing
    # just that sample and other views of it.
    mask = diagonal_mask
  else:
    labels = tf.cast(labels, tf.float32)  # TPU matmul op unsupported for ints.
    global_labels = utils.cross_replica_concat(labels)
    mask = tf.linalg.matmul(labels, global_labels, transpose_b=True)
  mask = tf.ensure_shape(mask, [local_batch_size, global_batch_size])

  # To streamline the subsequent TF, the first two dimensions of
  # `global_features` (i.e., global_batch_size and num_views) should be
  # transposed and then flattened. The result has shape
  # [num_views * global_batch_size, num_features], and its first dimension
  # elements are grouped by view, not by sample.
  all_global_features = tf.reshape(
      tf.transpose(global_features, perm=[1, 0, 2]),
      [num_views * global_batch_size, -1])

  if contrast_mode == enums.LossContrastMode.ONE_VIEW:
    anchor_features = features[:, 0]
    num_anchor_views = 1
  else:  # contrast_mode == enums.LossContrastMode.ALL_VIEWS
    # Reshape features to match how global_features is reshaped above.
    anchor_features = tf.reshape(
        tf.transpose(features, perm=[1, 0, 2]),
        [num_views * local_batch_size, -1])
    num_anchor_views = num_views

  # Generate `logits`, the tensor of (temperature-scaled) dot products of the
  # anchor features with all features. It has shape
  # [local_batch_size * num_anchor_views, global_batch_size * num_views]. To
  # improve numerical stability, subtract out the largest |logits| element in
  # each row from all elements in that row. Since |logits| is only ever used as
  # a ratio of exponentials of |logits| values, this subtraction does not change
  # the results correctness. A stop_gradient() is needed because this change is
  # just for numerical precision.
  logits = tf.linalg.matmul(
      anchor_features, all_global_features, transpose_b=True)
  temperature = tf.cast(temperature, tf.float32)
  logits = logits / temperature
  logits = (
      logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True))
  exp_logits = tf.exp(logits)

  # The following masks are all tiled by the number of views, i.e., they have
  # shape [local_batch_size * num_anchor_views, global_batch_size * num_views].
  positives_mask, negatives_mask = (
      _create_tiled_masks(mask, diagonal_mask, num_views, num_anchor_views,
                          positives_cap))
  num_positives_per_row = tf.reduce_sum(positives_mask, axis=1)

  if denominator_mode == enums.LossDenominatorMode.ALL:
    denominator = tf.reduce_sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(
            exp_logits * positives_mask, axis=1, keepdims=True)
  elif denominator_mode == enums.LossDenominatorMode.ONE_POSITIVE:
    denominator = exp_logits + tf.reduce_sum(
        exp_logits * negatives_mask, axis=1, keepdims=True)
  else:  # denominator_mode == enums.LossDenominatorMode.ONLY_NEGATIVES
    denominator = tf.reduce_sum(
        exp_logits * negatives_mask, axis=1, keepdims=True)

  # Note that num_positives_per_row can be zero only if 1 view is used. The
  # various tf.math.divide_no_nan() calls below are to handle this case.
  if summation_location == enums.LossSummationLocation.OUTSIDE:
    log_probs = (logits - tf.math.log(denominator)) * positives_mask
    log_probs = tf.reduce_sum(log_probs, axis=1)
    log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
  else:  # summation_location == enums.LossSummationLocation.INSIDE
    log_probs = exp_logits / denominator * positives_mask
    log_probs = tf.reduce_sum(log_probs, axis=1)
    log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
    log_probs = tf.math.log(log_probs)

  loss = -log_probs
  if scale_by_temperature:
    loss *= temperature
  loss = tf.reshape(loss, [num_anchor_views, local_batch_size])

  if num_views != 1:
    loss = tf.reduce_mean(loss, axis=0)
  else:
    # The 1 view case requires special handling bc, unlike in the > 1 view case,
    # not all samples are guaranteed to have a positive. Also, no reduction over
    # views is needed.
    num_valid_views_per_sample = (
        tf.reshape(num_positives_per_row, [1, local_batch_size]))
    loss = tf.squeeze(tf.math.divide_no_nan(loss, num_valid_views_per_sample))

  return loss
