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

"""Clustering metrics."""

from typing import  Optional, Sequence, Union

from clu import metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np

Ndarray = Union[np.ndarray, jnp.ndarray]


def check_shape(x, expected_shape, name):
  """Check whether shape x is as expected.

  Args:
    x: Any data type with `shape` attribute. If `shape` attribute is not present
      it is assumed to be a scalar with shape ().
    expected_shape: The shape that is expected of x. For example,
      [None, None, 3] can be the `expected_shape` for a color image,
      [4, None, None, 3] if we know that batch size is 4.
    name: Name of `x` to provide informative error messages.

  Raises: ValueError if x's shape does not match expected_shape. Also raises
    ValueError if expected_shape is not a list or tuple.
  """
  if not isinstance(expected_shape, (list, tuple)):
    raise ValueError(
        "expected_shape should be a list or tuple of ints but got "
        f"{expected_shape}.")

  # Scalars have shape () by definition.
  shape = getattr(x, "shape", ())

  if (len(shape) != len(expected_shape) or
      any(j is not None and i != j for i, j in zip(shape, expected_shape))):
    raise ValueError(
        f"Input {name} had shape {shape} but {expected_shape} was expected.")


def _validate_inputs(predicted_segmentations,
                     ground_truth_segmentations,
                     padding_mask,
                     mask = None):
  """Checks that all inputs have the expected shapes.

  Args:
    predicted_segmentations: An array of integers of shape [bs, seq_len, H, W]
      containing model segmentation predictions.
    ground_truth_segmentations: An array of integers of shape [bs, seq_len, H,
      W] containing ground truth segmentations.
    padding_mask: An array of integers of shape [bs, seq_len, H, W] defining
      regions where the ground truth is meaningless, for example because this
      corresponds to regions which were padded during data augmentation. Value 0
      corresponds to padded regions, 1 corresponds to valid regions to be used
      for metric calculation.
    mask: An optional array of boolean mask values of shape [bs]. `True`
      corresponds to actual batch examples whereas `False` corresponds to
      padding.

  Raises:
    ValueError if the inputs are not valid.
  """

  check_shape(
      predicted_segmentations, [None, None, None, None],
      "predicted_segmentations [bs, seq_len, h, w]")
  check_shape(
      ground_truth_segmentations, [None, None, None, None],
      "ground_truth_segmentations [bs, seq_len, h, w]")
  check_shape(
      predicted_segmentations, ground_truth_segmentations.shape,
      "predicted_segmentations [should match ground_truth_segmentations]")
  check_shape(
      padding_mask, ground_truth_segmentations.shape,
      "padding_mask [should match ground_truth_segmentations]")

  if not jnp.issubdtype(predicted_segmentations.dtype, jnp.integer):
    raise ValueError("predicted_segmentations has to be integer-valued. "
                     "Got {}".format(predicted_segmentations.dtype))

  if not jnp.issubdtype(ground_truth_segmentations.dtype, jnp.integer):
    raise ValueError("ground_truth_segmentations has to be integer-valued. "
                     "Got {}".format(ground_truth_segmentations.dtype))

  if not jnp.issubdtype(padding_mask.dtype, jnp.integer):
    raise ValueError("padding_mask has to be integer-valued. "
                     "Got {}".format(padding_mask.dtype))

  if mask is not None:
    check_shape(mask, [None], "mask [bs]")
    if not jnp.issubdtype(mask.dtype, jnp.bool_):
      raise ValueError("mask has to be boolean. Got {}".format(mask.dtype))


def adjusted_rand_index(true_ids, pred_ids,
                        num_instances_true, num_instances_pred,
                        padding_mask = None,
                        ignore_background = False):
  """Computes the adjusted Rand index (ARI), a clustering similarity score.

  Args:
    true_ids: An integer-valued array of shape
      [batch_size, seq_len, H, W]. The true cluster assignment encoded
      as integer ids.
    pred_ids: An integer-valued array of shape
      [batch_size, seq_len, H, W]. The predicted cluster assignment
      encoded as integer ids.
    num_instances_true: An integer, the number of instances in true_ids
      (i.e. max(true_ids) + 1).
    num_instances_pred: An integer, the number of instances in true_ids
      (i.e. max(pred_ids) + 1).
    padding_mask: An array of integers of shape [batch_size, seq_len, H, W]
        defining regions where the ground truth is meaningless, for example
        because this corresponds to regions which were padded during data
        augmentation. Value 0 corresponds to padded regions, 1 corresponds to
        valid regions to be used for metric calculation.
    ignore_background: Boolean, if True, then ignore all pixels where
      true_ids == 0 (default: False).

  Returns:
    ARI scores as a float32 array of shape [batch_size].

  References:
    Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
      https://link.springer.com/article/10.1007/BF01908075
    Wikipedia
      https://en.wikipedia.org/wiki/Rand_index
    Scikit Learn
      http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
  """
  # pylint: disable=invalid-name
  true_oh = jax.nn.one_hot(true_ids, num_instances_true)
  pred_oh = jax.nn.one_hot(pred_ids, num_instances_pred)
  if padding_mask is not None:
    true_oh = true_oh * padding_mask[Ellipsis, None]
    # pred_oh = pred_oh * padding_mask[..., None]  # <--  not needed

  if ignore_background:
    true_oh = true_oh[Ellipsis, 1:]  # Remove the background row.

  N = jnp.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
  A = jnp.sum(N, axis=-1)  # row-sum  (batch_size, c)
  B = jnp.sum(N, axis=-2)  # col-sum  (batch_size, k)
  num_points = jnp.sum(A, axis=1)

  rindex = jnp.sum(N * (N - 1), axis=[1, 2])
  aindex = jnp.sum(A * (A - 1), axis=1)
  bindex = jnp.sum(B * (B - 1), axis=1)
  expected_rindex = aindex * bindex / jnp.clip(num_points * (num_points-1), 1)
  max_rindex = (aindex + bindex) / 2
  denominator = max_rindex - expected_rindex
  ari = (rindex - expected_rindex) / denominator

  # There are two cases for which the denominator can be zero:
  # 1. If both label_pred and label_true assign all pixels to a single cluster.
  #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
  # 2. If both label_pred and label_true assign max 1 point to each cluster.
  #    (max_rindex == expected_rindex == rindex == 0)
  # In both cases, we want the ARI score to be 1.0:
  return jnp.where(denominator, ari, 1.0)


@flax.struct.dataclass
class Ari(metrics.Average):
  """Adjusted Rand Index (ARI) computed from predictions and labels.

  ARI is a similarity score to compare two clusterings. ARI returns values in
  the range [-1, 1], where 1 corresponds to two identical clusterings (up to
  permutation), i.e. a perfect match between the predicted clustering and the
  ground-truth clustering. A value of (close to) 0 corresponds to chance.
  Negative values corresponds to cases where the agreement between the
  clusterings is less than expected from a random assignment.

  In this implementation, we use ARI to compare predicted instance segmentation
  masks (including background prediction) with ground-truth segmentation
  annotations.
  """

  @classmethod
  def from_model_output(cls,
                        predicted_segmentations,
                        ground_truth_segmentations,
                        padding_mask,
                        ground_truth_max_num_instances,
                        predicted_max_num_instances,
                        ignore_background = False,
                        mask = None,
                        **_):
    """Computation of the ARI clustering metric.

    NOTE: This implementation does not currently support padding masks.

    Args:
      predicted_segmentations: An array of integers of shape
        [bs, seq_len, H, W] containing model segmentation predictions.
      ground_truth_segmentations: An array of integers of shape
        [bs, seq_len, H, W] containing ground truth segmentations.
      padding_mask: An array of integers of shape [bs, seq_len, H, W]
        defining regions where the ground truth is meaningless, for example
        because this corresponds to regions which were padded during data
        augmentation. Value 0 corresponds to padded regions, 1 corresponds to
        valid regions to be used for metric calculation.
      ground_truth_max_num_instances: Maximum number of instances (incl.
        background, which counts as the 0-th instance) possible in the dataset.
      predicted_max_num_instances: Maximum number of predicted instances (incl.
        background).
      ignore_background: If True, then ignore all pixels where
        ground_truth_segmentations == 0 (default: False).
      mask: An optional array of boolean mask values of shape [bs]. `True`
        corresponds to actual batch examples whereas `False` corresponds to
        padding.

    Returns:
      Object of Ari with computed intermediate values.
    """
    _validate_inputs(
        predicted_segmentations=predicted_segmentations,
        ground_truth_segmentations=ground_truth_segmentations,
        padding_mask=padding_mask,
        mask=mask)

    batch_size = predicted_segmentations.shape[0]

    if mask is None:
      mask = jnp.ones(batch_size, dtype=padding_mask.dtype)
    else:
      mask = jnp.asarray(mask, dtype=padding_mask.dtype)

    ari_batch = adjusted_rand_index(
        pred_ids=predicted_segmentations,
        true_ids=ground_truth_segmentations,
        num_instances_true=ground_truth_max_num_instances,
        num_instances_pred=predicted_max_num_instances,
        padding_mask=padding_mask,
        ignore_background=ignore_background)
    return cls(total=jnp.sum(ari_batch * mask), count=jnp.sum(mask))  # pylint: disable=unexpected-keyword-arg


@flax.struct.dataclass
class AriNoBg(Ari):
  """Adjusted Rand Index (ARI), ignoring the ground-truth background label."""

  @classmethod
  def from_model_output(cls, **kwargs):
    """See `Ari` docstring for allowed keyword arguments."""
    return super().from_model_output(**kwargs, ignore_background=True)
