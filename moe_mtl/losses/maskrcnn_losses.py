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


"""Mask RCNN losses.

This is a JAX reimplementation of:
third_party/cloud_tpu/models/detection/modeling/losses.py

as well as the build_loss function in:
third_party/cloud_tpu/models/detection/modeling/maskrcnn_model.py
"""

from typing import Dict, Union, Any
from absl import logging
import gin
import jax
import jax.numpy as jnp
import optax
Array = jnp.ndarray
LevelArray = Dict[int, Array]
NamedLevelArray = Dict[str, Union[Array, LevelArray]]


def safe_divide(x,
                y,
                rtol = 1e-5,
                atol = 1e-8):
  """Computes a safe divide which returns 0 if the denominator is zero.

  Reference:
  https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan
  Args:
    x: A float of numerator.
    y: A float of denominator.
    rtol: The relative tolerance parameter. See numpy.isclose for more info.
    atol: The absolute tolerance parameter. See numpy.isclose for more info.

  Returns:
    z: output x / y or 0.
  """
  is_zero = jnp.isclose(y, 0.0, rtol=rtol, atol=atol)
  safe_y = jnp.where(is_zero, jnp.ones_like(y), y)
  return jnp.where(is_zero, jnp.zeros_like(x), x / safe_y)


def compute_weighted_loss(
    loss,
    weights,
    dtype,
    loss_reduction,
):
  """Weights and reduces the loss (borrowing VAL's implementation)."""
  if loss_reduction == "RETURN_AS_IS":
    # Handle no loss reduction, by returning tensor as-is.
    return loss
  loss = loss.astype(jnp.float32)
  loss_weight = jnp.broadcast_to(weights, loss.shape).astype(jnp.float32)
  loss *= loss_weight
  total_loss = jnp.sum(loss)

  if loss_reduction == "SUM_BY_NONZERO_WEIGHTS":
    total_loss = safe_divide(total_loss, jnp.sum(loss_weight != 0.0) + 0.01)
  elif loss_reduction == "MEAN":
    total_loss = safe_divide(total_loss, jnp.sum(loss_weight))
  elif loss_reduction != "SUM":
    raise NotImplementedError("LossReductionType not supported for this loss:"
                              f"{loss_reduction}.")

  return total_loss.astype(dtype)


def sigmoid_cross_entropy(logits,
                          labels,
                          weights = 1.0,
                          loss_reduction = "SUM_BY_NONZERO_WEIGHTS",
                          **kwargs):
  """Returns the sigmoid cross entropy loss. Borrowing VAL's implementation."""
  del kwargs
  # check_shape_equal(logits, labels)
  logits = logits.astype(jnp.float32)
  labels = labels.astype(jnp.float32)
  loss = optax.sigmoid_binary_cross_entropy(logits, labels)
  return compute_weighted_loss(loss, weights, logits.dtype, loss_reduction)


def weighted_softmax_cross_entropy(
    logits,
    labels,
    label_smoothing = 0.0,
    weights = 1.0,
    loss_reduction = "SUM_BY_NONZERO_WEIGHTS",
    background_weight = 1.0,
    **kwargs):
  """Weighted softmax cross entropy loss. Borrowing VAL's implementation."""
  del kwargs
  # check_shape_equal(logits, labels)

  labels = labels.astype(logits.dtype)
  if label_smoothing > 0:
    num_classes = labels.shape[-1]
    smooth_weight = label_smoothing / num_classes
    smooth_weight = jnp.array(smooth_weight, dtype=logits.dtype)
    labels = (1. - label_smoothing) * labels + smooth_weight

  logits = jax.nn.log_softmax(logits)
  loss = -labels * logits

  # Apply background class weights
  class_weights = jnp.ones(loss.shape)
  class_weights[Ellipsis, :1] = background_weight  # Background is class 0.
  loss = loss * jnp.array(class_weights)

  loss = jnp.sum(loss, axis=-1)
  return compute_weighted_loss(loss, weights, logits.dtype, loss_reduction)


def huber_loss(predictions,
               labels,
               weights = 1.0,
               delta = 1.0,
               loss_reduction = "SUM_BY_NONZERO_WEIGHTS",
               **kwargs):
  """Returns the Huber loss. Borrowing VAL's implementation."""
  del kwargs
  # check_shape_equal(predictions, labels)
  labels = labels.astype(predictions.dtype)
  x = labels - predictions

  # Apply the formula above.
  loss = jnp.where(
      jnp.abs(x) <= delta, 0.5 * jax.lax.square(x),
      0.5 * delta * delta + delta * (jnp.abs(x) - delta))

  return compute_weighted_loss(loss, weights, predictions.dtype, loss_reduction)


class RpnScoreLoss(object):
  """Region Proposal Network score loss function.

  Attributes:
    rpn_batch_size_per_im: RPN batch size per image.
  """
  rpn_batch_size_per_im: int = 256

  def __call__(self, score_outputs, labels):
    """Computes and gathers RPN score losses from all levels.

    Args:
      score_outputs: A dictionary with keys representing levels and values
        representing scores in [batch_size, height, width, num_anchors].
      labels: A dictionary returned from dataloader with keys representing
        levels and values representing 0/1/-1 groundtruth targets in
        [batch_size, height, width, num_anchors].

    Returns:
      rpn_score_loss: A scalar representing total score loss.
    """
    levels = sorted(score_outputs.keys())

    score_losses = []
    for level in levels:
      score_losses.append(
          self._rpn_score_loss(  # pytype: disable=wrong-arg-types
              score_outputs[level],
              labels[level],
              normalizer=jnp.array(
                  score_outputs[level].shape[0] *
                  self.rpn_batch_size_per_im, dtype=jnp.float32)))

    # Sums per level losses to total loss.
    rpn_score_loss = jnp.sum(jnp.array(score_losses))

    return rpn_score_loss

  def _rpn_score_loss(self,
                      score_outputs,
                      score_targets,
                      normalizer = 1.0):
    """Computes RPN score loss for one level in the feature pyramid.

    Args:
      score_outputs: Scores with shape [batch_size, height, width, num_anchors].
      score_targets: Labels with shape [batch_size, height, width, num_anchors].
      normalizer: A scalar to normalize the loss by.

    Returns:
      score_loss: A scalar representing the RPN loss at this level.
    """
    # score_targets has three values:
    # (1) score_targets[i]=1, the anchor is a positive sample.
    # (2) score_targets[i]=0, negative.
    # (3) score_targets[i]=-1, the anchor is don"t care (ignore).
    mask = jnp.logical_or(score_targets == 1, score_targets == 0)
    score_targets = jnp.maximum(score_targets, jnp.zeros_like(score_targets))
    # RPN score loss is sum over all except ignored examples.
    score_loss = sigmoid_cross_entropy(
        logits=score_outputs, labels=score_targets, weights=mask,
        loss_reduction="SUM")

    return score_loss / normalizer


class RpnBoxLoss(object):
  """Region Proposal Network box regression loss function.

  Attributes:
    delta: The delta of box Huber loss.
  """
  delta: float = 0.111

  def __call__(self, box_outputs, labels):
    """Computes and gathers RPN box losses from all levels.

    Args:
      box_outputs: A dictionary with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4].
      labels: A dictionary returned from dataloader with keys representing
        levels and values representing groundtruth targets in
        [batch_size, height, width, num_anchors * 4].

    Returns:
      rpn_box_loss: a scalar representing total box regression loss.
    """
    levels = sorted(box_outputs.keys())

    box_losses = []
    for level in levels:
      box_losses.append(
          self._rpn_box_loss(
              box_outputs[level], labels[level], delta=self.delta))

    # Sums per level losses to total loss.
    rpn_box_loss = jnp.sum(jnp.array(box_losses))

    return rpn_box_loss

  def _rpn_box_loss(self,
                    box_outputs,
                    box_targets,
                    normalizer = 1.0,
                    delta = 0.111):
    """Computes RPN box loss for one level.

    Args:
      box_outputs: Box regression outputs with shape
        [batch_size, height, width, num_anchors * 4].
      box_targets: Box regression targets with shape
        [batch_size, height, width, num_anchors * 4].
      normalizer: A scalar to normalize the loss by.
      delta: A scalar to set the huber loss threshold. It"s a hyperparameter
        which perhaps needs to be tuned on different datasets for optimal
        performance.

    Returns:
      box_loss: A scalar representing the RPN loss at this level.
    """
    # The delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
    mask = box_targets != 0.0
    # The loss is normalized by the sum of non-zero weights before additional
    # normalizer provided by the function caller.
    box_loss = huber_loss(
        box_outputs,
        box_targets,
        weights=mask,
        delta=delta)

    return box_loss / normalizer


def fast_rcnn_class_loss(class_outputs,
                         class_targets,
                         normalizer = 1.0,
                         background_weight = 1.0):
  """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.

  This function implements the classification loss of the Fast-RCNN.

  The classification loss is softmax on all RoIs.
  Reference: https://github.com/facebookresearch/Detectron: fast_rcnn_heads.py

  Args:
    class_outputs: A float array representing the class prediction for each box
      with a shape of [batch_size, num_boxes, num_classes].
    class_targets: A float array representing the class label for each box
      with a shape of [batch_size, num_boxes].
    normalizer: A float of loss normalizer.
    background_weight: A float to adjust the weights of background. Default
      1.0 is a no-op.

  Returns:
    class_loss: A scalar representing total class loss.
  """
  _, _, num_classes = class_outputs.shape
  class_targets_one_hot = jax.nn.one_hot(class_targets.astype(jnp.int32),
                                         num_classes)
  class_loss = weighted_softmax_cross_entropy(
      class_outputs, class_targets_one_hot, background_weight=background_weight)

  return class_loss / normalizer


def fast_rcnn_box_loss(box_outputs,
                       class_targets,
                       box_targets,
                       class_box_regression = True,
                       normalizer = 1.0,
                       delta = 1.0,
                       one_hot_gather = True):
  """Computes the box loss (Fast-RCNN branch) of Mask-RCNN.

  This function implements the box regression loss of the Fast-RCNN. As the
  `box_outputs` produces `num_classes` boxes for each RoI, the reference model
  expands `box_targets` to match the shape of `box_outputs` and selects only
  the target that the RoI has a maximum overlap.
  (Reference: https://github.com/facebookresearch/Detectron: fast_rcnn.py)
  Instead, this function selects the `box_outputs` by the `class_targets` so
  that it doesn"t expand `box_targets`.

  The box loss is smooth L1-loss on only positive samples of RoIs.
  Reference: https://github.com/facebookresearch/Detectron: fast_rcnn_heads.py

  Args:
    box_outputs: A float array representing the box prediction for each box
      with a shape of [batch_size, num_boxes, num_classes * 4].
    class_targets: A float array representing the class label for each box
      with a shape of [batch_size, num_boxes].
    box_targets: A float array representing the box label for each box
      with a shape of [batch_size, num_boxes, 4].
    class_box_regression: Whether to compute loss from class-specific or
      class-agnostic outputs.
    normalizer: A scalar of loss normalizer.
    delta: A scalar to set the huber loss threshold.
    one_hot_gather: Whether or not to use a one-hot style gather of the
      boxes per class.

  Returns:
    box_loss: A scalar representing total box regression loss.
  """
  class_targets = class_targets.astype(jnp.int32)
  # Selects the box from `box_outputs` based on `class_targets`, with which
  # the box has the maximum overlap.
  if class_box_regression:
    batch_size, num_rois, num_class_specific_boxes = box_outputs.shape
    num_classes = num_class_specific_boxes // 4

    if one_hot_gather:
      box_indices = jnp.reshape(
          class_targets + jnp.tile(
              jnp.arange(batch_size)[:, None] * num_rois * num_classes,
              [1, num_rois]) + jnp.tile(
                  jnp.arange(num_rois)[None, :] * num_classes, [batch_size, 1]),
          [-1]).astype(box_outputs.dtype)

      box_outputs = jnp.matmul(
          jax.nn.one_hot(box_indices, batch_size * num_rois * num_classes),
          jnp.reshape(box_outputs, [-1, 4])
          )
      box_outputs = jnp.reshape(box_outputs, [batch_size, -1, 4])
    else:
      box_outputs = jnp.reshape(
          box_outputs, (batch_size, num_rois, num_classes, 4))

      def _gather_per_roi(per_roi_box, per_roi_class):
        return per_roi_box[per_roi_class, :]

      gather_boxes_fn = jax.vmap(
          jax.vmap(_gather_per_roi, in_axes=(0, 0), out_axes=0),
          in_axes=(0, 0), out_axes=0
      )

      box_outputs = gather_boxes_fn(box_outputs, class_targets)

  mask = jnp.tile(class_targets[Ellipsis, None] > 0, [1, 1, 4])
  # The loss is normalized by the sum of non-zero weights before applying an
  # additional scalar normalizer provided by the function caller.
  box_loss = huber_loss(
      box_outputs,
      box_targets,
      weights=mask,
      delta=delta)

  return box_loss / normalizer


def mask_rcnn_loss(mask_outputs,
                   mask_targets,
                   class_targets):
  """Computes the mask loss of Mask-RCNN.

  This implementation selects the `mask_outputs` by the `class_targets`
  so that the loss only occurs on foreground objects. The class selection logic
  happens in the heads.py of architecture.
  (Reference: https://github.com/facebookresearch/Detectron: mask_rcnn.py)

  Args:
    mask_outputs: A float tensor representing the prediction for each mask,
      with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    mask_targets: A float tensor representing the binary mask of ground truth
      labels for each mask with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    class_targets: A tensor with a shape of [batch_size, num_masks],
      representing the foreground mask targets.

  Returns:
    mask_loss: A float tensor representing total mask loss.
  """
  batch_size, num_masks, mask_height, mask_width = mask_outputs.shape
  weights = jnp.tile(
      (class_targets > 0).reshape([batch_size, num_masks, 1, 1]),
      [1, 1, mask_height, mask_width])

  mask_loss = sigmoid_cross_entropy(
      mask_outputs, mask_targets, weights=weights)

  return mask_loss


@gin.configurable
class ClassificationLoss(object):

  @gin_utils.allow_remapping
  def __call__(self, outputs, labels, **kwargs):
    class_outputs = outputs["logits"]
    class_loss = weighted_softmax_cross_entropy(class_outputs, labels)

    return {"model_loss": class_loss}


@gin.configurable
class MaskRCNNClassificationLoss(object):
  """A class to compute Mask R-CNN loss from model outputs and labels."""

  def __init__(
      self,
      include_masks = True,
      class_box_regression = True,
      include_frcnn = True):
    self._rpn_score_loss_fn = RpnScoreLoss()
    self._rpn_box_loss_fn = RpnBoxLoss()
    self._frcnn_class_loss_fn = fast_rcnn_class_loss
    self._frcnn_box_loss_fn = fast_rcnn_box_loss
    self._mask_loss_fn = mask_rcnn_loss
    self._include_masks = include_masks
    self._include_frcnn = include_frcnn
    self._class_box_regression = class_box_regression

  @gin_utils.allow_remapping
  def __call__(self, outputs, labels_det, labels_cls,
               **kwargs):
    """Compute Mask R-CNN loss from model outputs and groundtruth labels.

    Args:
      outputs: A dictionary with the following key-value pairs:
        "rpn_score_outputs": A dictionary with keys representing levels and
        values representing scores in [batch_size, height, width, num_anchors].
        "rpn_box_outputs": A dictionary with keys representing levels and values
        representing box regression targets in [batch_size, height, width,
        num_anchors * 4]. "class_outputs": An array representing the class
        prediction for each box with a shape of [batch_size, num_boxes,
        num_classes]. "box_outputs": An array representing the box prediction
        for each box with a shape of [batch_size, num_boxes, num_classes * 4].
        "class_targets": An array representing the class label for each box with
        a shape of [batch_size, num_boxes]. "box_targets": An array representing
        the box label for each box with a shape of [batch_size, num_boxes, 4].
        "mask_outputs": An array representing the prediction for each mask, with
        a shape of [batch_size, num_masks, mask_height, mask_width].
        "mask_targets": An array representing the binary mask of ground truth
        labels for each mask with a shape of [batch_size, num_masks,
        mask_height, mask_width]. "mask_class_targets": An array with a shape of
        [batch_size, num_masks], representing the classes of mask targets.
      labels_det: A dictionary with the following key-value pairs at least:
        "rpn_score_targets": A dictionary returned from dataloader with keys
        representing levels and values representing 0/1/-1 groundturth targets
        in [batch_size, height, width, num_anchors]. "rpn_box_targets": A
        dictionary returned from dataloader with keys representing levels and
        values representing groundturth targets in [batch_size, height, width,
        num_anchors * 4].
      labels_cls: The label for the classification task.
      **kwargs: Additional arguments.

    Returns:
      model_loss: A dictionary of losses for mask rcnn model.
    """
    del kwargs
    output_keys_for_loss = ["rpn_score_outputs", "rpn_box_outputs"]
    if self._include_frcnn:
      output_keys_for_loss.extend(
          ["class_outputs", "box_outputs", "box_targets"])

    rpn_score_loss = self._rpn_score_loss_fn(outputs["rpn_score_outputs"],
                                             labels_det["rpn_score_targets"])
    rpn_box_loss = self._rpn_box_loss_fn(outputs["rpn_box_outputs"],
                                         labels_det["rpn_box_targets"])
    if self._include_frcnn:
      frcnn_class_loss = self._frcnn_class_loss_fn(outputs["class_outputs"],
                                                   outputs["class_targets"])
      frcnn_box_loss = self._frcnn_box_loss_fn(outputs["box_outputs"],
                                               outputs["class_targets"],
                                               outputs["box_targets"],
                                               self._class_box_regression)
    else:
      frcnn_class_loss = frcnn_box_loss = 0.0

    if self._include_masks:
      mask_loss = self._mask_loss_fn(outputs["mask_outputs"],
                                     outputs["mask_targets"],
                                     outputs["mask_class_targets"])
    else:
      mask_loss = 0

    class_outputs = outputs["logits"]
    logging.info(class_outputs.shape)

    class_loss = weighted_softmax_cross_entropy(class_outputs, labels_cls)

    det_loss = (
        rpn_score_loss + rpn_box_loss + frcnn_class_loss + frcnn_box_loss +
        mask_loss)

    model_loss_dict = {
        "det_loss": det_loss,
        "cls_loss": class_loss,
        "rpn_score_loss": rpn_score_loss,
        "rpn_box_loss": rpn_box_loss,
        "fast_rcnn_class_loss": frcnn_class_loss,
        "fast_rcnn_box_loss": frcnn_box_loss,
        "mask_loss": mask_loss,
        "model_loss": det_loss,
    }
    return model_loss_dict
