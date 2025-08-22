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
third_party/cloud_tpu/models/detection/modeling/losses.py and
third_party/tensorflow_models/official/legacy/detection/modeling/losses.py

as well as the build_loss function in:
third_party/cloud_tpu/models/detection/modeling/maskrcnn_model.py
"""

from typing import Dict, Union

import gin
import jax
import jax.numpy as jnp
from losses import base_losses
from utils import box_utils
from utils import gin_utils

Array = jnp.ndarray
LevelArray = Dict[int, Array]
NamedLevelArray = Dict[str, Union[Array, LevelArray]]


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
          self._rpn_score_loss(
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
    # (3) score_targets[i]=-1, the anchor is don't care (ignore).
    mask = jnp.logical_or(score_targets == 1, score_targets == 0)
    score_targets = jnp.maximum(score_targets, jnp.zeros_like(score_targets))
    # RPN score loss is sum over all except ignored examples.
    score_loss = base_losses.sigmoid_cross_entropy(
        logits=score_outputs, labels=score_targets, weights=mask,
        loss_reduction=base_losses.LossReductionType.SUM)

    return score_loss / normalizer


class RpnCenterLoss(object):
  """Region Proposal Network center loss function.

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
          self._rpn_center_loss(
              score_outputs[level],
              labels[level],
              normalizer=jnp.array(
                  score_outputs[level].shape[0] *
                  self.rpn_batch_size_per_im, dtype=jnp.float32)))

    # Sums per level losses to total loss.
    rpn_score_loss = jnp.sum(jnp.array(score_losses))
    return rpn_score_loss

  def _rpn_center_loss(self,
                       score_outputs,
                       score_targets,
                       normalizer = 1.0):
    """Computes RPN centerness loss for one level in the feature pyramid.

    Args:
      score_outputs: Scores with shape [batch_size, height, width, num_anchors].
      score_targets: Labels with shape [batch_size, height, width, num_anchors].
      normalizer: A scalar to normalize the loss by.

    Returns:
      score_loss: A scalar representing the RPN loss at this level.
    """
    # score_targets has three values:
    # (1) score_targets[i] in [0, 1] if the anchor is a valid sample.
    # (2) score_targets[i]=-1, the anchor is don't care (ignore).
    score_outputs = jax.nn.sigmoid(score_outputs)
    mask = (score_targets > -0.5)
    score_targets = jnp.maximum(score_targets, jnp.zeros_like(score_targets))
    # RPN center loss is sum over all except ignored examples.
    score_loss = jnp.abs(score_outputs - score_targets)
    score_loss = base_losses.compute_weighted_loss(
        score_loss,
        weights=mask,
        loss_reduction=base_losses.LossReductionType.SUM,
        dtype=score_outputs.dtype
    )
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
      delta: A scalar to set the huber loss threshold. It's a hyperparameter
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
    box_loss = base_losses.huber_loss(
        box_outputs,
        box_targets,
        weights=mask,
        delta=delta)

    return box_loss / normalizer


class RpnBoxIoULoss(object):
  """Region Proposal Network box regression IoU loss function.

  Attributes:
    iou_loss_weight:
    rpn_batch_size_per_im:
  """
  iou_loss_weight: float = 8.0
  rpn_batch_size_per_im: int = 256

  def __call__(self, box_outputs, labels):
    """Computes and gathers RPN box losses from all levels.

    Args:
      box_outputs: A dictionary with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4]. The last channel is
        (left, right, top, bottom).
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
          self._rpn_iou_loss(
              box_outputs[level], labels[level],
              loss_weight=self.iou_loss_weight,
              normalizer=jnp.array(
                  box_outputs[level].shape[0] *
                  self.rpn_batch_size_per_im, dtype=jnp.float32)))
    # Sums per level losses to total loss.
    rpn_box_loss = jnp.sum(jnp.array(box_losses))

    return rpn_box_loss

  def _rpn_iou_loss(self,
                    box_outputs,
                    box_targets,
                    loss_weight = 8.0,
                    normalizer = 1.0):
    """Computes RPN box loss for one level.

    Args:
      box_outputs: Box regression outputs with shape
        [batch_size, height, width, num_anchors * 4].
      box_targets: Box regression targets with shape
        [batch_size, height, width, num_anchors * 4].
      loss_weight: A loss weight.
      normalizer: A scalar to normalize the loss by.

    Returns:
      box_loss: A scalar representing the RPN loss at this level.
    """
    mask_box = jnp.max(box_targets, -1) != 0.0
    batch, height, width = mask_box.shape
    if box_targets.shape != box_outputs.shape:
      raise ValueError(
          f'box_targets {box_targets.shape} != box_outputs {box_outputs.shape}')
    assert len(box_targets.shape) == 4
    box_outputs = jnp.reshape(box_outputs, [batch, height * width, 4])
    box_targets = jnp.reshape(box_targets, [batch, height * width, 4])
    mask_box = jnp.reshape(mask_box, [batch, height * width, 1])
    iou, mask_iou = box_utils.bbox_overlap(
        box_outputs, box_targets, is_aligned=True)
    if mask_box.shape != mask_iou.shape:
      raise ValueError(
          f'mask_box {mask_box.shape} != mask_iou {mask_iou.shape}')

    mask = jnp.logical_and(mask_iou, mask_box)
    if mask.shape != iou.shape:
      raise ValueError(
          f'mask {mask.shape} != iou {iou.shape}')

    iou_loss = 1. - iou
    iou_loss = base_losses.compute_weighted_loss(
        iou_loss,
        weights=mask,
        dtype=iou_loss.dtype,
        loss_reduction=base_losses.LossReductionType.SUM)
    return loss_weight * iou_loss / normalizer


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
  class_loss = base_losses.weighted_softmax_cross_entropy(
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
  that it doesn't expand `box_targets`.

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
  box_loss = base_losses.huber_loss(
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

  mask_loss = base_losses.sigmoid_cross_entropy(
      mask_outputs, mask_targets, weights=weights)

  return mask_loss


@gin.configurable
class MaskRCNNLoss(object):
  """A class to compute Mask R-CNN loss from model outputs and labels."""

  def __init__(self,
               include_masks = True,
               class_box_regression = True,
               include_frcnn = True,
               include_rpn = True,
               frcnn_background_weight = 1.0,
               frcnn_box_loss_one_hot_gather = True,
               use_centerness_rpn = False,
               use_box_iou_loss = False):
    """Initialize Mask R-CNN losses.

    Args:
      include_masks: Whether to add mask losses.
      class_box_regression: Whether to use class specific box losses.
      include_frcnn: Whether to include Faster R-CNN losses.
      include_rpn: Whether to include RPN losses.
      frcnn_background_weight: A float to adjust the weights of background.
        Default 1.0 is a no-op.
      frcnn_box_loss_one_hot_gather: Whether to use a one-hot gather of boxes
        per class in the frcnn_box_loss_fn.
      use_centerness_rpn: Whether to learn centerness prediction in RPN.
      use_box_iou_loss: Whether to use box IoU loss.
    """
    self._rpn_score_loss_fn = (
        RpnCenterLoss() if use_centerness_rpn else RpnScoreLoss())
    self._rpn_box_loss_fn = (
        RpnBoxIoULoss() if use_box_iou_loss else RpnBoxLoss())
    self._frcnn_class_loss_fn = fast_rcnn_class_loss
    self._frcnn_box_loss_fn = fast_rcnn_box_loss
    self._mask_loss_fn = mask_rcnn_loss
    self._include_masks = include_masks
    self._include_frcnn = include_frcnn
    self._include_rpn = include_rpn
    self._class_box_regression = class_box_regression
    self._frcnn_background_weight = frcnn_background_weight
    self._frcnn_box_loss_one_hot_gather = frcnn_box_loss_one_hot_gather

  @gin_utils.allow_remapping
  def __call__(self,
               outputs,
               labels,
               **kwargs):
    """Compute Mask R-CNN loss from model outputs and groundtruth labels.

    Args:
      outputs: A dictionary with the following key-value pairs:
        'rpn_score_outputs': A dictionary with keys representing levels and
          values representing scores in [batch_size, height, width,
          num_anchors].
        'rpn_box_outputs': A dictionary with keys representing levels and values
          representing box regression targets in
          [batch_size, height, width, num_anchors * 4].
        'class_outputs': An array representing the class prediction for each
          box with a shape of [batch_size, num_boxes, num_classes].
        'box_outputs': An array representing the box prediction for each box
          with a shape of [batch_size, num_boxes, num_classes * 4].
        'class_targets': An array representing the class label for each box
          with a shape of [batch_size, num_boxes].
        'box_targets': An array representing the box label for each box
          with a shape of [batch_size, num_boxes, 4].
        'mask_outputs': An array representing the prediction for each mask,
          with a shape of
          [batch_size, num_masks, mask_height, mask_width].
        'mask_targets': An array representing the binary mask of ground truth
          labels for each mask with a shape of
          [batch_size, num_masks, mask_height, mask_width].
        'mask_class_targets': An array with a shape of [batch_size, num_masks],
          representing the classes of mask targets.
      labels: A dictionary with the following key-value pairs at least:
        'rpn_score_targets': A dictionary returned from dataloader with keys
          representing levels and values representing 0/1/-1 groundtruth targets
          in [batch_size, height, width, num_anchors].
        'rpn_box_targets': A dictionary returned from dataloader with keys
          representing levels and values representing groundtruth targets in
          [batch_size, height, width, num_anchors * 4].
      **kwargs: Additional arguments.

    Returns:
      model_loss: A dictionary of losses for mask rcnn model.
    """
    del kwargs
    output_keys_for_loss = []
    if self._include_rpn:
      output_keys_for_loss.extend(
          ['rpn_score_outputs', 'rpn_box_outputs'])
    if self._include_frcnn:
      output_keys_for_loss.extend(
          ['class_outputs', 'box_outputs', 'box_targets'])
    ouputs_for_loss = {key: outputs[key] for key in output_keys_for_loss}

    base_losses.check_dtype_equal(
        ouputs_for_loss, jnp.float32, exclude_list=[])
    label_keys_for_loss = ['rpn_box_targets']
    labels_for_loss = {key: labels[key] for key in label_keys_for_loss}
    base_losses.check_dtype_equal(
        labels_for_loss, jnp.float32, exclude_list=[])

    if self._include_rpn:
      rpn_score_loss = self._rpn_score_loss_fn(outputs['rpn_score_outputs'],
                                               labels['rpn_score_targets'])
      rpn_box_loss = self._rpn_box_loss_fn(outputs['rpn_box_outputs'],
                                           labels['rpn_box_targets'])
    else:
      rpn_score_loss = rpn_box_loss = 0.0

    if self._include_frcnn:
      frcnn_class_loss = self._frcnn_class_loss_fn(
          outputs['class_outputs'],
          outputs['class_targets'],
          background_weight=self._frcnn_background_weight)
      frcnn_box_loss = self._frcnn_box_loss_fn(
          outputs['box_outputs'],
          outputs['class_targets'],
          outputs['box_targets'],
          self._class_box_regression,
          one_hot_gather=self._frcnn_box_loss_one_hot_gather)
    else:
      frcnn_class_loss = frcnn_box_loss = 0.0

    if self._include_masks:
      # Mask out the dummy mask losses when mask_loss_valid is provided.
      mask_loss = self._mask_loss_fn(
          outputs['mask_outputs'],
          outputs['mask_targets'],
          outputs['mask_class_targets'] * labels.get('mask_loss_valid', 1.0))
    else:
      mask_loss = 0

    model_loss = (rpn_score_loss + rpn_box_loss + frcnn_class_loss
                  + frcnn_box_loss + mask_loss)

    model_loss_dict = {
        'model_loss': model_loss,
        'rpn_score_loss': rpn_score_loss,
        'rpn_box_loss': rpn_box_loss,
        'fast_rcnn_class_loss': frcnn_class_loss,
        'fast_rcnn_box_loss': frcnn_box_loss,
        'mask_loss': mask_loss
    }
    return model_loss_dict


@gin.configurable
class BoxClassLoss:
  """A class to compute box classification loss from model outputs and labels."""

  def __init__(self):
    """Initialize losses."""
    self._frcnn_class_loss_fn = fast_rcnn_class_loss

  @gin_utils.allow_remapping
  def __call__(
      self, outputs, labels, **kwargs
  ):
    """Compute Mask R-CNN loss from model outputs and groundtruth labels.

    Args:
      outputs: A dictionary with the following key-value pairs: 'class_outputs':
        An array representing the class prediction for each box with a shape of
        [batch_size, num_boxes, num_classes]. 'class_targets': An array
        representing the class label for each box with a shape of [batch_size,
        num_boxes].
      labels: A dictionary of labels. This is unused here.
      **kwargs: Additional arguments.

    Returns:
      model_loss: A dictionary of losses for mask rcnn model.
    """
    del kwargs
    del labels
    frcnn_class_loss = self._frcnn_class_loss_fn(
        outputs['class_outputs'], outputs['class_targets']
    )

    model_loss = frcnn_class_loss

    model_loss_dict = {
        'model_loss': model_loss,
    }
    return model_loss_dict

