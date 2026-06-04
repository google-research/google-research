# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Loss logic adapted from DETR."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import ops

from Uboreshaji_Modeli.common import box_utils


def _get_src_permutation_idx(
    indices,
):
  """Computes permutation indices for the source predictions."""
  batch_idx = torch.cat(
      [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
  )
  return batch_idx, torch.cat([src for (src, _) in indices])


class SetCriterion(nn.Module):
  """A module that computes the bipartite matching loss for DETR.

  Computes a total loss that includes classification (Focal Loss b/479421226),
  bounding box (L1), and generalized IoU components.

  Attributes:
    num_classes: The number of object classes.
    matcher: An instance of `HungarianMatcher` used to find the best assignment
      between predictions and ground truth.
    weight_dict: A dictionary containing the weights for each loss type (e.g.,
      'loss_bbox', 'loss_giou, 'loss_sigmoid_focal').
    eos_coef: The coefficient for the "no object" class in the loss.
    losses: A list of strings specifying which losses to compute (e.g.,
      'labels', 'boxes', 'cardinality').
  """

  def __init__(
      self,
      num_classes,
      matcher,
      weight_dict,
      eos_coef,
      losses,
      focal_alpha,
      focal_gamma,
  ):
    """Initializes the instance."""
    super().__init__()
    self.num_classes = num_classes
    self.matcher = matcher
    self.weight_dict = weight_dict
    self.eos_coef = eos_coef
    self.losses = losses
    self.focal_alpha = focal_alpha
    self.focal_gamma = focal_gamma

  def sigmoid_focal_loss_labels(
      self,
      outputs,
      targets,
      indices,
      num_boxes,
  ):
    """Computes the sigmoid focal loss for class labels."""
    assert "logits" in outputs
    src_logits = outputs["logits"]
    idx = _get_src_permutation_idx(indices)

    target_classes_o = torch.cat(
        [t["class_labels"][j] for t, (_, j) in zip(targets, indices)]
    )

    target_classes = torch.full(
        src_logits.shape,
        0.0,
        dtype=src_logits.dtype,
        device=src_logits.device,
    )

    # Matched queries should predict their assigned class.
    target_classes[idx[0], idx[1], target_classes_o] = 1.0

    loss_sigmoid_focal = ops.sigmoid_focal_loss(
        src_logits,
        target_classes,
        alpha=self.focal_alpha,
        gamma=self.focal_gamma,
        reduction="none",
    )

    is_negative = target_classes == 0.0
    loss_sigmoid_focal = torch.where(
        is_negative,
        loss_sigmoid_focal * self.eos_coef,
        loss_sigmoid_focal,
    )

    return {"loss_sigmoid_focal": loss_sigmoid_focal.sum() / num_boxes}

  @torch.no_grad()
  def loss_cardinality(
      self,
      outputs,
      targets,
      indices,
      num_boxes,
  ):
    """Computes the cardinality loss for class labels.

    This loss is computed by taking the L1 distance between the number of
    predicted foreground classes and the number of target classes. It does not
    impact the bounding box predictions.

    Args:
        outputs: A dictionary of outputs from the model.
        targets: A list of dictionaries, each containing the ground truth labels
          for an image.
        indices: A list of tuples, where each tuple contains the indices of the
          predicted queries that are matched, and the second contains the
          indices of the target boxes that they are matched to.
        num_boxes: The number of boxes in the batch.
    Returns:
        A dictionary containing the cardinality loss.
    """
    del indices, num_boxes
    pred_logits = outputs["logits"]
    device = pred_logits.device
    tgt_lengths = torch.as_tensor(
        [len(v["class_labels"]) for v in targets], device=device
    )
    # For sigmoid models, we count queries with at least one
    # high-confidence foreground class.
    pred_probs = pred_logits.sigmoid()

    # This logic is made robust to a potential off-by-one error in
    # `self.num_classes`.
    card_pred = (pred_probs.max(-1).values > 0.5).sum(1)
    return {
        "cardinality_error": F.l1_loss(card_pred.float(), tgt_lengths.float())
    }

  def loss_boxes(
      self,
      outputs,
      targets,
      indices,
      num_boxes,
  ):
    """Computes the L1 and GIoU loss for bounding boxes."""
    assert "pred_boxes" in outputs
    idx = _get_src_permutation_idx(indices)
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = torch.cat(
        [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
    )

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

    losses = {}
    losses["loss_bbox"] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(
        box_utils.generalized_box_iou(
            boxes1=box_utils.box_cxcywh_to_xyxy(src_boxes),
            boxes2=box_utils.box_cxcywh_to_xyxy(target_boxes),
        )
    )
    losses["loss_giou"] = loss_giou.sum() / num_boxes
    return losses

  def get_loss(
      self,
      loss,
      outputs,
      targets,
      indices,
      num_boxes,
      **kwargs,
  ):
    """Computes the loss for a given loss type."""
    loss_map = {
        "labels": self.sigmoid_focal_loss_labels,
        "cardinality": self.loss_cardinality,
        "boxes": self.loss_boxes,
    }
    assert loss in loss_map, f"do you really want to compute {loss} loss?"
    return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

  def forward(
      self, outputs, targets
  ):
    """Computes the loss for the model outputs."""
    outputs_without_aux = {
        k: v for k, v in outputs.items() if k != "aux_outputs"
    }

    indices = self.matcher(outputs_without_aux, targets)

    num_boxes = sum(len(t["class_labels"]) for t in targets)
    num_boxes = torch.as_tensor(
        max(num_boxes, 1),
        dtype=torch.float,
        device=next(iter(outputs.values())).device,
    )
    if torch.distributed.is_initialized():
      torch.distributed.all_reduce(num_boxes)
      num_boxes = torch.clamp(
          num_boxes / torch.distributed.get_world_size(), min=1
      )

    losses = {}
    for loss in self.losses:
      losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
    return losses
