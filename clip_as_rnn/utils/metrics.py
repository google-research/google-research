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

"""Metrics for evaluating the performance of the model."""

import torch


def IoU(mask1, mask2, threshold=0.5):
  """Calculate Intersection over Union (IoU) between prediction and GT masks.

  Args:
      mask1: A torch.Tensor denoting the prediction, shape (N, H, W), where N is
        the number of masks.
      mask2: A torch.Tensor denoting the ground truth, shape (N, H, W), where N
        is the number of masks.
      threshold: The threshold to binarize masks.
  Returns:
      IoU of `mask1` and `mask2`.
  """
  if threshold > 0:
    mask1, mask2 = (mask1 > threshold).to(torch.bool), (mask2 > threshold).to(
        torch.bool
    )
  intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
  union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
  if union.sum() == 0:
    return 0
  return (intersection.to(torch.float) / union).mean().item()


def IoM(pred, target, min_pred_threshold=0.2):
  """Calculate Intersection over the area of gt Mask and pred Mask (IoM).

  between prediction and each ground truth masks.
  Precaution:
      this function works for prediction and target that are binary masks,
      where 1 represents the mask and 0 represents the background.
  Args:
      pred: A torch.Tensor denoting the prediction, shape (N, H, W), where N is
        the number of masks.
      target: A torch.Tensor denoting the ground truth, shape (N, H, W), where N
        is the number of masks.
      min_pred_threshold: prediction threshold.

  Returns:
      ious: A torch.Tensor denoting the IoU, shape (N,).
  """
  # calculate the intersection over all masks
  intersection = torch.einsum("mij,nij->mn", pred.to(target.device), target)
  area_pred = torch.einsum("mij->m", pred)
  area_target = torch.einsum("nij->n", target)
  # we calculate the IoM by dividing the intersection over the minimum area.
  iom_target = torch.einsum("mn,n->mn", intersection, 1 / area_target)
  iom_pred = torch.einsum("mn,m->mn", intersection, 1 / area_pred)
  # if the intersection is smaller than a certain percentage of the area of
  # the pred mask, we consider it as background.
  iom_target[iom_pred < min_pred_threshold] = 0
  # we consider the IoM as the maximum IoM between the pred mask and
  # the target mask.
  iom = torch.max(iom_target, iom_pred)
  iom = iom.max(dim=0)[0]
  return iom
