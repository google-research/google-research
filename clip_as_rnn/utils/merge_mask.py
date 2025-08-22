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

"""Mask merging functions for post-processing."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def merge_masks_simple(
    all_masks, target_h, target_w, threshold=0.5, scores=None
):
  """Merge masks."""
  merged_mask = None
  if scores is not None:
    merged_mask = torch.sum(all_masks * scores[:, None, None], dim=0)
    merged_mask /= torch.sum(scores)
  merged_mask = merged_mask.detach().cpu().numpy()
  # resize the mask to the target size
  merged_mask = cv2.resize(merged_mask, (target_w, target_h))
  merged_mask = np.where(merged_mask >= threshold, 1, 0).astype(np.uint8)
  if np.sum(merged_mask) <= 0.05 * (target_h * target_w):
    merged_mask = torch.any(all_masks > 0, dim=0)
    merged_mask = merged_mask.detach().cpu().numpy().astype(np.uint8)
    # resize the mask to the target size
    merged_mask = cv2.resize(merged_mask, (target_w, target_h))
    merged_mask = merged_mask > threshold
  merged_mask = torch.from_numpy(merged_mask).float()
  return merged_mask[None]


def merge_masks(all_masks, target_h, target_w, threshold=0.5):
  all_masks = torch.from_numpy(np.stack(all_masks)).float()
  mask_tensor = F.interpolate(
      all_masks[None], size=(target_h, target_w), mode='bilinear'
  ).squeeze(0)
  bg_mask = threshold * torch.ones((1, target_h, target_w))
  merged_mask = torch.cat([bg_mask, mask_tensor], dim=0)
  mask_idx = torch.argmax(merged_mask, dim=0)
  merged_mask = mask_idx > 0
  if merged_mask.sum() <= 0.05 * (target_h * target_w):
    merged_mask = torch.any(mask_tensor, dim=0)
  return merged_mask.float()[None]
