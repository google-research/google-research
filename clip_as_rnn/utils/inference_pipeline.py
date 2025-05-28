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

"""The inference pipeline for the CaR model."""

import numpy as np
from PIL import Image
import torch

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
from modeling.post_process.post_process import generate_masks_from_sam
from modeling.post_process.post_process import match_masks
from utils.utils import process_sentence
from utils.metrics import IoU

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


def get_sam_masks(
    config, image_path, masks, matching_thresh=0.9, img_sam=None, pipeline=None
):
  """Generate SAM masks."""
  print("generating sam masks online")
  mask_tensor, mask_list = generate_masks_from_sam(
      image_path,
      save_path="./",
      pipeline=pipeline,
      img_sam=img_sam,
      visualize=False,
  )
  mask_tensor = mask_tensor.to(masks.device)
  # only conduct sam on masks that is not all zero
  attn_map, mask_ids = [], []
  for mask_id, mask in enumerate(masks):
    if torch.sum(mask) > 0:
      attn_map.append(mask.unsqueeze(0))
      mask_ids.append(mask_id)
  matched_masks = [
      match_masks(
          mask_tensor,
          attn,
          mask_list,
          iom_thres=config.car.iom_thres,
          min_pred_threshold=config.sam.min_pred_threshold,
      )
      for attn in attn_map
  ]
  for matched_mask, mask_id in zip(matched_masks, mask_ids):
    sam_masks = np.array([item["segmentation"] for item in matched_mask])
    sam_mask = np.any(sam_masks, axis=0)
    cur_mask = masks[mask_id]
    iou = IoU(torch.from_numpy(sam_mask).to(cur_mask.device), cur_mask)
    if iou > matching_thresh:
      masks[mask_id] = torch.from_numpy(sam_mask).to(masks.device)
  return masks


def inference_car(cfg, car_model, image_path, sentences, sam_pipeline=None):
  sentences = [process_sentence(sen, cfg.test.ds_name) for sen in sentences]
  img = Image.open(image_path).convert("RGB")
  if cfg.test.use_pseudo:
    masks, scores = car_model(img, sentences)
    return masks, scores

  masks, scores = car_model(img, sentences, cfg.car.num_iteration)
  sam_masks = get_sam_masks(
      cfg, image_path, masks, cfg.sam.matching_thresh, pipeline=sam_pipeline
  )
  return sam_masks, scores
