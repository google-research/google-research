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

"""A pipeline for segmenting objects using the SAM model."""

# Copyright 2024 The Google Research Authors.
# This file is based on the SAM (Segment Anything) and HQ-SAM.
#
# 		https://github.com/facebookresearch/segment-anything
# 		https://github.com/SysCV/sam-hq/tree/main
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


# pylint: disable=all
# pylint: disable=g-importing-member
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sam.utils import show_anns
from sam.utils import show_box
from sam.utils import show_mask
from sam.utils import show_points
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor


class SAMPipeline:

  def __init__(
      self,
      checkpoint,
      model_type,
      device="cuda:0",
      points_per_side=32,
      pred_iou_thresh=0.88,
      stability_score_thresh=0.95,
      box_nms_thresh=0.7,
  ):
    self.checkpoint = checkpoint
    self.model_type = model_type
    self.device = device
    self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
    self.sam.to(device=self.device)
    self.load_mask_generator(
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
    )

    # Default Prompt Args
    self.click_args = {"k": 5, "order": "max", "how_filter": "median"}
    self.box_args = None

  def load_sam(self):
    print("Loading SAM")
    sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
    sam.to(device=self.device)
    self.predictor = SamPredictor(sam)
    print("Loading Done")

  def load_mask_generator(
      self,
      points_per_side,
      pred_iou_thresh,
      stability_score_thresh,
      box_nms_thresh,
  ):
    print("Loading SAM")
    self.mask_generator = SamAutomaticMaskGenerator(
        model=self.sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
    )
    print("Loading Done")

  # segment single object
  def segment_image_single(
      self,
      image_path,
      input_point=None,
      input_label=None,
      input_box=None,
      input_mask=None,
      multimask_output=True,
      visualize=False,
      save_path=None,
      fname="",
      image=None,
  ):
    if image is None:
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    self.predictor.set_image(image)
    masks, scores, logits = self.predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        mask_input=None,
        multimask_output=multimask_output,
    )

    if visualize:
      self.visualize(
          image,
          masks,
          scores,
          save_path,
          input_point=input_point,
          input_label=input_label,
          input_box=input_box,
          input_mask=input_mask,
          fname=fname,
      )

    return masks, scores, logits

  def segment_automask(
      self,
      image_path,
      visualize=False,
      save_path=None,
      image=None,
      fname="automask.jpg",
  ):
    if image is None:
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_list, bbox_list = [], []
    masks = self.mask_generator.generate(image)
    mask_list.extend([mask["segmentation"] for mask in masks])
    bbox_list.extend([mask["bbox"] for mask in masks])

    if visualize:
      self.visualize_automask(image, masks, save_path, fname=fname)

    masks_arr, bbox_arr = np.array(mask_list), np.array(bbox_list)
    return masks_arr, bbox_arr, masks

  def visualize_automask(self, image, masks, save_path, fname="mask.jpg"):
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis("off")
    plt.savefig(os.path.join(save_path, fname))

  def visualize(
      self,
      image,
      masks,
      scores,
      save_path,
      input_point=None,
      input_label=None,
      input_box=None,
      input_mask=None,
      fname="",
  ):
    for i, (mask, score) in enumerate(zip(masks, scores)):
      plt.figure(figsize=(10, 10))
      plt.imshow(image)
      show_mask(mask, plt.gca())
      if input_point is not None:
        show_points(input_point, input_label, plt.gca())
      if input_box is not None:
        show_box(input_box, plt.gca())
      if input_mask is not None:
        show_mask(input_mask[0], plt.gca(), True)
      plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
      plt.axis("off")
      plt.savefig(os.path.join(save_path, f"{fname}{i}.jpg"))

    return input_point, input_label, input_box, input_mask
