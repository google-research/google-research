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

"""Visualization functions."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
# pylint: disable=g-importing-member
from utils.utils import normalize

_VIS_HEIGHT = 512
_VIS_WIDTH = 512


def show_cam_on_image(img, mask):
  if img.shape[1] != mask.shape[1]:
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
  heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
  heatmap = np.float32(heatmap) / 255
  cam = heatmap + np.float32(img)
  cam = cam / np.max(cam)
  cam = np.uint8(255 * cam)
  return cam


def save_img(array, img_name):
  numpy_array = array.astype(np.uint8)
  image = Image.fromarray(numpy_array, mode="RGB")
  image.save(f"{img_name}.png")


def viz_attn(img, attn_map, prefix="vis_results/clipcam_img", img_name="cam"):
  """Visualize attention map."""
  num_masks = 1
  if len(attn_map.shape) == 3:
    num_masks = attn_map.shape[0]
  attn_map = attn_map.float().squeeze(1).detach().cpu().numpy()
  attn_map = normalize(attn_map)
  img = normalize(img)
  if num_masks == 1:
    vis = show_cam_on_image(img, attn_map)
    if not os.path.exists(prefix):
      os.makedirs(prefix)
    save_img(vis, os.path.join(prefix, f"{img_name}"))
    return vis
  for i in range(num_masks):
    vis = show_cam_on_image(img, attn_map[i])
    if not os.path.exists(prefix):
      os.makedirs(prefix)
    save_img(vis, os.path.join(prefix, f"{img_name}_{i}"))


def vis_mask(mask, gt_mask, img, output_dir, fname):
  """Visualize mask."""
  mask_img = torch.zeros((_VIS_WIDTH, _VIS_HEIGHT))
  mask_img[mask[0]] = 1

  # print(gt_mask.shape, img.size())
  # Assume img and gt_mask are also torch.Tensor with size (512, 512)
  img = img[0].permute(1, 2, 0).numpy()
  gt_mask_img = torch.zeros((_VIS_WIDTH, _VIS_HEIGHT))
  gt_mask_img[gt_mask[0]] = 1

  _, axs = plt.subplots(
      1, 3, figsize=(15, 5)
  )  # change the figsize if necessary

  axs[0].imshow(img)  # if image is grayscale, otherwise remove cmap argument
  axs[0].axis("off")
  axs[0].set_title("Original Image")

  axs[1].imshow(
      mask_img.numpy(), cmap="jet", alpha=0.5
  )  # using alpha for transparency
  axs[1].axis("off")
  axs[1].set_title("Mask")

  axs[2].imshow(
      gt_mask_img.numpy(), cmap="jet", alpha=0.5
  )  # using alpha for transparency
  axs[2].axis("off")
  axs[2].set_title("Ground Truth Mask")

  plt.savefig(
      os.path.join(output_dir, f"{fname}.jpg"),
      bbox_inches="tight",
      dpi=300,
      pad_inches=0.0,
  )
