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

"""Run a demo of the CaR model on a single image."""
# pylint: disable=consider-using-from-import
# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member
import argparse
import colorsys
from functools import reduce
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch

from modeling.model import CaR
from modeling.post_process.post_process import generate_masks_from_sam
from modeling.post_process.post_process import match_masks
from sam.sam import SAMPipeline
from sam.utils import build_sam_config
from utils.utils import Config
from utils.utils import load_yaml


def generate_distinct_colors(n):
  """Generate a color pallate."""
  colors = []
  # generate a random number from 0 to 1
  random_color_bias = random.random()

  for i in range(n):
    hue = float(i) / n
    hue += random_color_bias
    hue = hue % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    # Convert RGB values from [0, 1] range to [0, 255]
    colors.append(tuple(int(val * 255) for val in rgb))
  return colors


def overlap_masks(masks):
  """Overlap masks to generate a single mask for visualization.

  Args:
    masks: list of np.arrays of shape (H, W) representing binary masks
        for each class.

  Returns:
    overlap_mask: list of np.array of shape (H, W) that have no overlaps
  """
  overlap_mask = torch.zeros_like(masks[0])
  for mask_idx, mask in enumerate(masks):
    overlap_mask[mask > 0] = mask_idx + 1

  clean_masks = [overlap_mask == mask_idx + 1 for mask_idx in range(len(masks))]
  clean_masks = torch.stack(clean_masks, dim=0)

  return clean_masks


def visualize_segmentation(
    image, masks, class_names, alpha=0.45, y_list=None, x_list=None
):
  """Visualize segmentation masks on an image.

  Args:
    image: np.array of shape (H, W, 3) representing the RGB image
    masks: list of np.arrays of shape (H, W) representing binary masks
        for each class.
    class_names: list of strings representing names of each class
    alpha: float, transparency level of masks on the image
    y_list: list of y coordinates.
    x_list: list of x coordinates.
  Returns:
    visualization: plt.figure object
  """
  # Create a figure and axis
  fig, ax = plt.subplots(1, figsize=(12, 9))
  # Display the image
  # ax.imshow(image)
  # Generate distinct colors for each mask
  final_mask = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.float32)
  colors = generate_distinct_colors(len(class_names))
  idx = 0
  for mask, color, class_name in zip(masks, colors, class_names):
    # Overlay the mask
    final_mask += np.dstack([mask * c for c in color])
    # Find a representative point (e.g., centroid) for placing the label
    if y_list is None or x_list is None:
      y, x = np.argwhere(mask).mean(axis=0)
    else:
      y, x = y_list[idx], x_list[idx]
    ax.text(
        x,
        y,
        class_name,
        color="white",
        fontsize=36,
        va="center",
        ha="center",
        bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
    )

    idx += 1

  final_image = image * (1 - alpha) + final_mask * alpha
  final_image = final_image.astype(np.uint8)
  ax.imshow(final_image)
  # Remove axis ticks and labels
  ax.axis("off")
  return fig


def get_sam_masks(config, image_path, masks, img_sam=None, pipeline=None):
  """Get SAM masks online."""

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
    masks[mask_id] = torch.from_numpy(sam_mask).to(masks.device)
  return masks


def load_sam(config, sam_device):
  sam_checkpoint, model_type = build_sam_config(config)
  pipelines = SAMPipeline(
      sam_checkpoint,
      model_type,
      device=sam_device,
      points_per_side=config.sam.points_per_side,
      pred_iou_thresh=config.sam.pred_iou_thresh,
      stability_score_thresh=config.sam.stability_score_thresh,
      box_nms_thresh=config.sam.box_nms_thresh,
  )
  return pipelines


if __name__ == "__main__":
  parser = argparse.ArgumentParser("CaR")
  # default arguments

  # additional arguments
  parser.add_argument(
      "--output_path", type=str, default="", help="path to save outputs"
  )
  parser.add_argument(
      "--cfg-path",
      default="configs/voc_test.yaml",
      help="path to configuration file.",
  )
  args = parser.parse_args()

  cfg = Config(**load_yaml(args.cfg_path))
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # device = 'cpu'
  folder_name = reduce(
      lambda x, y: x.replace(" ", "_") + "_" + y, cfg.image_caption
  )
  if len(folder_name) > 20:
    folder_name = folder_name[:20]

  car_model = CaR(
      cfg, visualize=True, seg_mode=cfg.test.seg_mode, device=device
  )

  sam_pipeline = load_sam(cfg, device)

  img = Image.open(cfg.image_path).convert("RGB")

  # resize image by dividing 2 if the size is larger than 1000
  if img.size[0] > 1000:
    img = img.resize((img.size[0] // 3, img.size[1] // 3))

  label_space = cfg.image_caption
  pseudo_masks, scores, _ = car_model(img, label_space)

  if not cfg.test.use_pseudo:
    t1 = time.time()
    pseudo_masks = get_sam_masks(
        cfg,
        cfg.image_path,
        pseudo_masks,
        img_sam=np.array(img),
        pipeline=sam_pipeline,
    )
    pseudo_masks = overlap_masks(pseudo_masks)
    t2 = time.time()
    print(f"sam time: {t2 - t1}")

  # visualize segmentation masks
  demo_fig = visualize_segmentation(
      np.array(img),
      pseudo_masks.detach().cpu().numpy(),
      label_space,
  )
  save_path = f"vis_results/{folder_name}"
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  demo_fig.savefig(os.path.join(save_path, "demo.png"), bbox_inches="tight")

  print(f"results saved to {save_path}.")
