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

"""Run a Gradio demo of the CaR model on a single image."""
# pylint: disable=consider-using-from-import
# pylint: disable=g-bad-import-order
import argparse
import colorsys
import random
import gradio as gr
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

# Set random seed.
random.seed(15)
np.random.seed(0)
torch.manual_seed(0)


CFG_PATH = "configs/demo/pokemon.yaml"


def generate_distinct_colors(n):
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
    masks: list of np.arrays of shape (H, W) representing binary masks for each
        class

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
    image, masks, class_names, alpha=0.7, y_list=None, x_list=None
):
  """Visualize segmentation masks on an image.

  Args:
    image: np.array of shape (H, W, 3) representing the RGB image
    masks: list of np.arrays of shape (H, W) representing binary masks for each
        class
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
  binary_final_mask = np.zeros(
      (masks.shape[1], masks.shape[2]), dtype=np.float32
  )
  colors = generate_distinct_colors(len(class_names))
  idx = 0
  for mask, color, class_name in zip(masks, colors, class_names):
    # Overlay the mask
    final_mask += np.dstack([mask * c for c in color])
    binary_final_mask += mask
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
        fontsize=22,
        va="center",
        ha="center",
        bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
    )
    idx += 1

  image[binary_final_mask > 0] = image[binary_final_mask > 0] * (1 - alpha)
  final_image = image + final_mask * alpha
  final_image = final_image.astype(np.uint8)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  ax.imshow(final_image)
  # Remove axis ticks and labels
  ax.axis("off")
  return fig


def get_sam_masks(cfg, masks, image_path=None, img_sam=None, pipeline=None):
  """Get SAM masks."""

  print("generating sam masks online")
  if img_sam is None and image_path is not None:
    raise ValueError(
        "Please provide either the image path or the image numpy array."
    )

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
          iom_thres=cfg.car.iom_thres,
          min_pred_threshold=cfg.sam.min_pred_threshold,
      )
      for attn in attn_map
  ]
  for matched_mask, mask_id in zip(matched_masks, mask_ids):
    sam_masks = np.array([item["segmentation"] for item in matched_mask])
    sam_mask = np.any(sam_masks, axis=0)
    masks[mask_id] = torch.from_numpy(sam_mask).to(masks.device)
  return masks


def load_sam(cfg, device):
  sam_checkpoint, model_type = build_sam_config(cfg)
  pipeline = SAMPipeline(
      sam_checkpoint,
      model_type,
      device=device,
      points_per_side=cfg.sam.points_per_side,
      pred_iou_thresh=cfg.sam.pred_iou_thresh,
      stability_score_thresh=cfg.sam.stability_score_thresh,
      box_nms_thresh=cfg.sam.box_nms_thresh,
  )
  return pipeline


def generate(
    img,
    class_names,
    clip_thresh,
    mask_thresh,
    confidence_thresh,
    post_process,
    stability_score_thresh,
    box_nms_thresh,
    iom_thres,
    min_pred_threshold,
):
  """Generate segmentation masks given class names."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  cfg = Config(**load_yaml(CFG_PATH))
  cfg.car.clipes_threshold = clip_thresh
  cfg.car.mask_threshold = mask_thresh
  cfg.car.confidence_threshold = confidence_thresh
  cfg.sam.stability_score_thresh = stability_score_thresh
  cfg.sam.box_nms_thresh = box_nms_thresh
  cfg.car.iom_thres = iom_thres
  cfg.sam.min_pred_threshold = min_pred_threshold
  car_model = CaR(cfg, visualize=True, seg_mode="semantic", device=device)

  # resize image by dividing 2 if the size is larger than 1000
  if img.size[0] > 1000:
    img = img.resize((img.size[0] // 2, img.size[1] // 2))

  y_list, x_list = None, None
  class_names = class_names.split(",")
  sentences = class_names

  pseudo_masks, _ = car_model(img, sentences, 1)

  if post_process == "SAM":
    pipeline = load_sam(cfg, device)
    pseudo_masks = get_sam_masks(
        cfg,
        pseudo_masks,
        image_path=None,
        img_sam=np.array(img),
        pipeline=pipeline,
    )
    pseudo_masks = overlap_masks(pseudo_masks)

  # visualize segmentation masks
  demo_fig = visualize_segmentation(
      np.array(img),
      pseudo_masks.detach().cpu().numpy(),
      class_names,
      y_list=y_list,
      x_list=x_list,
  )

  # convert the demo figure to an pil image
  # pylint: disable=protected-access
  demo_fig.canvas.draw()
  demo_img = np.array(demo_fig.canvas.renderer._renderer)
  demo_img = Image.fromarray(demo_img)
  return demo_img


if __name__ == "__main__":
  parser = argparse.ArgumentParser("car")
  parser.add_argument(
      "--cfg-path",
      default="configs/local_car.yaml",
      help="path to configuration file.",
  )
  args = parser.parse_args()

  demo = gr.Interface(
      generate,
      inputs=[
          gr.Image(label="upload an image", type="pil"),
          "text",
          gr.Slider(
              label="clip thresh",
              minimum=0,
              maximum=1,
              value=0.4,
              step=0.1,
              info="the threshold for clip-es adversarial heatmap clipping",
          ),
          gr.Slider(
              label="mask thresh",
              minimum=0,
              maximum=1,
              value=0.6,
              step=0.1,
              info=(
                  "the binariation threshold for the mask to generate visual"
                  " prompt"
              ),
          ),
          gr.Slider(
              label="confidence thresh",
              minimum=0,
              maximum=1,
              value=0,
              step=0.1,
              info="the threshold for filtering the proposed classes",
          ),
          gr.Radio(
              ["CRF", "SAM"],
              label="post process",
              value="CRF",
              info="choose the post process method",
          ),
          gr.Slider(
              label=(
                  "stability score thresh for SAM mask proposal \n(only when"
                  " SAM is chosen for post process)"
              ),
              minimum=0,
              maximum=1,
              value=0.95,
              step=0.1,
          ),
          gr.Slider(
              label=(
                  "box nms thresh for SAM mask proposal \n(only when SAM is"
                  " chosen for post process)"
              ),
              minimum=0,
              maximum=1,
              value=0.7,
              step=0.1,
          ),
          gr.Slider(
              label=(
                  "intersection over mask threshold for SAM mask proposal"
                  " \n(only when SAM is chosen for post process)"
              ),
              minimum=0,
              maximum=1,
              value=0.5,
              step=0.1,
          ),
          gr.Slider(
              label=(
                  "minimum prediction threshold for SAM mask proposal \n(only"
                  " when SAM is chosen for post process)"
              ),
              minimum=0,
              maximum=1,
              value=0.03,
              step=0.01,
          ),
      ],
      outputs="image",
      title=(
          "CLIP as RNN: Segment Countless Visual Concepts without Training"
          " Endeavor"
      ),
      description=(
          "This is the official demo for CLIP as RNN. Please upload an image"
          " and type in the class names (connected by ',' e.g. cat,dog,human)"
          " you want to segment. The model will generate the segmentation masks"
          " for the input image. You can also adjust the clip thresh, mask"
          " thresh and confidence thresh to get better results."
      ),
      examples=[
          [
              "demo/pokemon.jpg",
              "Pikachu,Eevee",
              0.6,
              0.6,
              0,
              "CRF",
              0.95,
              0.7,
              0.6,
              0.01,
          ],
          [
              "demo/Eiffel_tower.jpg",
              "Eiffel Tower",
              0.6,
              0.6,
              0,
              "CRF",
              0.95,
              0.7,
              0.6,
              0.01,
          ],
          [
              "demo/superhero.jpeg",
              "Batman,Superman,Wonder Woman,Flash,Cyborg",
              0.6,
              0.6,
              0,
              "CRF",
              0.89,
              0.65,
              0.5,
              0.03,
          ],
      ],
  )
  demo.launch(share=True)
