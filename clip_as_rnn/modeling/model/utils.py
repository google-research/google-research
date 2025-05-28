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

"""CAM utils."""

# pylint: disable=g-importing-member
import os

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
import torch
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

# pylint: disable=g-import-not-at-top
try:
  from torchvision.transforms import InterpolationMode

  BICUBIC = InterpolationMode.BICUBIC
except ImportError:
  BICUBIC = Image.BICUBIC

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def _convert_image_to_rgb(image):
  return image.convert('RGB')


def _transform_resize(h, w):
  return Compose([
      Resize((h, w), interpolation=BICUBIC),
      _convert_image_to_rgb,
      ToTensor(),
      Normalize(
          (0.48145466, 0.4578275, 0.40821073),
          (0.26862954, 0.26130258, 0.27577711),
      ),
  ])


def img_ms_and_flip(image, ori_height, ori_width, scales=1.0, patch_size=16):
  """Resizes and flips the image."""
  if isinstance(scales, float):
    scales = [scales]

  all_imgs = []
  for scale in scales:
    preprocess = _transform_resize(
        int(np.ceil(scale * int(ori_height) / patch_size) * patch_size),
        int(np.ceil(scale * int(ori_width) / patch_size) * patch_size),
    )
    image = preprocess(image)
    image_ori = image
    image_flip = torch.flip(image, [-1])
    all_imgs.append(image_ori)
    all_imgs.append(image_flip)
  return all_imgs


def reshape_transform(tensor, height=28, width=28):
  tensor = tensor.permute(1, 0, 2)
  result = tensor[:, 1:, :].reshape(
      tensor.size(0), height, width, tensor.size(2)
  )

  # Bring the channels to the first dimension, like in CNNs.
  result = result.transpose(2, 3).transpose(1, 2)
  return result


def vis_mask(image, mask, mask_color):
  # switch the height and width of image
  # image = image.transpose(1, 0, 2)
  if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
  fg = mask > 0.5
  rgb = np.copy(image)
  rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
  return Image.fromarray(rgb)


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
  """Get bounding boxes from scoremap."""
  height, width = scoremap.shape
  scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
  while True:
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY,
    )
    if thr_gray_heatmap.max() > 0 or threshold <= 0:
      break
    threshold -= 0.1
  contours = cv2.findContours(
      image=thr_gray_heatmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
  )[_CONTOUR_INDEX]

  # if len(contours) == 0:
  if not contours:
    return np.asarray([[0, 0, 0, 0]]), 1

  if not multi_contour_eval:
    contours = [max(contours, key=cv2.contourArea)]

  estimated_boxes = []
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0, x1, y1 = x, y, x + w, y + h
    x1 = min(x1, width - 1)
    y1 = min(y1, height - 1)
    estimated_boxes.append([x0, y0, x1, y1])

  return np.asarray(estimated_boxes), len(contours)


def mask2chw(arr):
  # Find the row and column indices where the array is 1
  rows, cols = np.where(arr == 1)
  # Calculate center of the mask
  center_y = int(np.mean(rows))
  center_x = int(np.mean(cols))
  # Calculate height and width of the mask
  height = rows.max() - rows.min() + 1
  width = cols.max() - cols.min() + 1
  return (center_y, center_x), height, width


def unpad(image_array, pad=None):
  if pad is not None:
    left, top, width, height = pad
    image_array = image_array[top : top + height, left : left + width, :]
  return image_array


def apply_visual_prompts(
    image_array,
    mask,
    visual_prompt_type=('circle',),
    visualize=False,
    color=(255, 0, 0),
    thickness=1,
    blur_strength=(15, 15),
):
  """Applies visual prompts to the image."""
  prompted_image = image_array.copy()
  if 'blur' in visual_prompt_type:
    # blur the part out side the mask
    # Blur the entire image
    blurred = cv2.GaussianBlur(prompted_image.copy(), blur_strength, 0)
    # Get the sharp region using the mask
    sharp_region = cv2.bitwise_and(
        prompted_image.copy(),
        prompted_image.copy(),
        mask=np.clip(mask, 0, 255).astype(np.uint8),
    )
    # Get the blurred region using the inverted mask
    inv_mask = 1 - mask
    blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
    # Combine the sharp and blurred regions
    prompted_image = cv2.add(sharp_region, blurred_region)
  if 'gray' in visual_prompt_type:
    gray = cv2.cvtColor(prompted_image.copy(), cv2.COLOR_BGR2GRAY)
    # make gray part 3 channel
    gray = np.stack([gray, gray, gray], axis=-1)
    # Get the sharp region using the mask
    color_region = cv2.bitwise_and(
        prompted_image.copy(),
        prompted_image.copy(),
        mask=np.clip(mask, 0, 255).astype(np.uint8),
    )
    # Get the blurred region using the inverted mask
    inv_mask = 1 - mask
    gray_region = (gray * inv_mask[:, :, None]).astype(np.uint8)
    # Combine the sharp and blurred regions
    prompted_image = cv2.add(color_region, gray_region)
  if 'black' in visual_prompt_type:
    prompted_image = cv2.bitwise_and(
        prompted_image.copy(),
        prompted_image.copy(),
        mask=np.clip(mask, 0, 255).astype(np.uint8),
    )
  if 'circle' in visual_prompt_type:
    mask_center, mask_height, mask_width = mask2chw(mask)
    center_coordinates = (mask_center[1], mask_center[0])
    axes_length = (mask_width // 2, mask_height // 2)
    prompted_image = cv2.ellipse(
        prompted_image,
        center_coordinates,
        axes_length,
        0,
        0,
        360,
        color,
        thickness,
    )
  if 'rectangle' in visual_prompt_type:
    mask_center, mask_height, mask_width = mask2chw(mask)
    # center_coordinates = (mask_center[1], mask_center[0])
    # axes_length = (mask_width // 2, mask_height // 2)
    start_point = (
        mask_center[1] - mask_width // 2,
        mask_center[0] - mask_height // 2,
    )
    end_point = (
        mask_center[1] + mask_width // 2,
        mask_center[0] + mask_height // 2,
    )
    prompted_image = cv2.rectangle(
        prompted_image, start_point, end_point, color, thickness
    )
  if 'contour' in visual_prompt_type:
    # Find the contours of the mask
    # fill holes for the mask
    mask = binary_fill_holes(mask)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # Draw the contours on the image
    prompted_image = cv2.drawContours(
        prompted_image.copy(), contours, -1, color, thickness
    )

  if visualize:
    cv2.imwrite(os.path.join('masked_img.png'), prompted_image)
  prompted_image = Image.fromarray(prompted_image.astype(np.uint8))
  return prompted_image
