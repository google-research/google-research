# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Implementation of some utils for zeroshot segmenter."""

# pylint: disable=g-importing-member

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

BICUBIC = transforms.InterpolationMode.BICUBIC

RGB_MEAN = (0.48145466, 0.4578275, 0.40821073)
RGB_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_image_to_rgb(image):
  return image.convert('RGB')


def _preprocess(h, w):
  return transforms.Compose([
      transforms.Resize((h, w), interpolation=BICUBIC),
      _convert_image_to_rgb,
      transforms.ToTensor(),
      transforms.Normalize(RGB_MEAN, RGB_STD),
  ])


def reshape_transform(tensor, height=28, width=28):
  """Reshape transformation for output tensors of CAM.

  Args:
    tensor: the torch.Tensor to be reshaped.
    height: the target height.
    width: the target width.

  Returns:
    result: the reshaped tensor.
  """
  tensor = tensor.permute(1, 0, 2)
  result = tensor[:, 1:, :].reshape(
      tensor.size(0), height, width, tensor.size(2)
  )

  # Bring the channels to the first dimension like in CNNs.
  result = result.transpose(2, 3).transpose(1, 2)
  return result


def normalize_and_scale(imgs, target_size=None):
  """Normalize and scale image.

  Args:
    imgs: a list of np.ndarray representing images.
    target_size: A tuple of height and width for the target.

  Returns:
    result: A np.ndarray that has been normalized and resized to target size.
  """
  result = []
  for img in imgs:
    img = img - np.min(img)
    img = img / (1e-15 + np.max(img))
    if target_size is not None:
      img = cv2.resize(img, target_size)
    result.append(img)
  result = np.float32(result)

  return result


def image_multiscale_and_flip(
    image, original_height, original_width, scales=(1.0,), patch_size=16
):
  """Resize the image so h/w are multiples of patch_size."""

  all_imgs = []
  for scale in scales:
    preprocess = _preprocess(
        int(np.ceil(scale * int(original_height) / patch_size) * patch_size),
        int(np.ceil(scale * int(original_width) / patch_size) * patch_size),
    )
    image = preprocess(image)
    image_original = image
    image_flip = torch.flip(image, [-1])
    all_imgs.append(image_original)
    all_imgs.append(image_flip)
  return all_imgs


def score_map_to_bounding_box(score_map, threshold, multi_contour_eval=False):
  """Convert a binary score map to bounding boxes.

  Args:
    score_map: A np.ndarray representing the score map.
    threshold: The threshold to convert the mask to bounding box.
    multi_contour_eval: A boolean flag to determine whether to use multiple
      contour to get the bounding box.

  Returns:
    A tuple of (np.ndarray, int) where the first element represents the
    bounding boxes and the second represents the number of boxes.
  """
  height, width = score_map.shape
  score_map_image = np.expand_dims((score_map * 255).astype(np.uint8), 2)
  threshold_gray_heatmap = score_map_image
  while threshold > 0:
    _, threshold_gray_heatmap = cv2.threshold(
        src=score_map_image,
        thresh=int(threshold * np.max(score_map_image)),
        maxval=255,
        type=cv2.THRESH_BINARY,
    )
    if threshold_gray_heatmap.max() > 0:
      break
    threshold -= 0.1
  contours = cv2.findContours(
      image=threshold_gray_heatmap,
      mode=cv2.RETR_TREE,
      method=cv2.CHAIN_APPROX_SIMPLE,
  )[_CONTOUR_INDEX]

  if not contours:
    return np.zeros((1, 4)), 1

  if not multi_contour_eval:
    contours = [max(contours, key=cv2.contourArea)]

  estimated_boxes = []
  for contour in contours:
    xmin, ymin, w, h = cv2.boundingRect(contour)
    xmax, ymax = xmin + w, ymin + h
    xmax = min(xmax, width - 1)
    ymax = min(ymax, height - 1)
    estimated_boxes.append([xmin, ymin, xmax, ymax])

  return np.asarray(estimated_boxes), len(contours)


def mask_to_box(mask):
  """Convert a mask to a bounding box.

  Args:
    mask: A uint8 np.ndarray representing the mask.

  Returns:
    A tuple of (tuple(int, int), int, int) where representing
    (center, height, width).
  """
  # Find the row and column indices where the array >= 1.
  rows, cols = np.where(mask >= 1)
  # Calculate center of the mask.
  # We calculate the mass center to avoid the influence of the outliers.
  center_y = int(np.mean(rows))
  center_x = int(np.mean(cols))
  # Calculate height and width of the mask.
  height = rows.max() - rows.min() + 1
  width = cols.max() - cols.min() + 1
  return (center_y, center_x), height, width


def unpad(image_array, pad=None):
  """unpad an image.

  Args:
    image_array: The image to be unpadded.
    pad: A tuple representing the pad with shape (left, top, width, height).
      when pad is None, we simply return the original image.

  Returns:
    image_array: A new np.ndarray represeting the unpadded image.
  """
  if pad is not None:
    left, top, width, height = pad
    assert left >= 0 and top >= 0
    assert width > 0 and height > 0
    assert top + height < image_array.shape[0]
    assert left + width < image_array.shape[1]
    image_array = image_array[top : top + height, left : left + width, :]
  return image_array


def apply_visual_prompts(
    image_array,
    mask,
    visual_prompt_type=('circle',),
    color=(255, 0, 0),
    thickness=3,
):
  """Apply visual prompts for an image.

  Args:
    image_array: A np.ndarray representing the input image.
    mask: A np.ndarray representing the mask.
    visual_prompt_type: A tuple of visual prompts that may include 'circle',
      'blur', 'gray', 'rectangle' and 'contour'.
    color: The color for the boundary of visual prompts 'circle', 'rectangle'
      and 'contour'. Default value: (255, 0, 0) red.
    thickness: The thickness for the visual prompts including 'circle',
      'rectangle' and 'contour'. Default value: 3.

  Returns:
    prompted_image: A PIL.Image object represent the image applied with visual
      prompts
  """
  prompted_image = image_array.copy()
  if 'blur' in visual_prompt_type:
    # Blur the part out side the mask.
    # Blur the entire image.
    blur_strength = (15, 15)
    blurred = cv2.GaussianBlur(prompted_image.copy(), blur_strength, 0)
    # Get the sharp region using the mask.
    sharp_region = cv2.bitwise_and(
        prompted_image.copy(),
        prompted_image.copy(),
        mask=np.clip(mask, 0, 255).astype(np.uint8),
    )
    # Get the blurred region using the inverted mask.
    inv_mask = 1 - mask
    blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
    # Combine the sharp and blurred regions.
    prompted_image = cv2.add(sharp_region, blurred_region)
  if 'gray' in visual_prompt_type:
    gray = cv2.cvtColor(prompted_image.copy(), cv2.COLOR_BGR2GRAY)
    # Make gray part 3-channel.
    gray = np.stack([gray, gray, gray], axis=-1)
    # Get the sharp region using the mask.
    color_region = cv2.bitwise_and(
        prompted_image.copy(),
        prompted_image.copy(),
        mask=np.clip(mask, 0, 255).astype(np.uint8),
    )
    # Get the blurred region using the inverted mask.
    inv_mask = 1 - mask
    gray_region = (gray * inv_mask[:, :, None]).astype(np.uint8)
    # Combine the sharp and blurred regions.
    prompted_image = cv2.add(color_region, gray_region)
  if 'circle' in visual_prompt_type:
    mask_center, mask_height, mask_width = mask_to_box(mask)
    axes_length = (mask_width // 2, mask_height // 2)

    prompted_image = cv2.ellipse(
        prompted_image,
        (mask_center[1], mask_center[0]),
        axes_length,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=color,
        thickness=thickness,
    )
  if 'rectangle' in visual_prompt_type:
    mask_center, mask_height, mask_width = mask_to_box(mask)
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
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # Draw the contours on the image.
    prompted_image = cv2.drawContours(
        prompted_image.copy(), contours, -1, color, thickness
    )
  prompted_image = Image.fromarray(prompted_image.astype(np.uint8))
  return prompted_image
