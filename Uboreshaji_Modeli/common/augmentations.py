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

"""Detection-aware data augmentation using torchvision v2."""

import ml_collections
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# TODO: b/529787903 - Make these configurable.
# Match albumentations: min_visibility=0.3, min_area=1.0.
_MAX_AUG_RETRIES = 3
_MIN_VISIBILITY = 0.3
_MIN_AREA = 1.0
_EPSILON = 1e-8


class RandomRotate90(v2.Transform):
  """Randomly rotates the input by 90, 180, or 270 degrees."""

  def __init__(self, p = 0.5):
    super().__init__()
    self.p = p

  def make_params(self, flat_inputs):
    apply = torch.rand(1).item() < self.p
    k = torch.randint(1, 4, (1,)).item() if apply else 0
    return dict(angle=k * 90)

  def transform(
      self, inpt, params
  ):
    angle = params["angle"]
    if angle == 0:
      return inpt
    return self._call_kernel(
        v2.functional.rotate, inpt, angle=angle, expand=True
    )


class RandomGaussianBlur(v2.Transform):
  """Randomly applies Gaussian Blur with varying kernel sizes."""

  def __init__(self, kernel_sizes, p = 0.5):
    super().__init__()
    self.kernel_sizes = kernel_sizes
    self.p = p

  def make_params(self, flat_inputs):
    apply = torch.rand(1).item() < self.p
    if not apply:
      return dict(kernel_size=0)
    idx = torch.randint(0, len(self.kernel_sizes), (1,)).item()
    return dict(kernel_size=self.kernel_sizes[idx])

  def transform(
      self, inpt, params
  ):
    kernel_size = params["kernel_size"]
    if kernel_size == 0:
      return inpt
    return self._call_kernel(
        v2.functional.gaussian_blur,
        inpt,
        kernel_size=[kernel_size, kernel_size],
    )


def get_train_augmentation(
    cfg,
):
  """Builds a detection-aware augmentation pipeline from config.

  Args:
    cfg: Experiment config with `augmentation` and `dataset` sections.

  Returns:
    A torchvision v2 Compose pipeline, or None if augmentation is disabled.
  """
  if not cfg.augmentation.enabled:
    return None

  transforms = []
  transforms.append(
      v2.RandomHorizontalFlip(p=cfg.augmentation.horizontal_flip_p)
  )
  transforms.append(v2.RandomVerticalFlip(p=cfg.augmentation.vertical_flip_p))
  transforms.append(RandomRotate90(p=cfg.augmentation.rotate90_p))

  color_jitter = v2.ColorJitter(
      brightness=cfg.augmentation.color_jitter_brightness,
      contrast=cfg.augmentation.color_jitter_contrast,
      saturation=cfg.augmentation.color_jitter_saturation,
      hue=cfg.augmentation.color_jitter_hue,
  )
  if cfg.augmentation.color_jitter_p < 1.0:
    transforms.append(
        v2.RandomApply([color_jitter], p=cfg.augmentation.color_jitter_p)
    )
  else:
    transforms.append(color_jitter)

  transforms.append(
      RandomGaussianBlur(
          kernel_sizes=[3, 5, 7],
          p=cfg.augmentation.gaussian_blur_p,
      )
  )

  if cfg.augmentation.random_crop_enabled:
    scale_min, scale_max = cfg.augmentation.random_crop_scale
    h, w = cfg.dataset.image_size, cfg.dataset.image_size
    crop = v2.RandomResizedCrop(
        size=(h, w),
        scale=(scale_min, scale_max),
        ratio=(0.75, 1.333),
    )
    if cfg.augmentation.random_crop_p < 1.0:
      transforms.append(
          v2.RandomApply([crop], p=cfg.augmentation.random_crop_p)
      )
    else:
      transforms.append(crop)

  return v2.Compose(transforms)


def _apply_augmentation_once(
    augmentation,
    image,
    bboxes,
    category_ids,
):
  """Applies torchvision v2 transforms, handling conversion and bbox filtering."""
  image_tensor = torch.from_numpy(image).permute(2, 0, 1)
  image_tv = tv_tensors.Image(image_tensor)

  if bboxes:
    bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
    h, w = image.shape[:2]
    boxes_tv = tv_tensors.BoundingBoxes(
        bbox_tensor,
        format=tv_tensors.BoundingBoxFormat.XYWH,
        canvas_size=(h, w),
    )
    orig_areas = boxes_tv[:, 2] * boxes_tv[:, 3]
  else:
    boxes_tv = tv_tensors.BoundingBoxes(
        torch.empty((0, 4), dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYWH,
        canvas_size=(image.shape[0], image.shape[1]),
    )
    orig_areas = torch.empty((0,), dtype=torch.float32)

  category_tensor = torch.tensor(category_ids, dtype=torch.int64)

  inputs = {
      "image": image_tv,
      "boxes": boxes_tv,
      "labels": category_tensor,
  }
  outputs = augmentation(inputs)

  aug_image = outputs["image"]
  aug_boxes = outputs["boxes"]
  aug_categories = outputs["labels"]

  if bboxes:
    new_areas = aug_boxes[:, 2] * aug_boxes[:, 3]
    visibility = new_areas / (orig_areas + _EPSILON)
    # Ensure width and height remain > 0 (non-degenerate).
    keep = (
        (visibility >= _MIN_VISIBILITY)
        & (new_areas >= _MIN_AREA)
        & (aug_boxes[:, 2] > 0)
        & (aug_boxes[:, 3] > 0)
    )
  else:
    keep = torch.empty((0,), dtype=torch.bool)

  final_boxes = aug_boxes[keep]
  final_categories = aug_categories[keep]

  out_image = aug_image.permute(1, 2, 0).numpy()
  out_boxes = final_boxes.tolist()
  out_categories = final_categories.tolist()

  return out_image, out_boxes, out_categories


def apply_augmentation(
    augmentation,
    image,
    bboxes,
    category_ids,
):
  """Applies augmentation, retrying if all bboxes are lost.

  Args:
    augmentation: A torchvision v2 Compose pipeline.
    image: Input image as a numpy array.
    bboxes: COCO-format bounding boxes.
    category_ids: Class id for each bbox.

  Returns:
    Tuple of (augmented_image, augmented_bboxes, augmented_category_ids).
    Falls back to the original inputs after max retries.
  """
  if not bboxes:
    return _apply_augmentation_once(augmentation, image, bboxes, category_ids)

  for _ in range(_MAX_AUG_RETRIES):
    out_img, out_boxes, out_cats = _apply_augmentation_once(
        augmentation, image, bboxes, category_ids
    )
    if out_boxes:
      return out_img, out_boxes, out_cats
  return image, bboxes, category_ids
