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

"""Detection-aware data augmentation using albumentations."""

import albumentations as A
import ml_collections
import numpy as np


def get_train_augmentation(
    cfg,
):
  """Builds a detection-aware augmentation pipeline from config.

  Args:
    cfg: Experiment config with `augmentation` and `dataset` sections.

  Returns:
    An albumentations Compose pipeline with COCO bbox handling, or None if
    augmentation is disabled.
  """
  if not cfg.augmentation.enabled:
    return None

  transforms = []
  transforms.append(A.HorizontalFlip(p=cfg.augmentation.horizontal_flip_p))
  transforms.append(A.VerticalFlip(p=cfg.augmentation.vertical_flip_p))
  transforms.append(A.RandomRotate90(p=cfg.augmentation.rotate90_p))
  transforms.append(
      A.ColorJitter(
          brightness=cfg.augmentation.color_jitter_brightness,
          contrast=cfg.augmentation.color_jitter_contrast,
          saturation=cfg.augmentation.color_jitter_saturation,
          hue=cfg.augmentation.color_jitter_hue,
          p=cfg.augmentation.color_jitter_p,
      )
  )
  transforms.append(
      A.GaussianBlur(
          blur_limit=(3, 7),
          p=cfg.augmentation.gaussian_blur_p,
      )
  )
  if cfg.augmentation.random_crop_enabled:
    scale_min, scale_max = cfg.augmentation.random_crop_scale
    h, w = cfg.dataset.image_size, cfg.dataset.image_size
    transforms.append(
        A.RandomResizedCrop(
            height=h,
            width=w,
            scale=(scale_min, scale_max),
            ratio=(0.75, 1.333),
            p=cfg.augmentation.random_crop_p,
        )
    )
  return A.Compose(
      transforms,
      bbox_params=A.BboxParams(
          format="coco",
          min_area=1.0,
          min_visibility=0.3,
          label_fields=["category_ids"],
      ),
  )


_MAX_AUG_RETRIES = 3


def apply_augmentation(
    augmentation,
    image,
    bboxes,
    category_ids,
):
  """Applies augmentation, retrying if all bboxes are lost.

  Args:
    augmentation: An albumentations Compose pipeline.
    image: Input image as a numpy array.
    bboxes: COCO-format bounding boxes.
    category_ids: Class id for each bbox.

  Returns:
    Tuple of (augmented_image, augmented_bboxes, augmented_category_ids).
    Falls back to the original inputs after max retries.
  """
  for _ in range(_MAX_AUG_RETRIES):
    result = augmentation(image=image, bboxes=bboxes, category_ids=category_ids)
    if result["bboxes"]:
      return result["image"], result["bboxes"], result["category_ids"]
  return image, bboxes, category_ids
