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

"""Smoke test for augmentation pipeline."""

from absl.testing import absltest
import albumentations as A
from common import augmentations
import ml_collections
import numpy as np


def _make_test_cfg(enabled=True, crop=False):
  cfg = ml_collections.ConfigDict()
  cfg.augmentation = ml_collections.ConfigDict()
  cfg.augmentation.enabled = enabled
  cfg.augmentation.horizontal_flip_p = 0.5
  cfg.augmentation.vertical_flip_p = 0.5
  cfg.augmentation.rotate90_p = 0.5
  cfg.augmentation.color_jitter_p = 0.8
  cfg.augmentation.color_jitter_brightness = 0.2
  cfg.augmentation.color_jitter_contrast = 0.25
  cfg.augmentation.color_jitter_saturation = 0.25
  cfg.augmentation.color_jitter_hue = 0.05
  cfg.augmentation.gaussian_blur_p = 0.3
  cfg.augmentation.random_crop_enabled = crop
  cfg.augmentation.random_crop_p = 0.5
  cfg.augmentation.random_crop_scale = (0.5, 1.0)
  cfg.dataset = ml_collections.ConfigDict()
  cfg.dataset.image_size = 960
  return cfg


class AugmentationTest(absltest.TestCase):

  def test_disabled_returns_none(self):
    cfg = _make_test_cfg(enabled=False)
    self.assertIsNone(augmentations.get_train_augmentation(cfg))

  def test_flip_mirrors_boxes(self):
    cfg = _make_test_cfg()
    aug = augmentations.get_train_augmentation(cfg)
    self.assertIsNotNone(aug)
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    bboxes = [[10, 20, 100, 50]]
    cats = [0]
    aug_flip = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(
            format="coco",
            min_area=1.0,
            min_visibility=0.3,
            label_fields=["category_ids"],
        ),
    )
    result = aug_flip(image=image, bboxes=bboxes, category_ids=cats)
    self.assertEqual(result["image"].shape, (500, 800, 3))
    self.assertLen(result["bboxes"], 1)
    out_x, out_y, out_w, out_h = result["bboxes"][0]
    self.assertAlmostEqual(out_x, 800 - 10 - 100, places=0)
    self.assertAlmostEqual(out_y, 20, places=0)
    self.assertAlmostEqual(out_w, 100, places=0)
    self.assertAlmostEqual(out_h, 50, places=0)

  def test_color_jitter_preserves_boxes(self):
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    bboxes = [[10, 20, 100, 50], [200, 300, 80, 60]]
    cats = [0, 1]
    aug = A.Compose(
        [
            A.ColorJitter(
                brightness=0.2, contrast=0.25, saturation=0.25, hue=0.05, p=1.0
            )
        ],
        bbox_params=A.BboxParams(
            format="coco",
            min_area=1.0,
            min_visibility=0.3,
            label_fields=["category_ids"],
        ),
    )
    result = aug(image=image, bboxes=bboxes, category_ids=cats)
    self.assertEqual(result["image"].shape, (500, 800, 3))
    self.assertLen(result["bboxes"], 2)
    self.assertLen(result["category_ids"], 2)
    for orig, out in zip(bboxes, result["bboxes"]):
      for a, b in zip(orig, out):
        self.assertAlmostEqual(a, b, places=0)

  def test_full_pipeline_preserves_box_count(self):
    cfg = _make_test_cfg(crop=False)
    aug = augmentations.get_train_augmentation(cfg)
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    bboxes = [[10, 20, 100, 50], [200, 300, 80, 60], [400, 100, 120, 90]]
    cats = [0, 1, 0]
    kept = 0
    total = 0
    for _ in range(50):
      _, out_boxes, _ = augmentations.apply_augmentation(
          aug, image, bboxes, cats
      )
      kept += len(out_boxes)
      total += len(bboxes)
    retention = kept / total
    self.assertGreater(retention, 0.95)

  def test_apply_augmentation_return_types(self):
    cfg = _make_test_cfg(crop=False)
    aug = augmentations.get_train_augmentation(cfg)
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    bboxes = [[10, 20, 100, 50]]
    cats = [0]
    out_img, out_boxes, out_cats = augmentations.apply_augmentation(
        aug, image, bboxes, cats
    )
    self.assertIsInstance(out_img, np.ndarray)
    self.assertEqual(out_img.shape, (800, 800, 3))
    self.assertIsInstance(out_boxes, list)
    self.assertIsInstance(out_cats, list)


if __name__ == "__main__":
  absltest.main()
