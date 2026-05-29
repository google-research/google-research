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

"""Tests for SAM 3 mask-to-bbox conversion utilities."""

from absl.testing import absltest
from models.sam3.scripts import masks_to_bboxes
import numpy as np


class MasksToBboxesTest(absltest.TestCase):

  def test_mask_to_bbox_empty(self):
    mask = np.zeros((10, 10), dtype=bool)
    bbox = masks_to_bboxes.mask_to_bbox(mask)
    self.assertIsNone(bbox)

  def test_mask_to_bbox_single_pixel(self):
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 5] = True
    bbox = masks_to_bboxes.mask_to_bbox(mask)
    self.assertEqual(bbox, [5, 4, 1, 1])

  def test_mask_to_bbox_rectangle(self):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    bbox = masks_to_bboxes.mask_to_bbox(mask)
    # Rows 2, 3, 4 (y_min=2, height=3)
    # Cols 3, 4, 5, 6 (x_min=3, width=4)
    self.assertEqual(bbox, [3, 2, 4, 3])

  def test_pcs_output_to_coco_predictions(self):
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[1:3, 2:4] = True
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[5:7, 6:8] = True

    masks = [mask1, mask2]
    scores = [0.95, 0.85]

    predictions = masks_to_bboxes.pcs_output_to_coco_predictions(
        masks, scores, image_id=42, category_id=3
    )

    self.assertLen(predictions, 2)
    self.assertEqual(
        predictions[0],
        {"image_id": 42, "category_id": 3, "bbox": [2, 1, 2, 2], "score": 0.95},
    )
    self.assertEqual(
        predictions[1],
        {"image_id": 42, "category_id": 3, "bbox": [6, 5, 2, 2], "score": 0.85},
    )


if __name__ == "__main__":
  absltest.main()
