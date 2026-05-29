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

import json
import os
import tempfile
from absl.testing import absltest
from common import evaluate


class EvaluateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()

    # Create mock COCO ground truth JSON
    self.gt_data = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "image2.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 0, "name": "unripe"},
            {"id": 1, "name": "ripe"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 30, 30],
                "area": 900,
                "iscrowd": 0,
            },
        ],
    }
    self.gt_path = os.path.join(self.test_dir.name, "gt.json")
    with open(self.gt_path, "w") as f:
      json.dump(self.gt_data, f)

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_perfect_match(self):
    # Create predictions that perfectly match the ground truth
    preds = [
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [10, 10, 20, 20],
            "score": 0.95,
        },
        {
            "image_id": 2,
            "category_id": 1,
            "bbox": [50, 50, 30, 30],
            "score": 0.88,
        },
    ]

    # Run evaluation
    results = evaluate.evaluate_coco(self.gt_path, preds, verbose=False)

    # Perfect match should give AP@50 = 1.0
    self.assertAlmostEqual(results["AP@50"], 1.0)
    self.assertAlmostEqual(results["per_class_ap50"]["unripe"], 1.0)
    self.assertAlmostEqual(results["per_class_ap50"]["ripe"], 1.0)

  def test_wrong_predictions(self):
    # Predictions that do not match well (e.g. wrong positions/categories)
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [80, 80, 10, 10],
            "score": 0.95,
        },
    ]

    results = evaluate.evaluate_coco(self.gt_path, preds, verbose=False)

    self.assertAlmostEqual(results["AP@50"], 0.0)
    # No prediction found above threshold / overlap for category 0, so AP is 0.0
    self.assertAlmostEqual(results["per_class_ap50"]["unripe"], 0.0)
    # Prediction for category 1 has 0 overlap, so AP is 0.0
    self.assertAlmostEqual(results["per_class_ap50"]["ripe"], 0.0)

  def test_score_threshold_filtering(self):
    # One high score (perfect match), one low score (filtered out)
    preds = [
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [10, 10, 20, 20],
            "score": 0.95,
        },
        {
            "image_id": 2,
            "category_id": 1,
            "bbox": [50, 50, 30, 30],
            "score": 0.20,
        },
    ]

    # Evaluate with score_threshold=0.5
    results = evaluate.evaluate_coco(
        self.gt_path, preds, score_threshold=0.5, verbose=False
    )

    # Category 'unripe' (id 0) was matched perfectly (AP=1.0)
    # Category 'ripe' (id 1) prediction was filtered out, so its AP is 0.0
    self.assertAlmostEqual(results["per_class_ap50"]["unripe"], 1.0)
    self.assertAlmostEqual(results["per_class_ap50"]["ripe"], 0.0)
    # mean AP@50 of unripe (1.0) and ripe (0.0) = 0.5
    self.assertAlmostEqual(results["AP@50"], 0.5)


if __name__ == "__main__":
  absltest.main()
