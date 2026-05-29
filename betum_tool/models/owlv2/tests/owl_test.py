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

"""Tests for OWL-v2 engine."""

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
from models.owlv2 import config
from models.owlv2 import losses
from models.owlv2 import matcher
from models.owlv2 import owl
import numpy as np
import torch


class Owlv2EngineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = owl.Owlv2Engine()

  def test_normalize_annotation_for_owlv2(self):
    boxes = torch.tensor([[10, 20, 100, 200]], dtype=torch.float32)
    original_size = (1000, 800)  # (h, w)
    norm_boxes = owl.normalize_annotation_for_owlv2(boxes, original_size)
    expected = torch.tensor([[0.06, 0.12, 0.1, 0.2]], dtype=torch.float32)
    torch.testing.assert_close(norm_boxes, expected)

  @parameterized.named_parameters(
      ("fp32", None, torch.float32),
      ("bf16", config.Precision.BF16, torch.bfloat16),
  )
  def test_collate_fn(self, precision, expected_dtype):
    cfg = None
    if precision:
      cfg = ml_collections.ConfigDict()
      cfg.training = ml_collections.ConfigDict()
      cfg.training.precision = precision
    collate_fn = self.engine.get_collate_fn(cfg=cfg)
    batch = [
        {
            "input_ids": torch.ones((2, 10), dtype=torch.int),
            "attention_mask": torch.ones((2, 10), dtype=torch.int),
            "pixel_values": torch.ones((3, 960, 960), dtype=torch.float),
            "labels": {
                "class_labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
            },
        },
        {
            "input_ids": torch.ones((2, 10), dtype=torch.int),
            "attention_mask": torch.ones((2, 10), dtype=torch.int),
            "pixel_values": torch.ones((3, 960, 960), dtype=torch.float),
            "labels": {
                "class_labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
            },
        },
    ]
    collated = collate_fn(batch)
    self.assertEqual(collated["input_ids"].shape, (4, 10))
    self.assertEqual(collated["attention_mask"].shape, (4, 10))
    self.assertEqual(collated["pixel_values"].shape, (2, 3, 960, 960))
    self.assertEqual(collated["pixel_values"].dtype, expected_dtype)
    self.assertLen(collated["labels"], 2)

  def test_transform_fn_mapping_and_filtering(self):
    def mock_processor(text, images, return_tensors, *args, **kwargs):
      del text, images, return_tensors, args, kwargs
      return {
          "input_ids": torch.tensor([[1, 2, 3]]),
          "attention_mask": torch.tensor([[1, 1, 1]]),
          "pixel_values": torch.ones((1, 3, 960, 960)),
      }

    text_inputs = ["leaf", "stem"]
    dataset_id2label = ["background", "leaf", "stem", "discard"]
    model_label2id = {"leaf": 0, "stem": 1}

    tf = self.engine.get_transform_fn(
        mock_processor, text_inputs, dataset_id2label, model_label2id
    )

    class MockImage:
      def convert(self, mode):
        del mode
        return type(
            "Image",
            (),
            {"__array__": lambda s, *a, **k: np.zeros((100, 100, 3))},
        )()

    examples = {
        "image_id": [1],
        "image": [MockImage()],
        "objects": [{
            "category": [1, 2, 3],
            "bbox": [[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]],
        }],
    }

    result = tf(examples)
    self.assertIsNotNone(result)
    labels = result["labels"][0]
    torch.testing.assert_close(labels["class_labels"], torch.tensor([0, 1]))
    self.assertLen(labels["boxes"], 2)

  def test_get_criterion(self):
    cfg = ml_collections.ConfigDict()
    cfg.detection = ml_collections.ConfigDict()
    cfg.detection.cost_class = 1.0
    cfg.detection.cost_bbox = 2.0
    cfg.detection.cost_giou = 2.0
    cfg.detection.weight_sigmoid_focal = 1.0
    cfg.detection.weight_bbox = 2.0
    cfg.detection.weight_giou = 2.0
    cfg.detection.eos_coef = 0.1
    cfg.detection.losses = ["labels", "boxes", "cardinality"]
    cfg.detection.focal_loss_alpha = 0.25
    cfg.detection.focal_loss_gamma = 2.0
    cfg.matcher = ml_collections.ConfigDict()
    cfg.matcher.matcher_type = config.MatcherType.HUNGARIAN
    cfg.matcher.cost_class = 1.0
    cfg.matcher.cost_bbox = 1.0
    cfg.matcher.cost_giou = 1.0

    num_classes = 5
    device = torch.device("cpu")
    criterion, weight_dict = self.engine.get_criterion(num_classes, cfg, device)

    self.assertIsInstance(criterion, losses.SetCriterion)
    self.assertEqual(criterion.num_classes, num_classes)
    self.assertEqual(weight_dict["loss_sigmoid_focal"], 1.0)
    self.assertEqual(weight_dict["loss_bbox"], 2.0)
    self.assertEqual(weight_dict["loss_giou"], 2.0)

    # Test Hungarian
    cfg.matcher.matcher_type = config.MatcherType.HUNGARIAN
    criterion, _ = self.engine.get_criterion(5, cfg, device)
    self.assertIsInstance(criterion.matcher, matcher.HungarianMatcher)

    # Test Greedy
    cfg.matcher.matcher_type = config.MatcherType.GREEDY
    criterion, _ = self.engine.get_criterion(5, cfg, device)
    self.assertIsInstance(criterion.matcher, matcher.GreedyMatcher)


if __name__ == "__main__":
  absltest.main()
