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

import types

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
import numpy as np
import torch

from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import losses
from Uboreshaji_Modeli.common import matcher
from Uboreshaji_Modeli.engines import owl


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
    def mock_processor(*args, **kwargs):
      del args
      if "text" in kwargs:
        # Text-only call made once at get_transform_fn time.
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
      # Image-only call made per example inside transform_fn.
      return {"pixel_values": torch.ones((1, 3, 960, 960))}

    text_inputs = ["leaf", "stem"]
    dataset_id2label = ["background", "leaf", "stem", "discard"]
    model_label2id = {"leaf": 0, "stem": 1}

    tf = self.engine.get_transform_fn(
        mock_processor, text_inputs, dataset_id2label, model_label2id
    )

    class MockImage:
      def convert(self, mode):
        del self, mode
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

  def test_normalize_annotation_square_image(self):
    # When image is square, max_side == h == w, so normalization is symmetric.
    boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
    original_size = (200, 200)
    norm_boxes = owl.normalize_annotation_for_owlv2(boxes, original_size)
    # cx = (0 + 50) / 200 = 0.25, cy = (0 + 50) / 200 = 0.25, w = h = 0.5
    expected = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)
    torch.testing.assert_close(norm_boxes, expected)

  def test_transform_fn_missing_kwargs_raises(self):
    # The ValueError is raised immediately at get_transform_fn call time
    # (via the base-class delegate) when required kwargs are None.
    def mock_processor(*args, **kwargs):
      del args, kwargs
      return {}

    with self.assertRaisesRegex(
        ValueError,
        "text_inputs, dataset_id2label, and model_label2id are required in"
        " kwargs for Owlv2Preprocessor.",
    ):
      self.engine.get_transform_fn(
          mock_processor,
          text_inputs=None,
          dataset_id2label=None,
          model_label2id=None,
      )

  def test_transform_fn_no_valid_objects(self):
    def mock_processor(*args, **kwargs):
      del args
      if "text" in kwargs:
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
      return {"pixel_values": torch.ones((1, 3, 960, 960))}

    # "unknown" label is not in model_label2id, so all objects are filtered out.
    tf = self.engine.get_transform_fn(
        mock_processor,
        text_inputs=["cat"],
        dataset_id2label=["unknown"],
        model_label2id={"cat": 0},
    )

    class MockImage:
      def convert(self, mode):
        del self, mode
        return type(
            "Image",
            (),
            {"__array__": lambda s, *a, **k: np.zeros((100, 100, 3))},
        )()

    examples = {
        "image_id": [1],
        "image": [MockImage()],
        "objects": [{"category": [0], "bbox": [[0, 0, 10, 10]]}],
    }
    result = tf(examples)
    self.assertIsNotNone(result)
    labels = result["labels"][0]
    self.assertEqual(labels["class_labels"].shape, (0,))
    self.assertEqual(labels["boxes"].shape, (0, 4))

  def test_post_process_score_threshold(self):
    # Three predictions per image: scores [0.9, 0.1, 0.8], threshold=0.5
    # → only indices 0 and 2 should survive.
    logits = torch.tensor([[[2.2, -1.0], [0.0, -2.0], [2.0, -1.0]]])
    pred_boxes_cxcywh = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.1, 0.1]]]
    )

    outputs = types.SimpleNamespace(logits=logits, pred_boxes=pred_boxes_cxcywh)

    target_sizes = torch.tensor([[100, 100]])
    results = self.engine.post_process(
        processor=None,
        outputs=outputs,
        target_sizes=target_sizes,
        score_threshold=0.5,
    )
    self.assertLen(results, 1)
    self.assertLen(results[0]["scores"], 2)

  def test_post_process_no_target_sizes(self):
    logits = torch.tensor([[[2.2, -1.0]]])
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])

    outputs = types.SimpleNamespace(logits=logits, pred_boxes=pred_boxes)

    results = self.engine.post_process(
        processor=None,
        outputs=outputs,
        target_sizes=None,
        score_threshold=0.0,
    )
    self.assertLen(results, 1)
    # Box should not be scaled when target_sizes is None.
    self.assertEqual(results[0]["boxes"].shape[-1], 4)

  def test_collate_fn_output_dtypes(self):
    collate_fn = self.engine.get_collate_fn(cfg=None)
    batch = [{
        "input_ids": torch.ones((2, 5), dtype=torch.int),
        "attention_mask": torch.ones((2, 5), dtype=torch.int),
        "pixel_values": torch.ones((3, 32, 32), dtype=torch.float),
        "labels": {
            "class_labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
        },
    }]
    collated = collate_fn(batch)
    self.assertEqual(collated["input_ids"].dtype, torch.int64)
    self.assertEqual(collated["attention_mask"].dtype, torch.int64)

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

  def test_get_criterion_unsupported_matcher_raises(self):
    cfg = ml_collections.ConfigDict()
    cfg.detection = ml_collections.ConfigDict()
    cfg.detection.weight_sigmoid_focal = 1.0
    cfg.detection.weight_bbox = 1.0
    cfg.detection.weight_giou = 1.0
    cfg.detection.eos_coef = 0.1
    cfg.detection.losses = ["labels"]
    cfg.detection.focal_loss_alpha = 0.25
    cfg.detection.focal_loss_gamma = 2.0
    cfg.matcher = ml_collections.ConfigDict()
    cfg.matcher.matcher_type = "unsupported_type"
    cfg.matcher.cost_class = 1.0
    cfg.matcher.cost_bbox = 1.0
    cfg.matcher.cost_giou = 1.0
    with self.assertRaisesRegex(
        ValueError, "Unsupported matcher type: unsupported_type"
    ):
      self.engine.get_criterion(5, cfg, torch.device("cpu"))


if __name__ == "__main__":
  absltest.main()
