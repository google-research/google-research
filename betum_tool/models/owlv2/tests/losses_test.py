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

"""Tests the loss functions.

This file includes tests for the `SetCriterion` class, which computes the total
loss for the model, as well as individual loss functions for class labels,
bounding boxes, and cardinality. The tests are written using `absl.testing`
library and `torch`.
"""

from absl.testing import absltest
from models.owlv2 import losses
from models.owlv2 import matcher
import torch


class LossesTest(absltest.TestCase):

  def test_set_criterion_init(self):
    matcher_instance = matcher.HungarianMatcher()
    weight_dict = {
        "loss_sigmoid_focal": 1.0,
        "loss_bbox": 1.0,
        "loss_giou": 1.0,
    }
    criterion = losses.SetCriterion(
        num_classes=1,
        matcher=matcher_instance,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    self.assertIsNotNone(criterion)

  def test_set_criterion_forward(self):
    matcher_instance = matcher.HungarianMatcher()
    weight_dict = {
        "loss_sigmoid_focal": 1.0,
        "loss_bbox": 1.0,
        "loss_giou": 1.0,
    }
    criterion = losses.SetCriterion(
        num_classes=1,
        matcher=matcher_instance,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # Logits shape (batch, queries, channels).
    # For num_classes=1, we need 2 channels (1 foreground, 1 background).
    outputs = {
        "logits": torch.randn(1, 10, 2),
        "pred_boxes": torch.rand(1, 10, 4),
    }
    targets = [{
        "class_labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
    }]
    loss_dict = criterion(outputs, targets)
    self.assertIn("loss_sigmoid_focal", loss_dict)
    self.assertIn("loss_bbox", loss_dict)
    self.assertIn("loss_giou", loss_dict)
    self.assertIn("cardinality_error", loss_dict)

  def test_loss_cardinality_with_background(self):
    matcher_instance = matcher.HungarianMatcher()
    criterion = losses.SetCriterion(
        num_classes=1,
        matcher=matcher_instance,
        weight_dict={"loss_sigmoid_focal": 1.0},
        eos_coef=0.1,
        losses=["cardinality"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # Logits: Query 0 has high foreground (prob ~1),
    #         Query 1 has high background (prob ~1)
    # Shape: (1, 2, 1) -> (batch, queries, classes)
    # 1 queries, 1 class (all foreground, no background sink)
    logits = torch.tensor([[[10.0], [-10.0]]])
    outputs = {"logits": logits}
    targets = [{"class_labels": torch.tensor([0])}]

    # card_pred should be 1
    # (only Query 0 has max foreground prob > 0.5)
    loss_dict = criterion.loss_cardinality(
        outputs, targets, [], torch.tensor([1.0])
    )
    # tgt_length is 1. card_pred should be 1. cardinality_error = |1 - 1| = 0.
    self.assertEqual(loss_dict["cardinality_error"].item(), 0.0)

  def test_loss_labels_with_background(self):
    matcher_instance = matcher.HungarianMatcher()
    criterion = losses.SetCriterion(
        num_classes=1,
        matcher=matcher_instance,
        weight_dict={"loss_sigmoid_focal": 1.0},
        eos_coef=0.1,
        losses=["labels"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # 2 queries, 2 classes (1 foreground, 1 background sink)
    logits = torch.randn(1, 2, 2)
    outputs = {"logits": logits}
    # Match Query 0 to Target 0 (which is class 0)
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [{"class_labels": torch.tensor([0])}]
    num_boxes = torch.tensor([1.0])

    # Check if it computes without crash.
    loss_dict = criterion.sigmoid_focal_loss_labels(
        outputs, targets, indices, num_boxes
    )
    self.assertIn("loss_sigmoid_focal", loss_dict)

  def test_loss_labels_with_eos_coef(self):
    matcher_instance = matcher.HungarianMatcher()
    # Baseline with eos_coef = 1.0 (no down-weighting)
    criterion_1 = losses.SetCriterion(
        num_classes=1,
        matcher=matcher_instance,
        weight_dict={"loss_sigmoid_focal": 1.0},
        eos_coef=1.0,
        losses=["labels"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # Down-weighted with eos_coef = 0.5
    criterion_05 = losses.SetCriterion(
        num_classes=1,
        matcher=matcher_instance,
        weight_dict={"loss_sigmoid_focal": 1.0},
        eos_coef=0.5,
        losses=["labels"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # 2 queries, 2 classes (1 foreground, 1 background sink)
    # Query 0 is foreground (prob ~0.12), Query 1 is background (prob ~0.12)
    logits = torch.ones(1, 2, 2) * -2.0
    outputs = {"logits": logits}
    # Match Query 0 to Target 0
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [{"class_labels": torch.tensor([0])}]
    num_boxes = torch.tensor([1.0])

    loss_1 = criterion_1.sigmoid_focal_loss_labels(
        outputs, targets, indices, num_boxes
    )["loss_sigmoid_focal"]
    loss_05 = criterion_05.sigmoid_focal_loss_labels(
        outputs, targets, indices, num_boxes
    )["loss_sigmoid_focal"]

    # loss_05 should be smaller than loss_1 because the background dimension
    # is down-weighted.
    self.assertLess(loss_05.item(), loss_1.item())

  def test_set_criterion_zero_boxes(self):
    matcher_instance = matcher.HungarianMatcher()
    criterion = losses.SetCriterion(
        num_classes=1,
        matcher=matcher_instance,
        weight_dict={"loss_sigmoid_focal": 1.0, "loss_bbox": 1.0},
        eos_coef=0.1,
        losses=["labels", "boxes"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    outputs = {
        "logits": torch.randn(1, 10, 2),
        "pred_boxes": torch.rand(1, 10, 4),
    }
    targets = [{
        "class_labels": torch.tensor([], dtype=torch.long),
        "boxes": torch.zeros(0, 4),
    }]

    # Should not crash with division by zero.
    loss_dict = criterion(outputs, targets)
    self.assertIn("loss_sigmoid_focal", loss_dict)
    self.assertFalse(torch.isnan(loss_dict["loss_sigmoid_focal"]))

  def test_loss_labels_multi_class(self):
    matcher_instance = matcher.HungarianMatcher()
    num_classes = 3
    criterion = losses.SetCriterion(
        num_classes=num_classes,
        matcher=matcher_instance,
        weight_dict={"loss_sigmoid_focal": 1.0},
        eos_coef=0.1,
        losses=["labels"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # 2 queries, 4 classes (3 foreground, 1 background sink)
    # logits shape: (batch, queries, channels)
    logits = torch.randn(1, 2, 4)
    outputs = {"logits": logits}
    # Match Query 0 to Target 0 (class 1), Query 1 to Target 1 (class 2)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    targets = [{"class_labels": torch.tensor([1, 2])}]
    num_boxes = torch.tensor([2.0])

    loss_dict = criterion.sigmoid_focal_loss_labels(
        outputs, targets, indices, num_boxes
    )
    self.assertIn("loss_sigmoid_focal", loss_dict)

  def test_loss_cardinality_multi_class(self):
    matcher_instance = matcher.HungarianMatcher()
    num_classes = 3
    criterion = losses.SetCriterion(
        num_classes=num_classes,
        matcher=matcher_instance,
        weight_dict={"loss_sigmoid_focal": 1.0},
        eos_coef=0.1,
        losses=["cardinality"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # 2 queries, 3 classes (all foreground, no background sink)
    # Query 0 has high foreground class 2 (prob ~1)
    # Query 1 has low probability everywhere (background)
    logits = torch.tensor([[
        [-10.0, -10.0, 10.0],
        [-10.0, -10.0, -10.0],
    ]])
    outputs = {"logits": logits}
    targets = [{"class_labels": torch.tensor([2])}]

    loss_dict = criterion.loss_cardinality(
        outputs, targets, [], torch.tensor([1.0])
    )
    # Query 0 is predicted as class 2. Query 1 is predicted as background.
    # card_pred should be 1. tgt_length is 1.
    self.assertEqual(loss_dict["cardinality_error"].item(), 0.0)

  def test_set_criterion_forward_multi_class(self):
    matcher_instance = matcher.HungarianMatcher()
    num_classes = 3
    weight_dict = {
        "loss_sigmoid_focal": 1.0,
        "loss_bbox": 1.0,
        "loss_giou": 1.0,
    }
    criterion = losses.SetCriterion(
        num_classes=num_classes,
        matcher=matcher_instance,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.3,
        focal_gamma=2.0,
    )
    # batch size 2, 10 queries, 4 channels
    outputs = {
        "logits": torch.randn(2, 10, 4),
        "pred_boxes": torch.rand(2, 10, 4),
    }
    targets = [
        {
            "class_labels": torch.tensor([1, 2]),
            "boxes": torch.rand(2, 4),
        },
        {
            "class_labels": torch.tensor([0]),
            "boxes": torch.rand(1, 4),
        },
    ]
    loss_dict = criterion(outputs, targets)
    self.assertIn("loss_sigmoid_focal", loss_dict)
    self.assertIn("loss_bbox", loss_dict)
    self.assertIn("loss_giou", loss_dict)
    self.assertIn("cardinality_error", loss_dict)


if __name__ == "__main__":
  absltest.main()
