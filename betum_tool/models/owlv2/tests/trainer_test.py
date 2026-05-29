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

"""Tests for Poly-Sense2 trainer logic."""

import unittest

from absl.testing import absltest
from models.owlv2 import box_utils
from models.owlv2 import losses
from models.owlv2 import trainer
import torch


class SimpleMockCriterion(losses.SetCriterion):

  def __init__(self):
    torch.nn.Module.__init__(self)

  def __call__(self, outputs, samples):
    del outputs, samples
    return {"loss_sigmoid_focal": torch.tensor(1.0)}


class TrainerTest(absltest.TestCase):

  def test_rescale_bboxes(self):
    bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    size = (1000.0, 800.0)
    rescaled = box_utils.rescale_bboxes(bboxes, size)
    # cxcywh [0.5, 0.5, 0.2, 0.2] -> xyxy [0.4, 0.4, 0.6, 0.6]
    # [0.4*800, 0.4*1000, 0.6*800, 0.6*1000] = [320, 400, 480, 600]
    expected = torch.tensor([[320, 400, 480, 600]], dtype=torch.float32)
    torch.testing.assert_close(rescaled, expected)

  def test_box_iou(self):
    boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    iou, _ = box_utils.box_iou(boxes1=boxes1, boxes2=boxes2)
    self.assertEqual(iou.shape, (2, 1))
    self.assertEqual(iou[0, 0], 1.0)
    self.assertAlmostEqual(iou[1, 0].item(), 1 / 7)

  def test_generalized_box_iou(self):
    boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    giou = box_utils.generalized_box_iou(boxes1=boxes1, boxes2=boxes2)
    self.assertAlmostEqual(giou[0, 0].item(), 1.0)

  def test_custom_trainer_compute_loss(self):
    model = unittest.mock.MagicMock()
    model.return_value = {"logits": torch.randn(1, 10, 2)}

    criterion = SimpleMockCriterion()
    args = unittest.mock.MagicMock()

    with unittest.mock.patch(
        "transformers.Trainer.__init__",
        return_value=None,
        autospec=True,
    ):
      custom_trainer = trainer.CustomTrainer(
          model=model,
          args=args,
          criterion=criterion,  # pytype: disable=wrong-arg-types
          weight_dict={"loss_sigmoid_focal": 1.0},
      )

    # Manually set attributes that __init__ would have set
    custom_trainer.model = model
    custom_trainer.args = args
    custom_trainer.criterion = criterion
    custom_trainer.weight_dict = {"loss_sigmoid_focal": 1.0}

    inputs = {
        "labels": [{
            "class_labels": torch.tensor([0]),
            "boxes": torch.tensor([[0, 0, 1, 1]]),
        }],
        "pixel_values": torch.randn(1, 3, 960, 960),
    }

    loss = custom_trainer.compute_loss(model, inputs)
    self.assertEqual(loss.item(), 1.0)
    self.assertIn("labels", inputs)
    self.assertLen(custom_trainer._loss_components_buffer, 1)

  def test_custom_trainer_logging(self):
    model = unittest.mock.MagicMock()
    criterion = SimpleMockCriterion()
    args = unittest.mock.MagicMock()

    with unittest.mock.patch(
        "transformers.Trainer.__init__",
        return_value=None,
        autospec=True,
    ):
      custom_trainer = trainer.CustomTrainer(
          model=model,
          args=args,
          criterion=criterion,
          weight_dict={"loss_sigmoid_focal": 1.0},
      )

    # Manually trigger compute_loss to fill buffer
    inputs = {
        "labels": [{
            "class_labels": torch.tensor([0]),
            "boxes": torch.tensor([[0, 0, 1, 1]]),
        }],
        "pixel_values": torch.randn(1, 3, 960, 960),
    }
    custom_trainer.compute_loss(model, inputs)
    custom_trainer.compute_loss(model, inputs)  # Add another to test averaging

    logs = {"loss": 1.0}
    start_time = 123.45
    with unittest.mock.patch("transformers.Trainer.log") as mock_log:
      custom_trainer.log(logs, start_time)
      mock_log.assert_called_once_with(logs, start_time)

    self.assertIn("train_loss_sigmoid_focal", logs)
    self.assertEqual(logs["train_loss_sigmoid_focal"], 1.0)
    self.assertEmpty(custom_trainer._loss_components_buffer)


if __name__ == "__main__":
  absltest.main()
