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
import torch
from Uboreshaji_Modeli.common import box_utils
from Uboreshaji_Modeli.common import losses
from Uboreshaji_Modeli.common import trainer


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
    class DummyOutputs:
      def __init__(self):
        self.logits = torch.randn(1, 10, 2)
        self.pred_boxes = torch.randn(1, 10, 4)

    class DummyModel(torch.nn.Module):
      def forward(self, **kwargs):
        return DummyOutputs()

    model = DummyModel()

    criterion = SimpleMockCriterion()

    class DummyArgs:
      pass
    args = DummyArgs()

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
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.zeros(1, 10, dtype=torch.long),
    }

    loss = custom_trainer.compute_loss(model, inputs)
    self.assertEqual(loss.item(), 1.0)  # pyrefly: ignore[missing-attribute]
    self.assertIn("labels", inputs)
    self.assertLen(custom_trainer._loss_components_buffer, 1)

  def test_custom_trainer_run_forward(self):
    class DummyOutputs:
      def __init__(self):
        self.logits = torch.randn(1, 10, 2)
        self.pred_boxes = torch.randn(1, 10, 4)

    class DummyModel(torch.nn.Module):
      def forward(self, **kwargs):
        return DummyOutputs()

    model = DummyModel()

    class DummyArgs:
      pass
    args = DummyArgs()

    with unittest.mock.patch(
        "transformers.Trainer.__init__",
        return_value=None,
        autospec=True,
    ):
      custom_trainer = trainer.CustomTrainer(
          model=model,
          args=args,
      )

    inputs = {
        "pixel_values": torch.randn(1, 3, 960, 960),
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.zeros(1, 10, dtype=torch.long),
    }

    outputs = custom_trainer._run_forward(model, inputs)
    self.assertIn("logits", outputs)
    self.assertIn("pred_boxes", outputs)

  def test_custom_trainer_logging(self):
    model = torch.nn.Module()
    criterion = SimpleMockCriterion()

    class DummyArgs:
      pass
    args = DummyArgs()

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

    # Manually set attributes that __init__ would have set
    custom_trainer.model = model
    custom_trainer.args = args
    custom_trainer.criterion = criterion
    custom_trainer.weight_dict = {"loss_sigmoid_focal": 1.0}
    # Populate the buffer directly with known values to test averaging.
    custom_trainer._loss_components_buffer = [
        {"loss_sigmoid_focal": torch.tensor(2.0)},
        {"loss_sigmoid_focal": torch.tensor(4.0)},
    ]

    logs = {"loss": 1.0}
    start_time = 123.45
    with unittest.mock.patch("transformers.Trainer.log") as mock_log:
      custom_trainer.log(logs, start_time)
      mock_log.assert_called_once_with(logs, start_time)

    self.assertIn("train_loss_sigmoid_focal", logs)
    # Average of 2.0 and 4.0
    self.assertEqual(logs["train_loss_sigmoid_focal"], 3.0)
    self.assertEmpty(custom_trainer._loss_components_buffer)

  def test_custom_trainer_tpu_monkey_patch(self):
    from accelerate.utils import operations  # pylint: disable=g-import-not-at-top
    import accelerate.accelerator as acc_mod  # pylint: disable=g-import-not-at-top

    original_operations_gather = operations.gather
    original_acc_mod_gather = getattr(acc_mod, "gather", None)

    # Mock dist
    self.enter_context(
        unittest.mock.patch(
            "torch.distributed.is_initialized", return_value=True
        )
    )
    self.enter_context(
        unittest.mock.patch(
            "torch.distributed.get_backend", return_value="tpu_dist"
        )
    )
    self.enter_context(
        unittest.mock.patch("torch.distributed.get_world_size", return_value=2)
    )
    mock_all_gather = self.enter_context(
        unittest.mock.patch("torch.distributed.all_gather")
    )

    try:
      model = torch.nn.Module()

      class DummyArgs:
        pass
      args = DummyArgs()
      criterion = SimpleMockCriterion()

      with unittest.mock.patch(
          "transformers.Trainer.__init__", return_value=None
      ):
        _ = trainer.CustomTrainer(
            model=model,
            args=args,
            criterion=criterion,
            weight_dict={"loss_sigmoid_focal": 1.0},
        )

      # Verify patched
      self.assertNotEqual(operations.gather, original_operations_gather)

      # Test standard patched function
      tensor = torch.tensor([1, 2])
      def fake_all_gather(output_tensors, t):
        output_tensors[0] = t.clone()
        output_tensors[1] = t.clone() * 2

      mock_all_gather.side_effect = fake_all_gather

      result = operations.gather(tensor)

      # Verify result
      expected = torch.tensor([1, 2, 2, 4])
      torch.testing.assert_close(result, expected)

    finally:
      # Restore
      operations.gather = original_operations_gather
      if original_acc_mod_gather is not None:
        acc_mod.gather = original_acc_mod_gather


if __name__ == "__main__":
  absltest.main()
