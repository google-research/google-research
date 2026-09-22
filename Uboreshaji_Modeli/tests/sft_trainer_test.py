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

"""Tests for GemmaSFTTrainerStrategy precision behavior."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
import torch
import trl

from Uboreshaji_Modeli.engines import base as engines_base
from Uboreshaji_Modeli.google.common import sft_trainer
from Uboreshaji_Modeli.google.trainers import sft


class FakeCriterion(torch.nn.Module):

  def forward(self, *args, **kwargs):
    return {}


class FakeModelEngine(engines_base.ModelEngine):

  def load_model_and_processor(self, model_id, device, **kwargs):
    return torch.nn.Module(), object()

  def get_transform_fn(self, *args, **kwargs):
    return lambda x: x

  def get_collate_fn(self, *args, **kwargs):
    return lambda x: x

  def get_sft_config_overrides(self, cfg):
    return {}


class FakeDatasetSplit:

  def __init__(self):
    self.features = {}

  def with_transform(self, unused_transform_fn):
    return self


class MockTpuDevice:
  type = "tpu"


class MockCpuDevice:
  type = "cpu"


class MockCudaDevice:
  type = "cuda"


class GemmaSFTTrainerStrategyTest(parameterized.TestCase):

  def _make_config(self, precision="bf16"):
    return ml_collections.ConfigDict({
        "dataset": {
            "train_split": "train",
        },
        "training": {
            "batch_size": 2,
            "num_train_epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "gradient_accumulation_steps": 1,
            "seed": 42,
            "data_seed": 42,
            "gradient_checkpointing": True,
            "precision": precision,
            "logging_steps": 10,
            "max_steps": 10,
        },
    })

  @parameterized.named_parameters(
      dict(
          testcase_name="tpu_preserves_bf16",
          requested_precision="bf16",
          device_factory=MockTpuDevice,
          expected_bf16=True,
          expected_fp16=False,
      ),
      dict(
          testcase_name="cpu_falls_back_to_fp32",
          requested_precision="bf16",
          device_factory=MockCpuDevice,
          expected_bf16=False,
          expected_fp16=False,
      ),
      dict(
          testcase_name="gpu_preserves_bf16",
          requested_precision="bf16",
          device_factory=MockCudaDevice,
          expected_bf16=True,
          expected_fp16=False,
      ),
  )
  def test_precision_device(
      self,
      requested_precision,
      device_factory,
      expected_bf16,
      expected_fp16,
  ):
    trainer_strategy = sft.GemmaSFTTrainerStrategy()
    mock_engine = FakeModelEngine()

    mock_dataset = {
        "train": FakeDatasetSplit(),
    }

    cfg = self._make_config(precision=requested_precision)

    mock_model = torch.nn.Module()
    mock_processor = object()
    mock_output_path = "/tmp/dummy_output"

    mock_sft_config_class = self.enter_context(
        unittest.mock.patch.object(trl, "SFTConfig", autospec=True)
    )

    mock_trainer_instance = unittest.mock.Mock()
    mock_custom_sft_trainer_class = self.enter_context(
        unittest.mock.patch.object(
            sft_trainer, "CustomSFTTrainer", autospec=True
        )
    )
    mock_custom_sft_trainer_class.return_value = mock_trainer_instance

    device = device_factory()

    is_cuda = device_factory == MockCudaDevice
    self.enter_context(
        unittest.mock.patch.object(
            torch.cuda, "is_available", return_value=is_cuda, autospec=True
        )
    )
    if is_cuda:
      self.enter_context(
          unittest.mock.patch.object(torch.cuda, "set_device", autospec=True)
      )

    trainer_strategy.train(
        engine=mock_engine,
        dataset=mock_dataset,
        cfg=cfg,
        device=device,
        model=mock_model,
        processor=mock_processor,
        output_path=mock_output_path,
    )

    mock_sft_config_class.assert_called_once()
    captured_kwargs = mock_sft_config_class.call_args.kwargs
    self.assertEqual(captured_kwargs.get("bf16"), expected_bf16)
    self.assertEqual(captured_kwargs.get("fp16"), expected_fp16)
    mock_trainer_instance.train.assert_called_once()

  def test_tpu_device_raises_error_on_fp16(self):
    trainer_strategy = sft.GemmaSFTTrainerStrategy()
    mock_engine = FakeModelEngine()

    mock_dataset = {
        "train": FakeDatasetSplit(),
    }

    cfg = self._make_config(precision="fp16")  # Request fp16

    mock_model = torch.nn.Module()
    mock_processor = object()
    mock_output_path = "/tmp/dummy_output"

    tpu_device = MockTpuDevice()

    with self.assertRaisesRegex(
        ValueError, "FP16 precision is not supported on TPU. Use BF16 or FP32."
    ):
      trainer_strategy.train(
          engine=mock_engine,
          dataset=mock_dataset,
          cfg=cfg,
          device=tpu_device,
          model=mock_model,
          processor=mock_processor,
          output_path=mock_output_path,
      )


class CustomSFTTrainerTest(absltest.TestCase):

  def test_tpu_monkey_patch(self):
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

      with unittest.mock.patch("trl.SFTTrainer.__init__", return_value=None):
        _ = sft_trainer.CustomSFTTrainer(
            model=model,
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
