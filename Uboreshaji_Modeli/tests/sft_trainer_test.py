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

from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import sft_trainer
from Uboreshaji_Modeli.engines import base as engines_base
from Uboreshaji_Modeli.trainers import factory as trainers_factory
from Uboreshaji_Modeli.trainers import sft


class FakeCriterion(torch.nn.Module):

  def forward(self, *args, **kwargs):
    return {}


class FakeModelEngine(engines_base.ModelEngine):

  def load_model_and_processor(self, model_id, device, **kwargs):
    return torch.nn.Module(), object()

  def get_preprocess_fn(self, *args, is_train=False, **kwargs):
    self.preprocess_is_train = getattr(self, "preprocess_is_train", [])
    self.preprocess_is_train.append(is_train)
    return lambda x: x

  def get_transform_fn(self, *args, **kwargs):
    self.transform_kwargs = kwargs
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
    mock_trainer_instance.train.assert_called_once_with(
        resume_from_checkpoint=False
    )
    self.assertTrue(mock_engine.transform_kwargs["is_train"])
    self.assertIs(
        mock_custom_sft_trainer_class.call_args.kwargs["train_dataset"],
        mock_dataset["train"],
    )

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

  def test_bagz_eval_uses_unwrapped_eval_stream(self):
    cfg = self._make_config()
    cfg.task_type = config.TaskType.ASR
    cfg.dataset.update(
        format="bagz", eval_split="validation", eval_num_records=8
    )
    cfg.training.eval_steps = 5
    cfg.eval = {"eval_batch_size": 2}
    self.enter_context(
        unittest.mock.patch.object(
            sft.bagz_data, "split_source", return_value=[b""] * 8
        )
    )
    mock_config = self.enter_context(
        unittest.mock.patch.object(trl, "SFTConfig", autospec=True)
    )
    mock_trainer = self.enter_context(
        unittest.mock.patch.object(
            sft_trainer, "CustomSFTTrainer", autospec=True
        )
    )

    engine = FakeModelEngine()
    sft.GemmaSFTTrainerStrategy().train(
        engine=engine,
        dataset=[b""] * 8,
        cfg=cfg,
        device=MockCpuDevice(),
        model=torch.nn.Module(),
        processor=object(),
        output_path="/tmp/dummy_output",
    )

    self.assertEqual(mock_config.call_args.kwargs["eval_strategy"], "steps")
    self.assertEqual(mock_config.call_args.kwargs["eval_steps"], 5)
    self.assertEqual(engine.preprocess_is_train, [True, False])
    eval_dataset = mock_trainer.call_args.kwargs["eval_dataset"]
    self.assertIsNotNone(eval_dataset)
    self.assertIs(
        mock_trainer.return_value.get_eval_dataloader(), eval_dataset
    )

  def test_fsdp_forwarded_only_when_multi_worker(self):
    cfg = self._make_config()
    cfg.training.fsdp = True
    cfg.training.fsdp_config = {"state_dict_type": "FULL_STATE_DICT"}
    ds = {"train": FakeDatasetSplit()}
    mock_config = self.enter_context(
        unittest.mock.patch.object(trl, "SFTConfig", autospec=True)
    )
    self.enter_context(
        unittest.mock.patch.object(
            sft_trainer, "CustomSFTTrainer", autospec=True
        )
    )

    sft.GemmaSFTTrainerStrategy().train(
        engine=FakeModelEngine(),
        dataset=ds,
        cfg=cfg,
        device=MockCpuDevice(),
        model=torch.nn.Module(),
        processor=object(),
        output_path=self.create_tempdir().full_path,
    )
    self.assertNotIn("fsdp", mock_config.call_args.kwargs)

    self.enter_context(
        unittest.mock.patch.object(
            sft.dist, "is_initialized", return_value=True
        )
    )
    self.enter_context(
        unittest.mock.patch.object(sft.dist, "get_rank", return_value=0)
    )
    self.enter_context(
        unittest.mock.patch.object(sft.dist, "get_world_size", return_value=8)
    )
    self.enter_context(
        unittest.mock.patch.object(sft.dist, "broadcast_object_list")
    )
    self.enter_context(unittest.mock.patch.object(sft.dist, "barrier"))
    mock_cleanup = self.enter_context(
        unittest.mock.patch.object(
            sft.distributed_training, "cleanup_distributed", autospec=True
        )
    )
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model._no_split_modules = {"Linear", "SiglipTextEmbeddings"}
    sft.GemmaSFTTrainerStrategy().train(
        engine=FakeModelEngine(),
        dataset=ds,
        cfg=cfg,
        device=MockCpuDevice(),
        model=model,
        processor=object(),
        output_path=self.create_tempdir().full_path,
    )
    mock_cleanup.assert_called_once()
    self.assertTrue(mock_config.call_args.kwargs["fsdp"])
    self.assertEqual(model._no_split_modules, ["Linear"])
    self.assertEqual(
        mock_config.call_args.kwargs["fsdp_config"],
        {"state_dict_type": "FULL_STATE_DICT"},
    )

  def test_skips_cleanup_distributed_on_exception(self):
    cfg = self._make_config()
    ds = {"train": FakeDatasetSplit()}
    mock_trainer_cls = self.enter_context(
        unittest.mock.patch.object(
            sft.sft_trainer, "CustomSFTTrainer", autospec=True
        )
    )
    mock_trainer_cls.return_value.train.side_effect = RuntimeError("OOM")
    self.enter_context(
        unittest.mock.patch.object(
            sft.trl, "SFTConfig", return_value=unittest.mock.MagicMock()
        )
    )
    mock_cleanup = self.enter_context(
        unittest.mock.patch.object(
            sft.distributed_training, "cleanup_distributed", autospec=True
        )
    )
    with self.assertRaises(RuntimeError):
      sft.GemmaSFTTrainerStrategy().train(
          engine=FakeModelEngine(),
          dataset=ds,
          cfg=cfg,
          device=MockCpuDevice(),
          model=torch.nn.Module(),
          processor=object(),
          output_path=self.create_tempdir().full_path,
      )
    mock_cleanup.assert_not_called()


class CustomSFTTrainerTest(absltest.TestCase):

  def test_patch_hf_quantizer_delegates_from_base_model(self):
    base_model = torch.nn.Module()
    sentinel_quantizer = object()
    base_model.hf_quantizer = sentinel_quantizer

    class FakePeftModel(torch.nn.Module):

      def get_base_model(self):
        return base_model

    peft_model = FakePeftModel()
    with unittest.mock.patch("trl.SFTTrainer.__init__", return_value=None):
      trainer = sft_trainer.CustomSFTTrainer(model=peft_model)

    self.assertIs(peft_model.hf_quantizer, sentinel_quantizer)
    self.assertIsNone(trainer.create_model_card())



class TrainerFactoryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("asr", config.TaskType.ASR),
      ("ctc_asr", config.TaskType.CTC_ASR),
  )
  def test_get_sft_trainer_strategy(self, task_type):
    strategy = trainers_factory.get_trainer(task_type)
    self.assertIsInstance(strategy, sft.GemmaSFTTrainerStrategy)


class OnRank0Test(absltest.TestCase):

  def test_other_ranks_receive_broadcast_result(self):
    dist = sft.dist
    self.enter_context(
        unittest.mock.patch.object(dist, "is_initialized", return_value=True)
    )
    self.enter_context(
        unittest.mock.patch.object(dist, "get_rank", return_value=1)
    )
    broadcast = self.enter_context(
        unittest.mock.patch.object(dist, "broadcast_object_list")
    )
    fn = unittest.mock.Mock()
    sft._on_rank0(fn)
    fn.assert_not_called()
    broadcast.assert_called_once_with([None], src=0)


if __name__ == "__main__":
  absltest.main()
