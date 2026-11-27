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

"""Tests for DetectionTrainer strategy device propagation."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
import torch
import transformers

from Uboreshaji_Modeli.common import trainer
from Uboreshaji_Modeli.engines import base as engines_base
from Uboreshaji_Modeli.google.research import lumascope_visualizer
from Uboreshaji_Modeli.trainers import detection


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

  def get_criterion(self, num_classes, cfg, device, **kwargs):
    return FakeCriterion(), {}


class RecordingModelEngine(FakeModelEngine):
  """FakeModelEngine that records get_transform_fn keyword arguments."""

  def __init__(self):
    super().__init__()
    self.transform_kwargs = []

  def get_transform_fn(self, *args, **kwargs):
    self.transform_kwargs.append(kwargs)
    return super().get_transform_fn(*args, **kwargs)


class FakeCategoryFeature:

  def __init__(self, names):
    self.feature = unittest.mock.Mock()
    self.feature.names = names


class FakeDatasetSplit:

  def __init__(self, names):
    self.features = {"objects": {"category": FakeCategoryFeature(names)}}

  def with_transform(self, unused_transform_fn):
    return self

  def __len__(self):
    return 10

  def __getitem__(self, unused_idx):
    return {}


class MockTpuDevice:
  type = "tpu"


class MockCudaDevice:
  type = "cuda"


class SelectTargetCategoriesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("exact", ["background", "leaf"], ["background"], ["leaf"]),
      (
          "case_and_padding",
          ["Background", "plot row ", "leaf", "Stem"],
          ["background", " Plot Row", "discard"],
          ["leaf", "Stem"],
      ),
      ("nothing_excluded", ["leaf", "stem"], [], ["leaf", "stem"]),
  )
  def test_select_target_categories(self, names, exclude, expected):
    self.assertEqual(
        detection._select_target_categories(names, exclude), expected
    )


class DetectionTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Mock the visualization loop to prevent it from executing and
    # accessing stubs.
    self.mock_viz_loop = self.enter_context(
        unittest.mock.patch.object(
            lumascope_visualizer, "run_visualization_loop", autospec=True
        )
    )

    self.mock_dataset = {
        "train": FakeDatasetSplit(["class1", "class2"]),
        "validation": FakeDatasetSplit(["class1", "class2"]),
    }

    self.cfg = ml_collections.ConfigDict({
        "dataset": {
            "train_split": "train",
            "eval_split": "validation",
            "image_size": 960,
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
            "eval_steps": 10,
            "save_strategy": "no",
            "save_steps": 10,
            "save_total_limit": 1,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "max_steps": 10,
        },
        "eval": {
            "eval_batch_size": 2,
            "run_eval_only": True,
        },
        "visualization": {
            "score_threshold": 0.5,
            "num_samples": 5,
            "visualization_steps": 100,
        },
    })

  @parameterized.named_parameters(
      ("tpu", MockTpuDevice(), False),
      ("gpu", MockCudaDevice(), True),
  )
  def test_pin_memory_behavior(self, device, expected_pin_memory):
    trainer_strategy = detection.DetectionTrainer()
    mock_engine = FakeModelEngine()

    mock_model = torch.nn.Module()
    mock_processor = object()
    mock_output_path = "/tmp/dummy_output"

    # Capture Seq2SeqTrainingArguments initialization arguments.
    mock_args_class = self.enter_context(
        unittest.mock.patch.object(
            transformers, "Seq2SeqTrainingArguments", autospec=True
        )
    )

    # Stub out CustomTrainer to return a mock instance with state.
    mock_trainer_instance = unittest.mock.Mock()
    mock_trainer_instance.state = unittest.mock.Mock()
    mock_trainer_instance.state.log_history = []

    mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.3}

    mock_custom_trainer_class = self.enter_context(
        unittest.mock.patch.object(trainer, "CustomTrainer", autospec=True)
    )
    mock_custom_trainer_class.return_value = mock_trainer_instance

    trainer_strategy.train(
        engine=mock_engine,
        dataset=self.mock_dataset,
        cfg=self.cfg,
        device=device,  # pytype: disable=wrong-arg-types
        model=mock_model,
        processor=mock_processor,
        output_path=mock_output_path,
    )

    mock_args_class.assert_called_once()
    captured_kwargs = mock_args_class.call_args.kwargs
    self.assertEqual(
        captured_kwargs.get("dataloader_pin_memory"), expected_pin_memory
    )
    self.mock_viz_loop.assert_called_once()

  def test_excluded_classes_are_dropped_from_queries_and_targets(self):
    self.mock_dataset = {
        "train": FakeDatasetSplit(["Background", "leaf", "plot row ", "stem"]),
        "validation": FakeDatasetSplit(
            ["Background", "leaf", "plot row ", "stem"]
        ),
    }
    self.cfg.dataset.exclude_classes = ["background", "Plot Row"]
    engine = RecordingModelEngine()
    self.enter_context(
        unittest.mock.patch.object(
            transformers, "Seq2SeqTrainingArguments", autospec=True
        )
    )
    mock_trainer_instance = unittest.mock.create_autospec(
        trainer.CustomTrainer, instance=True
    )
    mock_trainer_instance.state = transformers.TrainerState()
    mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.3}
    self.enter_context(
        unittest.mock.patch.object(
            trainer,
            "CustomTrainer",
            autospec=True,
            return_value=mock_trainer_instance,
        )
    )

    detection.DetectionTrainer().train(
        engine=engine,
        dataset=self.mock_dataset,
        cfg=self.cfg,
        device=torch.device("cpu"),
        model=torch.nn.Module(),
        processor=object(),
        output_path=self.create_tempdir().full_path,
    )

    self.assertNotEmpty(engine.transform_kwargs)
    for kwargs in engine.transform_kwargs:
      self.assertEqual(kwargs["text_inputs"], ["leaf", "stem"])
      self.assertEqual(kwargs["model_label2id"], {"leaf": 0, "stem": 1})

if __name__ == "__main__":
  absltest.main()
