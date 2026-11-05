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

"""Tests for main_lib training orchestration."""

import datetime
from unittest import mock

from absl.testing import absltest
import torch

from Uboreshaji_Modeli import main_lib
from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.engines import base as engines_base
from Uboreshaji_Modeli.trainers import base as trainers_base


class RunTrainingTest(absltest.TestCase):

  def test_run_training_dispatches_to_engine_and_trainer_on_cpu(self):
    cfg = config.get_base_config()
    cfg.model_flavor = config.ModelFlavor.GEMMA_3N_ASR
    cfg.task_modality = config.TaskModality.AUDIO
    cfg.task_type = config.TaskType.ASR
    cfg.model_id = "google/gemma-3n-E2B-it"
    cfg.training.device = "cpu"
    cfg.training.seed = 7

    fake_model = torch.nn.Linear(2, 2)
    fake_processor = object()
    fake_dataset = {"train": []}

    mock_engine = mock.create_autospec(engines_base.ModelEngine, instance=True)
    mock_engine.load_model_and_processor.return_value = (
        fake_model,
        fake_processor,
    )
    mock_engine_factory = mock.Mock(return_value=mock_engine)
    mock_dataset_getter = mock.Mock(return_value=fake_dataset)
    mock_strategy = mock.create_autospec(
        trainers_base.TrainerStrategy, instance=True
    )
    mock_trainer_factory = mock.Mock(return_value=mock_strategy)

    fixed_now = datetime.datetime(2026, 11, 2, 12, 30, 45)
    output_dir = self.create_tempdir().full_path

    main_lib.run_training(
        cfg,
        output_path=output_dir,
        engine_factory=mock_engine_factory,
        dataset_getter=mock_dataset_getter,
        trainer_factory=mock_trainer_factory,
        now_fn=lambda: fixed_now,
    )

    self.assertEqual(cfg.run_timestamp, "20261102_123045")
    mock_engine_factory.assert_called_once_with(config.ModelFlavor.GEMMA_3N_ASR)
    mock_engine.load_model_and_processor.assert_called_once_with(
        "google/gemma-3n-E2B-it",
        torch.device("cpu"),
        cfg=cfg,
        output_path=output_dir,
    )
    mock_dataset_getter.assert_called_once_with(cfg)
    mock_trainer_factory.assert_called_once_with(config.TaskType.ASR)
    mock_strategy.train.assert_called_once_with(
        mock_engine,
        fake_dataset,
        cfg,
        model=fake_model,
        processor=fake_processor,
        device=torch.device("cpu"),
        output_path=output_dir,
    )


if __name__ == "__main__":
  absltest.main()
