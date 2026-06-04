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

"""Tests for experiment configuration."""

from absl.testing import absltest
from absl.testing import parameterized
from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import config_utils


class GetBaseConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = config.get_base_config()

  def test_returns_config_dict(self):
    self.assertIsNotNone(self.cfg)

  @parameterized.named_parameters(
      dict(
          testcase_name="dataset_uri",
          field_path="dataset.dataset_uri",
          expected="/path/to/huggingface_dataset",
      ),
      dict(
          testcase_name="dataset_version",
          field_path="dataset.dataset_version",
          expected="dataset_version",
      ),
      dict(
          testcase_name="matcher_type",
          field_path="matcher.matcher_type",
          expected=config.MatcherType.HUNGARIAN,
      ),
      dict(
          testcase_name="training_seed", field_path="training.seed", expected=42
      ),
      dict(
          testcase_name="training_precision",
          field_path="training.precision",
          expected=config.Precision.BF16,
      ),
      dict(
          testcase_name="training_save_strategy",
          field_path="training.save_strategy",
          expected="steps",
      ),
  )
  def test_default_value(self, field_path, expected):
    value = self.cfg
    for part in field_path.split("."):
      value = getattr(value, part)
    self.assertEqual(value, expected)


class DerivePathsTest(absltest.TestCase):

  def test_vision_modality_derives_paths(self):
    cfg = config.get_base_config()
    cfg.task_modality = config.TaskModality.VISION
    cfg.dataset.dataset_base = "/base"
    cfg.dataset.dataset_uri = "ds"
    cfg.dataset.dataset_version = "v1"
    cfg.model_base = "/mbase"
    cfg.model_name = "mname"
    config_utils.derive_paths(cfg)
    self.assertEqual(
        cfg.dataset.dataset_path, "/base/ds/huggingface_dataset/v1/"
    )
    self.assertEqual(cfg.model_id, "/mbase/mname")

  def test_audio_modality_preserves_explicit_model_id(self):
    cfg = config.get_base_config()
    cfg.task_modality = config.TaskModality.AUDIO
    cfg.model_id = "openai/whisper-large-v3"
    cfg.model_base = "/mbase"
    cfg.model_name = "mname"
    cfg.dataset.dataset_base = "/base"
    cfg.dataset.dataset_uri = "ds"
    config_utils.derive_paths(cfg)
    self.assertEqual(cfg.model_id, "openai/whisper-large-v3")
    self.assertEqual(cfg.dataset.dataset_path, "/base/ds")

  def test_audio_modality_derives_path_if_model_id_is_legacy_default(self):
    cfg = config.get_base_config()
    # cfg.model_id is pre-populated by get_base_config() with the vision default
    cfg.task_modality = config.TaskModality.AUDIO
    cfg.model_base = "/audio/base"
    cfg.model_name = "model"
    config_utils.derive_paths(cfg)
    self.assertEqual(cfg.model_id, "/audio/base/model")

  def test_vision_modality_preserves_explicit_dataset_path(self):
    cfg = config.get_base_config()
    cfg.task_modality = config.TaskModality.VISION
    cfg.dataset.dataset_path = "/explicit/path"
    cfg.dataset.dataset_base = "/ignored/base"
    cfg.dataset.dataset_uri = "ds"
    cfg.dataset.dataset_version = "v1"
    config_utils.derive_paths(cfg)
    self.assertEqual(cfg.dataset.dataset_path, "/explicit/path")

  def test_hasattr_value_unwraps_custom_enum_modality(self):
    import enum  # pylint: disable=g-import-not-at-top

    class CustomModality(enum.Enum):
      VISION = "VISION"

    cfg = config.get_base_config()
    del cfg.task_modality
    cfg.task_modality = CustomModality.VISION
    cfg.dataset.dataset_base = "/custom/base"
    cfg.dataset.dataset_uri = "ds"
    cfg.dataset.dataset_version = "v1"
    cfg.model_base = "/custom/mbase"
    cfg.model_name = "mname"
    config_utils.derive_paths(cfg)
    self.assertEqual(
        cfg.dataset.dataset_path, "/custom/base/ds/huggingface_dataset/v1/"
    )
    self.assertEqual(cfg.model_id, "/custom/mbase/mname")

if __name__ == "__main__":
  absltest.main()
