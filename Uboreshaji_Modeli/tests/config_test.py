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
    self.cfg = config.get_detection_base_config()

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
      dict(
          testcase_name="training_max_steps",
          field_path="training.max_steps",
          expected=-1,
      ),
  )
  def test_default_value(self, field_path, expected):
    value = self.cfg
    for part in field_path.split("."):
      value = getattr(value, part)
    self.assertEqual(value, expected)

  def test_get_base_config_minimalism(self):
    cfg = config.get_base_config()
    self.assertFalse(hasattr(cfg, "matcher"))
    self.assertFalse(hasattr(cfg, "detection"))

  def test_get_asr_base_config(self):
    cfg = config.get_asr_base_config()
    with self.subTest(name="test_task_modality"):
      self.assertEqual(cfg.task_modality, config.TaskModality.AUDIO)
    with self.subTest(name="test_load_from_hub"):
      self.assertTrue(cfg.dataset.load_from_hub)

  def test_get_text_sft_base_config(self):
    cfg = config.get_text_sft_base_config()
    with self.subTest(name="test_task_modality"):
      self.assertEqual(cfg.task_modality, config.TaskModality.TEXT)
    with self.subTest(name="test_load_from_hub"):
      self.assertTrue(cfg.dataset.load_from_hub)
    with self.subTest(name="test_use_lora"):
      self.assertTrue(cfg.training.use_lora)
    with self.subTest(name="test_merge_lora_on_save"):
      self.assertFalse(cfg.training.merge_lora_on_save)


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


class PublicConfigsDerivePathsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("gemma_asr_lora", "configs/gemma_asr_lora_config.py"),
      ("gemma_audio", "configs/gemma_audio_config.py"),
      ("mms_asr", "configs/mms_asr_config.py"),
      ("whisper_asr", "configs/whisper_asr_config.py"),
  )
  def test_public_asr_configs_populate_dataset_path(self, rel_config_path):
    import pathlib  # pylint: disable=g-import-not-at-top

    pkg_dir = pathlib.Path(__file__).parent.parent
    cfg = config_utils.load_config(str(pkg_dir / rel_config_path))
    self.assertNotEmpty(cfg.dataset.dataset_path)
    self.assertNotEmpty(cfg.model_id)

  def test_gemma_asr_lora_config_derives_language_subdir(self):
    import pathlib  # pylint: disable=g-import-not-at-top

    pkg_dir = pathlib.Path(__file__).parent.parent
    cfg = config_utils.load_config(
        str(pkg_dir / "configs/gemma_asr_lora_config.py")
    )
    self.assertTrue(cfg.dataset.dataset_path.endswith("/aka"))

  def test_gemma_asr_lora_config_effective_batch_size(self):
    import pathlib  # pylint: disable=g-import-not-at-top

    pkg_dir = pathlib.Path(__file__).parent.parent
    cfg = config_utils.load_config(
        str(pkg_dir / "configs/gemma_asr_lora_config.py")
    )
    self.assertEqual(cfg.training.batch_size, 4)
    self.assertEqual(cfg.training.gradient_accumulation_steps, 4)


if __name__ == "__main__":
  absltest.main()
