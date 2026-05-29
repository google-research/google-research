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
from models.owlv2 import config


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


if __name__ == "__main__":
  absltest.main()
