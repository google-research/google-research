# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
"""Tests for config_utils."""

import os
import tempfile

from absl.testing import absltest
import ml_collections

from ipagnn.config import config as config_lib
from ipagnn.lib import config_utils


Config = ml_collections.ConfigDict


class ConfigUtilsTest(absltest.TestCase):

  def test_simple_round_trip(self):
    tempdir = tempfile.mkdtemp()
    filepath = os.path.join(tempdir, 'config.json')

    config = Config()
    config.int = 1
    config.float = 2.0
    config.str = 'value'
    config.unicode = u'value'
    config_utils.save_config(config, filepath)
    restored_config = config_utils.load_config(filepath)
    self.assertTrue(config_utils.equals(config, restored_config))

    config.str = 'different'
    self.assertFalse(config_utils.equals(config, restored_config))

  def test_default_config_round_trip(self):
    tempdir = tempfile.mkdtemp()
    filepath = os.path.join(tempdir, 'config.json')

    config = config_lib.get_config()
    config_utils.save_config(config, filepath)
    restored_config = config_utils.load_config(filepath)
    self.assertTrue(config_utils.equals(config, restored_config))


if __name__ == '__main__':
  absltest.main()
