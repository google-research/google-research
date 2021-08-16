# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for model_export_utils."""

import os
from absl.testing import absltest
from non_semantic_speech_benchmark.export_model import model_export_utils


TEST_DIR = 'non_semantic_speech_benchmark/data_prep/testdata/12321'


class ModelExportUtilsTest(absltest.TestCase):

  def setUp(self):
    super(ModelExportUtilsTest, self).setUp()
    self.exp_dir = os.path.join(absltest.get_default_test_srcdir(), TEST_DIR)

  def test_get_experiment_dirs(self):
    dirs = model_export_utils.get_experiment_dirs(self.exp_dir)
    self.assertEqual(
        dirs,
        ['1-al=0.005,ap=True,bd=5,cop=True,lr=0.0001,ms=debug,qat=False,tbs=2'])

  def test_get_params(self):
    params = model_export_utils.get_params(
        '1-al=0.005,ap=True,bd=5,cop=True,lr=0.0001,ms=debug,qat=False,tbs=2')
    self.assertEqual(
        params,
        {'al': 0.005, 'ap': True, 'bd': 5, 'cop': True, 'lr': 0.0001,
         'ms': 'debug', 'qat': False, 'tbs': 2})

  def test_get_default_compressor(self):
    compressor = model_export_utils.get_default_compressor()
    self.assertIsNotNone(compressor)

if __name__ == '__main__':
  absltest.main()
