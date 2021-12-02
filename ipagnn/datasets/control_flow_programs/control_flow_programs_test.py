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

"""Tests for control_flow_programs.py."""

import tempfile

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest
import ml_collections
import tensorflow.compat.v1 as tf

from ipagnn.config import config as config_lib
from ipagnn.lib import dataset_utils


class ControlFlowProgramsTest(absltest.TestCase):

  def test_dataset_standard_batching(self):
    dataset_name = 'control_flow_programs/decimal-L10'

    data_dir = tempfile.mkdtemp()
    config = config_lib.get_config()

    with config.unlocked():
      config.dataset.name = dataset_name
      config.dataset.in_memory = True
      config.dataset.batch_size = 5
      config.dataset.representation = 'trace'
    config = ml_collections.FrozenConfigDict(config)

    dataset_info = dataset_utils.get_dataset(data_dir, config)
    item = next(iter(dataset_info.dataset))

    self.assertEqual(item['cfg']['data'].shape[0], 5)

  def test_dataset_multivar(self):
    dataset_name = 'control_flow_programs/decimal-multivar-templates-train-L10'

    data_dir = tempfile.mkdtemp()
    config = config_lib.get_config()

    with config.unlocked():
      config.dataset.name = dataset_name
      config.dataset.in_memory = True
      config.dataset.batch_size = 5
      config.dataset.representation = 'code'
      config.dataset.max_length = 1001
    config = ml_collections.FrozenConfigDict(config)

    dataset_info = dataset_utils.get_dataset(data_dir, config)
    item = next(iter(dataset_info.dataset))

    self.assertEqual(item['cfg']['data'].shape[0], 5)


if __name__ == '__main__':
  tf.enable_eager_execution()
  absltest.main()
