# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for train."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized

from action_angle_networks import train
from action_angle_networks.configs import action_angle_flow
from action_angle_networks.configs import action_angle_mlp
from action_angle_networks.configs import euler_update

_ALL_CONFIGS = {
    'action_angle_flow': action_angle_flow.get_config(),
    'action_angle_mlp': action_angle_mlp.get_config(),
    'euler_update': euler_update.get_config(),
}


class TrainTest(parameterized.TestCase):

  @parameterized.parameters('action_angle_flow', 'action_angle_mlp',
                            'euler_update')
  def test_train_and_evaluate(self, config_name):
    # Load config.
    config = _ALL_CONFIGS[config_name]
    config.num_train_steps = 5

    # Create a temporary directory where metrics are written.
    workdir = tempfile.mkdtemp()

    # Training should proceed without any errors.
    train.train_and_evaluate(config, workdir)


if __name__ == '__main__':
  absltest.main()
