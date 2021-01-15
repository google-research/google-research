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

"""Tests for ES-ENAS objects."""

from absl.testing import absltest
from absl.testing import parameterized
import pyglove as pg
from es_enas import config as config_util
from es_enas import objects


class ObjectsTest(parameterized.TestCase):

  def setUp(self):
    self.base_config = config_util.get_config()
    super().setUp()

  @parameterized.named_parameters(
      ('Random', 'random'), ('PolicyGradient', 'policy_gradient'),
      ('RegularizedEvolution', 'regularized_evolution'))
  def test_GeneralTopologyBlackboxObject(self, controller_str):
    self.base_config.controller_type_str = controller_str
    self.base_config.horizon = 2
    self.base_config.environment_name = 'Pendulum'

    self.config = config_util.generate_config(
        self.base_config, current_time_string='TEST')

    self.config.setup_controller_fn()
    self.object = objects.GeneralTopologyBlackboxObject(self.config)

    optimizer = self.config.es_blackbox_optimizer_fn(
        self.object.get_metaparams())

    params = self.object.get_initial()
    topology_str = pg.to_json(self.config.controller.propose_dna())
    core_hyperparams = optimizer.get_hyperparameters()
    hyperparams = [0] + list(core_hyperparams)
    self.object.execute_with_topology(params, topology_str, hyperparams)


if __name__ == '__main__':
  absltest.main()
