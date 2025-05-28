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

"""Tests for experiment."""

from absl.testing import absltest
from absl.testing import parameterized

from imp.max.config import registry
from imp.max.execution import config as exec_config
from imp.max.projects.imp.config import experiment as exp_config


class ExperimentTest(parameterized.TestCase):

  @parameterized.parameters(
      ('imp_base.img.train', exp_config.ImpBaseImgTrainExperiment),
      ('imp_base.img.eval', exp_config.ImpBaseImgEvalExperiment),
      ('imp_base.all.train', exp_config.ImpBaseAllTrainExperiment),
      ('imp_base.all.eval', exp_config.ImpBaseAllEvalExperiment),
      (
          'search.imp_base.v1.img.train',
          exp_config.SearchImpBaseImgV1TrainExperiment,
      ),
      (
          'search.imp_base.v1.img.eval',
          exp_config.SearchImpBaseImgV1EvalExperiment,
      ),
      ('sparse_moe_imp_base.img.train',
       exp_config.SparseMoeImpBaseImgTrainExperiment),
      ('sparse_moe_imp_base.img.eval',
       exp_config.SparseMoeImpBaseImgEvalExperiment),
  )
  def test_can_create_config(
      self,
      name,
      config):
    config = config()
    self.assertEqual(config.name, name)
    registry_config = registry.Registrar.get_config_by_name(name)
    self.assertIsInstance(config, registry_config)

if __name__ == '__main__':
  absltest.main()
