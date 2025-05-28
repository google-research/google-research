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

"""Tests for registry."""

import dataclasses

from absl.testing import absltest

from imp.max.config import registry
from imp.max.execution import config as exec_config
from imp.max.execution import executors
from imp.max.modeling import config as mdl_config
from imp.max.modeling import module


@dataclasses.dataclass
class TestExperiment(exec_config.Experiment):
  name: str = 'experiment_test_generic'
  dummy_steps: int = 500
  dummy_path: str = 'path/to/nothing'


@dataclasses.dataclass
class TestModel(mdl_config.Model):
  name: str = 'model_test_generic'
  dummy_d: int = 16
  dummy_rate: float = 0.1


class RegistryTest(absltest.TestCase):

  def tearDown(self):
    super().tearDown()
    registry.Registrar.reset()

  def test_generic_register(self):
    registry.Registrar.register(TestExperiment)
    test_exec_config = registry.Registrar.get_config_by_name(
        'experiment_test_generic')
    self.assertEqual(test_exec_config, TestExperiment)
    self.assertIn('experiment_test_generic',
                  registry.Registrar.config_names())

    registry.Registrar.register(TestModel)
    test_mdl_config = registry.Registrar.get_config_by_name(
        'model_test_generic')
    self.assertEqual(test_mdl_config, TestModel)
    self.assertIn('model_test_generic', registry.Registrar.config_names())

    with self.assertRaises(ValueError):
      registry.Registrar.get_config_by_name('some_unregistered_cfg')

    with self.assertRaises(ValueError):
      registry.Registrar.get_class_by_name('some_unregistered_cls')

  def test_registry_reset(self):

    registry.Registrar.register(TestExperiment)
    registry.Registrar.register(TestModel)
    self.assertNotEmpty(registry.Registrar.config_names())

    registry.Registrar.reset()
    self.assertEmpty(registry.Registrar.config_names())

  def test_model_register_decorator(self):

    @registry.Registrar.register_with_class(module.Model)
    @dataclasses.dataclass
    class SampleTestModel(TestModel):
      name: str = 'model_test_decorator'

    test_mdl_config = registry.Registrar.get_config_by_name(
        'model_test_decorator')
    test_mdl_class = registry.Registrar.get_class_by_name(
        'model_test_decorator')

    self.assertEqual(test_mdl_config, SampleTestModel)
    self.assertEqual(test_mdl_class, module.Model)
    self.assertIn('model_test_decorator', registry.Registrar.config_names())
    self.assertIn('model_test_decorator', registry.Registrar.class_names())

  def test_experiment_register_decorator(self):

    @registry.Registrar.register_with_class(executors.BaseExecutor)
    @dataclasses.dataclass
    class SampleTestExperiment(exec_config.Experiment):
      name: str = 'experiment_test_decorator'
      dummy_steps: int = 500
      dummy_path: str = 'path/to/nothing'

    test_exec_config = registry.Registrar.get_config_by_name(
        'experiment_test_decorator')
    test_exp_class = registry.Registrar.get_class_by_name(
        'experiment_test_decorator')

    self.assertEqual(test_exec_config, SampleTestExperiment)
    self.assertEqual(test_exp_class, executors.BaseExecutor)
    self.assertIn('experiment_test_decorator',
                  registry.Registrar.config_names())
    self.assertIn('experiment_test_decorator',
                  registry.Registrar.class_names())


if __name__ == '__main__':
  absltest.main()
