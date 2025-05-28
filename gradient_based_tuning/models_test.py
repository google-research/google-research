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
"""Tests for models.py."""

from absl.testing import absltest
from flax import optim
import numpy as np

from gradient_based_tuning import models


class InitOptimizerByTypeTest(absltest.TestCase):

  def test_init_optimizer_target_adafactor(self):
    optimizer_type = 'adafactor'
    model = np.array([1., 2.])
    opt = models.init_optimizer_by_type(model, optimizer_type)
    self.assertCountEqual(opt.target, model)

  def test_init_optimizer_def_adafactor(self):
    optimizer_type = 'adafactor'
    model = np.array([1., 2.])
    opt = models.init_optimizer_by_type(model, optimizer_type)
    self.assertIsInstance(opt.optimizer_def, optim.Adafactor)

  def test_init_optimizer_def_gd(self):
    optimizer_type = 'gd'
    model = np.array([1., 2.])
    opt = models.init_optimizer_by_type(model, optimizer_type)
    self.assertIsInstance(opt.optimizer_def, optim.GradientDescent)

  def test_init_optimizer_def_sgd(self):
    optimizer_type = 'sgd'
    model = np.array([1., 2.])
    opt = models.init_optimizer_by_type(model, optimizer_type)
    self.assertIsInstance(opt.optimizer_def, optim.GradientDescent)

  def test_init_optimizer_def_gradient_descent(self):
    optimizer_type = 'gradient_descent'
    model = np.array([1., 2.])
    opt = models.init_optimizer_by_type(model, optimizer_type)
    self.assertIsInstance(opt.optimizer_def, optim.GradientDescent)

  def test_init_optimizer_def_adam(self):
    optimizer_type = 'adam'
    model = np.array([1., 2.])
    opt = models.init_optimizer_by_type(model, optimizer_type)
    self.assertIsInstance(opt.optimizer_def, optim.Adam)

  def test_init_optimizer_def_lamb(self):
    optimizer_type = 'lamb'
    model = np.array([1., 2.])
    opt = models.init_optimizer_by_type(model, optimizer_type)
    self.assertIsInstance(opt.optimizer_def, optim.LAMB)

  def test_unrecognized_optimizer_type(self):
    optimizer_type = 'unk'
    model = np.array([1., 2.])
    with self.assertRaisesRegex(ValueError, '(?i)Unrecognized.*unk'):
      _ = models.init_optimizer_by_type(model, optimizer_type)


if __name__ == '__main__':
  absltest.main()
