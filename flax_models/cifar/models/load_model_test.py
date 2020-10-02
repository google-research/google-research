# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for flax_cifar.models.load_model."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import numpy as np

from flax_models.cifar.models import load_model


class LoadModelTest(parameterized.TestCase):

  # Parametrized because other models will be added in following CLs.
  @parameterized.named_parameters(
      ('WideResnet_mini', 'WideResnet_mini'),
      ('WideResnet_ShakeShake_mini', 'WideResnet_ShakeShake_mini'),
      ('Pyramid_ShakeDrop_mini', 'Pyramid_ShakeDrop_mini'))
  def test_CreateModel(self, model_name):
    model, state = load_model.get_model(model_name, 1, 32, 10)
    self.assertIsInstance(model, flax.nn.Model)
    self.assertIsInstance(state, flax.nn.Collection)
    fake_input = np.zeros([1, 32, 32, 3])
    with flax.nn.stateful(state, mutable=False):
      logits = model(fake_input, train=False)
    self.assertEqual(logits.shape, (1, 10))

  @parameterized.named_parameters(
      ('WideResnet28x10', 'WideResnet28x10'),
      ('WideResnet28x6_ShakeShake', 'WideResnet28x6_ShakeShake'),
      ('Pyramid_ShakeDrop', 'Pyramid_ShakeDrop'))
  def test_ParameterCount(self, model_name):
    # Parameter count from the autoaugment paper models, 100 classes:
    reference_parameter_count = {
        'WideResnet28x10': 36278324,
        'WideResnet28x6_ShakeShake': 26227572,
        'Pyramid_ShakeDrop': 26288692,
    }
    model, _ = load_model.get_model(model_name, 1, 32, 100)
    parameter_count = sum(np.prod(e.shape) for e in jax.tree_leaves(model))
    self.assertEqual(parameter_count, reference_parameter_count[model_name])


if __name__ == '__main__':
  absltest.main()
