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

"""Tests for models_resnet."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from flax_models.moco import model_resnet


class ModelsResnetTest(absltest.TestCase):

  def test_resnet50(self):
    num_outputs = 10
    model = model_resnet.ResNet50.partial(num_outputs=num_outputs)
    x = jnp.zeros((1, 224, 224, 3))
    (y_cls, y), params = model.init(jax.random.PRNGKey(0), x)
    self.assertEqual(y_cls.shape, (1, num_outputs))
    self.assertEqual(y.shape, (1, 2048))
    self.assertIn('clf', params)


if __name__ == '__main__':
  absltest.main()
