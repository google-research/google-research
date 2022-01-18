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

"""Tests for imagenet.train."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp

from aqt.jax.imagenet import hparams_config
from aqt.jax.imagenet import models
from aqt.jax.imagenet import train_utils
from aqt.jax.imagenet.configs import resnet50_w4
from aqt.jax.imagenet.configs import resnet50_w4_a4_fixed
from aqt.jax.imagenet.configs.paper import resnet50_bfloat16
from aqt.jax.imagenet.configs.paper import resnet50_w4_a4_auto
from aqt.utils import hparams_utils


class TrainTest(parameterized.TestCase):
  @parameterized.named_parameters(
      dict(
          testcase_name='quantization_none',
          base_config_filename=resnet50_bfloat16),
      dict(
          testcase_name='quantization_weights_only',
          base_config_filename=resnet50_w4),
      dict(
          testcase_name='quantization_weights_and_fixed_acts',
          base_config_filename=resnet50_w4_a4_fixed),
      dict(
          testcase_name='quantization_weights_and_auto_acts',
          base_config_filename=resnet50_w4_a4_auto),
  )  # pylint: disable=line-too-long

  def test_create_model(self, base_config_filename):
    hparams = hparams_utils.load_hparams_from_config_dict(
        hparams_config.TrainingHParams, models.ResNet.HParams,
        base_config_filename.get_config())
    model, state = train_utils.create_model(
        random.PRNGKey(0),
        8,
        224,
        jnp.float32,
        hparams.model_hparams,
        train=True)
    x = random.normal(random.PRNGKey(1), (8, 224, 224, 3))
    y, new_state = model.apply(state, x, mutable=True)
    state = jax.tree_map(onp.shape, state)
    new_state = jax.tree_map(onp.shape, new_state)
    self.assertEqual(state, new_state)
    self.assertEqual(y.shape, (8, 1000))


if __name__ == '__main__':
  absltest.main()
