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

"""Tests for train_utils."""
from absl.testing import absltest

from flax import linen as nn
import jax
from jax import numpy as jnp

from wildfire_perc_sim import config
from wildfire_perc_sim import train_utils


def _get_test_model(_, prng):
  model = nn.Conv(16, (5, 5))
  x = jnp.ones((4, 128, 128, 3))

  variables = model.init(prng, x)
  return model, variables


class TrainUtilsTest(absltest.TestCase):

  def test_TrainEvalSetup(self):
    cfg = config.ExperimentConfig()

    setup = train_utils.TrainEvalSetup.create(cfg, _get_test_model, True)

    # Test that initialization happens without any error
    self.assertIsInstance(setup, train_utils.TrainEvalSetup)


if __name__ == '__main__':
  absltest.main()
