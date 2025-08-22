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

"""Tests for schedules."""

from absl.testing import absltest
from jax import numpy as jnp

from imp.max.optimization import config as opt_config
from imp.max.optimization import schedules


class SchedulesTest(absltest.TestCase):

  def test_schedule(self):
    config = opt_config.PreWarmupCosineDecayLearningRate(init_value=7.)
    schedule = schedules.get_schedule(config)
    value = schedule(jnp.array([0.]))
    self.assertEqual(value, 7.)


if __name__ == '__main__':
  absltest.main()
