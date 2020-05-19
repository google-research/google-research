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
"""Tests for the wet lab."""

from absl.testing import absltest
import jax
import jax.numpy as np
import jax.test_util

from grouptesting import wet_lab


class WetLabTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def test_reset_frozen(self):
    num_patients = 10
    wetlab = wet_lab.WetLab(num_patients=num_patients, freeze_diseased=True)
    self.assertIsNone(wetlab.diseased)
    wetlab.reset(self.rng)
    self.assertIsNotNone(wetlab.diseased)
    diseased = np.array(wetlab.diseased)
    self.assertEqual(wetlab.diseased.shape[0], num_patients)
    rng = jax.random.split(self.rng)[0]
    wetlab.reset(rng)
    self.assertTrue(np.all(wetlab.diseased == diseased))

  def test_reset_unfrozen(self):
    num_patients = 10
    wetlab = wet_lab.WetLab(num_patients=num_patients, freeze_diseased=False)
    self.assertIsNone(wetlab.diseased)
    wetlab.reset(self.rng)
    diseased = np.array(wetlab.diseased)
    rng = jax.random.split(self.rng)[0]
    wetlab.reset(rng)
    self.assertFalse(np.all(wetlab.diseased == diseased))

  def test_group_tests_outputs(self):
    num_patients = 10
    wetlab = wet_lab.WetLab(num_patients=num_patients, freeze_diseased=True)
    rngs = jax.random.split(self.rng, 3)
    wetlab.reset(rngs[0])

    num_groups = 5
    groups = jax.random.uniform(rngs[1], shape=(num_groups, num_patients)) < 0.3
    output = wetlab.group_tests_outputs(rngs[2], groups)
    self.assertDtypesMatch(output, wetlab.diseased)
    self.assertEqual(output.shape, (num_groups,))
    self.assertEqual(np.any(wetlab.diseased), np.any(output))


if __name__ == '__main__':
  absltest.main()
