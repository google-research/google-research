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
"""Tests for the Simulator."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.test_util
from grouptesting import state
from grouptesting.samplers import exhaustive
from grouptesting.samplers import kernels
from grouptesting.samplers import loopy_belief_propagation
from grouptesting.samplers import sequential_monte_carlo


class SamplersTest(jax.test_util.JaxTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.state = state.State(num_patients=32,
                             num_tests_per_cycle=3,
                             max_group_size=5,
                             prior_infection_rate=0.05,
                             prior_specificity=0.95,
                             prior_sensitivity=0.75)

  @parameterized.parameters(
      [kernels.Gibbs(), kernels.ChopinGibbs(), kernels.Chopin()])
  def test_sequential_monte_carlo(self, kernel):
    num_particles = 100
    sampler = sequential_monte_carlo.SmcSampler(
        num_particles=num_particles, kernel=kernel)
    self.assertIsNone(sampler.particles)
    self.assertIsNone(sampler.particle_weights)
    sampler.produce_sample(self.rng, self.state)
    self.assertIsNotNone(sampler.particles)
    self.assertIsNotNone(sampler.particle_weights)
    self.assertEqual(sampler.particles.shape,
                     (num_particles, self.state.num_patients))

  def test_belief_propagation(self):
    sampler = loopy_belief_propagation.LbpSampler()
    self.assertIsNone(sampler.particles)
    self.assertIsNone(sampler.particle_weights)
    sampler.produce_sample(self.rng, self.state)
    self.assertIsNotNone(sampler.particles)
    self.assertIsNotNone(sampler.particle_weights)
    self.assertEqual(sampler.particles.shape, (1, self.state.num_patients))

  def test_binary_vectors(self):
    num_samples = 10
    bvs = exhaustive.all_binary_vectors(num_samples)
    self.assertEqual(bvs.shape, (2 ** num_samples, num_samples))

    bvs = exhaustive.all_binary_vectors(num_samples, upper_bound=0.56)
    self.assertLessEqual(bvs.shape[0], 2 ** num_samples)
    self.assertEqual(bvs.shape[1], num_samples)

  def test_exhaustive(self):
    # Reduce the number of possible states.
    exh_state = state.State(
        num_patients=8, num_tests_per_cycle=3, max_group_size=5,
        prior_infection_rate=0.05, prior_specificity=0.95,
        prior_sensitivity=0.75)
    sampler = exhaustive.ExhaustiveSampler()
    self.assertIsNone(sampler.particles)
    self.assertIsNone(sampler.particle_weights)
    sampler.produce_sample(self.rng, exh_state)
    self.assertIsNotNone(sampler.particles)
    self.assertIsNotNone(sampler.particle_weights)


if __name__ == '__main__':
  absltest.main()
