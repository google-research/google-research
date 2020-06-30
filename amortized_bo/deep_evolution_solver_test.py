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
"""Tests learning evolutionary solver."""
import functools

from absl.testing import parameterized
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import tensorflow.compat.v1 as tf  # tf

from amortized_bo import controller
from amortized_bo import deep_evolution_solver
from amortized_bo import domains
from amortized_bo import simple_ising_model


def build_domain(length=8, vocab_size=4, **kwargs):
  """Creates a `FixedLengthDiscreteDomain` with default arguments."""
  return domains.FixedLengthDiscreteDomain(
      length=length, vocab_size=vocab_size, **kwargs)


def build_small_model(output_size, depth, mode="eval"):
  return deep_evolution_solver.build_model_stax(
      output_size, depth, n_units=5, nlayers=0, mode=mode)


class DeepEvolutionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._solver_cls = deep_evolution_solver.MutationPredictorSolver
    tf.enable_eager_execution()
    self.problem = simple_ising_model.AlternatingChainIsingModel(
        length=20, vocab_size=4)
    self.vocab_size = self.problem.domain.vocab_size
    self.length = self.problem.domain.length

  def test_mutations(self):
    inp = np.random.randint(low=0, high=self.vocab_size - 1, size=(self.length))
    pos_mask = np.zeros((self.length))
    pos_mask[3] = 1

    set_to_two = functools.partial(deep_evolution_solver.set_pos, val=2)

    perturbed = set_to_two(inp, pos_mask)
    diff = perturbed - inp

    perturbed_pos = perturbed[pos_mask == 1]
    self.assertEqual(perturbed_pos, [[2]])
    self.assertEqual(np.sum(diff[pos_mask == 0]), 0)

  def test_apply_mutations(self):
    inp = np.random.randint(low=0, high=self.vocab_size-1,
                            size=(1, self.length))
    pos_mask = np.zeros((1, 1, self.length))
    pos_mask[0, 0, 2] = 1

    mut_types = np.array([[[0, 0.1, 0, 0.9]]])

    mutations = []
    for val in range(4):
      mutations.append(
          functools.partial(deep_evolution_solver.set_pos, val=val))

    mut_types = jnp.argmax(mut_types, -1)
    pos_mask = deep_evolution_solver.one_hot(
        jnp.argmax(pos_mask, -1), self.length)

    permuted_batch = deep_evolution_solver.apply_mutations(
        inp, mut_types, pos_mask, mutations)
    permuted_batch = permuted_batch[-1]

    diff = permuted_batch - inp
    perturbed_pos = permuted_batch[0, pos_mask[0, 0] == 1]
    self.assertEqual(perturbed_pos, np.array([3]))
    self.assertEqual(np.sum(diff[0, pos_mask[0, 0] == 0]), 0)

  def test_gumbel_max_sampler(self):
    rng = jrand.PRNGKey(0)
    # Test normalized logits
    logits = jnp.log(jnp.array([[[0.2, 0.3, 0.5], [0.1, 0.85, 0.05]]]))
    pos, = deep_evolution_solver.gumbel_max_sampler(logits, 1, rng)
    self.assertEqual(np.sum(np.array(pos) - np.array([2, 1])), 0.)

    # Test unnormalized logits
    logits = jnp.log(jnp.array([[[0.2, 0.3, 0.5], [0.1, 0.85, 0.05]]]) * 5)
    pos = deep_evolution_solver.gumbel_max_sampler(logits, 1, rng)
    self.assertEqual(np.sum(np.array(pos) - np.array([2, 1])), 0.)

  def test_solve(self):
    problem = simple_ising_model.AlternatingChainIsingModel(
        length=8, vocab_size=4)
    solver = deep_evolution_solver.MutationPredictorSolver(
        domain=problem.domain)
    controller.run(problem, solver, num_rounds=20, batch_size=10)


if __name__ == "__main__":
  tf.test.main()
