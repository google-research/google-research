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

"""Test to ensure gadgets can be trained without errors."""

import functools

from absl.testing import absltest
import jax
import jax.numpy as jnp
import optax

from gumbel_max_causal_gadgets import experiment_util
from gumbel_max_causal_gadgets import gadget_1
from gumbel_max_causal_gadgets import gadget_2


def logit_pair_distribution_fn(rng, dim, base_scale=.1, noise_scale=.1):
  p_rng, q_rng = jax.random.split(rng, 2)
  p_base = jnp.arange(dim) - (dim - 1.0) / 2
  q_base = -p_base
  p_logits = base_scale * p_base + noise_scale * jax.random.normal(
      p_rng, (dim,))
  q_logits = base_scale * q_base + noise_scale * jax.random.normal(
      q_rng, (dim,))
  return p_logits, q_logits


def maximal_coupling_loss_matrix_fn(logits1, logits2):
  del logits2
  return 1.0 - jnp.eye(logits1.shape[0])


S_DIM = 4
Z_DIM = 5


class GadgetTest(absltest.TestCase):

  def test_gadget_one(self):
    ex = experiment_util.CouplingExperimentConfig(
        name="Gadget 1 test experiment",
        model=gadget_1.GadgetOneMLPPredictor(
            S_dim=S_DIM, hidden_features=[32, 32], relaxation_temperature=1.0),
        logit_pair_distribution_fn=functools.partial(
            logit_pair_distribution_fn,
            dim=S_DIM,
            base_scale=.1,
            noise_scale=0.4),
        coupling_loss_matrix_fn=maximal_coupling_loss_matrix_fn,
        inner_num_samples=2,
        batch_size=3,
        use_transpose=True,
        tx=optax.adam(1e-5),
        num_steps=10,
        print_every=2,
    )
    result = ex.train(jax.random.PRNGKey(0))
    self.assertEqual(result.finished_reason, "done")

  def test_gadget_two(self):
    ex = experiment_util.CouplingExperimentConfig(
        name="Gadget 2 test experiment",
        model=gadget_2.GadgetTwoMLPPredictor(
            S_dim=S_DIM,
            Z_dim=Z_DIM,
            hidden_features=[32, 32],
            relaxation_temperature=1.0,
            learn_prior=False),
        logit_pair_distribution_fn=functools.partial(
            logit_pair_distribution_fn,
            dim=S_DIM,
            base_scale=.1,
            noise_scale=0.4),
        coupling_loss_matrix_fn=maximal_coupling_loss_matrix_fn,
        inner_num_samples=2,
        batch_size=3,
        use_transpose=False,
        tx=optax.adam(1e-5),
        num_steps=10,
        print_every=2,
    )
    result = ex.train(jax.random.PRNGKey(0))
    self.assertEqual(result.finished_reason, "done")


if __name__ == "__main__":
  absltest.main()
