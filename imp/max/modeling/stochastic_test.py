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

"""Tests for stochastic modules."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from imp.max.modeling import stochastic


class DropTokenTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_rate', 0, False, (10, 20, 30, 40), None),
      ('deterministic', 0.5, True, (10, 20, 30, 40), None),
      ('drop_all', 1, False, (10, 20, 30, 40), (10, 20, 0, 40)),
      ('medium', .5, False, (10, 20, 30, 40), (10, 20, 15, 40)),
      ('most', .9, False, (1, 2, 100, 4), (1, 2, 9, 4)),
      ('4d', .25, False, (4, 10, 20, 40), (4, 10, 15, 40)),
      ('3d', .25, False, (10, 20, 40), (10, 15, 40)),
      ('2d', .25, False, (20, 40), (15, 40)),
  )
  def test_drop_token(self, rate, deterministic, input_shape, output_shape):

    @functools.partial(jax.jit, static_argnums=(1,))
    def _run_forward(inputs, spmd):
      drop_token = stochastic.DropToken(rate=rate, spmd_enabled=spmd)
      variables = drop_token.init(
          rngs={'rng': jax.random.key(1)}, inputs=inputs)
      return drop_token.apply(
          variables=variables,
          rngs={'droptoken': jax.random.key(2)},
          inputs=inputs,
          deterministic=deterministic)

    inputs = jnp.ones(input_shape)
    outputs = _run_forward(inputs, True)
    outputs_non_spmd = _run_forward(inputs, False)
    output_shape = output_shape or input_shape
    chex.assert_equal(jnp.linalg.norm(outputs - outputs_non_spmd), 0.)
    chex.assert_shape(outputs, output_shape)

  @parameterized.named_parameters(
      ('1d', .25, (4,)),
      ('5d', .25, (2, 4, 10, 20, 40)),
  )
  def test_drop_token_with_unsupported_inputs(self, rate, input_shape):

    @functools.partial(jax.jit, static_argnums=(1,))
    def _run_forward(inputs, spmd):
      drop_token = stochastic.DropToken(rate=rate, spmd_enabled=spmd)
      variables = drop_token.init(
          rngs={'rng': jax.random.key(1)}, inputs=inputs)
      return drop_token.apply(
          variables=variables,
          rngs={'droptoken': jax.random.key(2)},
          inputs=inputs,
          deterministic=False)

    inputs = jnp.ones(input_shape)
    with self.assertRaises(ValueError):
      _run_forward(inputs, True)
      _run_forward(inputs, False)

if __name__ == '__main__':
  absltest.main()
