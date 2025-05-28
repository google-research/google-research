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

"""Indicator functions for use with DSM."""

import functools
from typing import Optional

import chex
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from pvn import networks


@chex.dataclass
class IndicatorOutput:
  rewards: chex.Array
  pre_threshold: Optional[chex.Array] = None


class StackedNatureDqnIndicator(nn.Module):
  """A network comprised of a stack of indictor networks."""

  num_auxiliary_tasks: int
  width_multiplier: float = 1.0
  tasks_per_module: int = 1
  apply_final_relu: bool = True
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  input_dtype: jnp.dtype = jnp.float32

  def setup(self):
    # We'll require that num_auxiliary_tasks is divisible by num_bins
    # If we didn't do this the output of this module would get multiplied
    # by num_bins and we'll get a shape mismatch down the line.
    if self.num_auxiliary_tasks % self.tasks_per_module != 0:
      raise ValueError(
          'StackedNatureDqnIndicator.num_auxiliary_tasks must be '
          'divisible by StackedMultiplyShiftHashIndicator.tasks_per_module. '
          f'Got num_auxiliary_tasks = {self.num_auxiliary_tasks} and '
          f'tasks_per_module = {self.num_auxiliary_tasks}.'
      )

  @nn.remat
  @nn.compact
  def __call__(self, obs):
    obs = obs.astype(self.input_dtype) / 255.0

    VmapNatureDqnEncoder = nn.vmap(  # pylint: disable=invalid-name
        networks.NatureDqnEncoder,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=0,
        axis_size=self.num_auxiliary_tasks // self.tasks_per_module,
    )
    outputs = VmapNatureDqnEncoder(
        name='encoder',
        width_multiplier=self.width_multiplier,
        num_features=self.tasks_per_module,
        apply_final_relu=self.apply_final_relu,
        dtype=self.dtype,
        param_dtype=self.param_dtype)(obs)  # pyformat: disable

    # Squeeze the last dimension as the NatureDqnEncoder outputs shape (1)
    outputs = jax.lax.reshape(outputs, (self.num_auxiliary_tasks,))
    outputs = jax.lax.stop_gradient(outputs)

    reward_bias = self.param(
        'reward_bias',
        nn.initializers.zeros,
        (self.num_auxiliary_tasks,),
        self.param_dtype,
    )
    outputs = outputs + reward_bias

    rewards = jnp.where(outputs <= 0.0, 0.0, 1.0)
    rewards = jax.lax.stop_gradient(rewards)

    return IndicatorOutput(pre_threshold=outputs, rewards=rewards)


@functools.partial(
    jax.jit, static_argnames=('mersenne_prime_exponent', 'num_bins')
)
@chex.assert_max_traces(5)
def multiply_shift_hash_function(
    x,
    params,
    *,
    mersenne_prime_exponent,
    num_bins,
):  # pyformat: disable
  """Multiply shift hash function."""
  prime = 2**mersenne_prime_exponent - 1

  # a_i * x_i
  result = x * params[1:]
  # \sum_i a_i * x_i mod p
  result = jnp.sum(
      jnp.bitwise_and(result, prime)
      + jnp.right_shift(result, mersenne_prime_exponent)
  )
  # a_0 + \sum_i a_i * x_i mod p
  result = params[0] + result
  # (a_0 + \sum_i a_i * x_i mod p) mod p
  result = jnp.bitwise_and(result, prime) + jnp.right_shift(
      result, mersenne_prime_exponent
  )
  # ((a_0 + \sum_i a_i * x_i mod p) mod p) mod m
  # We'll overload the meaning of "1 bin" to mean a single binary indicator
  # pyformat: disable
  result = jax.lax.select(num_bins == 1,
                          jnp.mod(result, 2),
                          jnp.mod(result, num_bins))
  # Only activate the indicator on a signle value.
  # Average activation is 1/num_bins
  return (result == 0).astype(jnp.uint8)


class StackedMultiplyShiftHashIndicator(nn.Module):
  """Stacked multiply-shift hash function indicator.

  This module uses a multiply-shift universal hash function as defined
  in Carter and Wegman 1979. That is,

    hᵢ(x) = (aⁱ₀ + ∑ⱼ aⁱⱼ ⋅ xⱼ mod p) mod m

  To speed-up the computation we choose p to be a Mersenne prime.
  This way we don't have to perform a division but just a bitwise and
  operation and a right shift.

  Constraints:
    p is prime
    p = 2ˢ - 1
    max(input) < p
    number of bins < p
  """

  num_auxiliary_tasks: int
  mersenne_prime_exponent: int = 13
  target_reward_proportion: float = 0.01

  @nn.compact
  def __call__(self, x):
    chex.assert_type(x, jnp.uint8)
    flat_shape = np.prod(x.shape)

    # Retrieve parameter vector
    hash_init = functools.partial(
        jax.random.randint, minval=0, maxval=2**16 - 1, dtype=jnp.uint16
    )
    hash_params = self.param(
        'hash_params',
        hash_init,
        ((
            self.num_auxiliary_tasks,
            flat_shape + 1,
        )),
    )

    hash_function = functools.partial(
        multiply_shift_hash_function,
        mersenne_prime_exponent=self.mersenne_prime_exponent,
        num_bins=int(1 / self.target_reward_proportion),
    )
    hash_function_vmap = jax.vmap(hash_function, in_axes=(None, 0))
    hashes = hash_function_vmap(x.flatten(), hash_params)

    chex.assert_type(hashes, jnp.uint8)
    chex.assert_shape(hashes, (self.num_auxiliary_tasks,))
    # Flatten one-hot dimension

    # Add stop-grad for good measure
    return IndicatorOutput(rewards=jax.lax.stop_gradient(hashes))
