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

# Lint as: python3
"""Learned optimizer search spaces in Jax using Flax style optimizers!"""
from jax.experimental import optimizers
import jax.numpy as jnp
from opt_list import common
from opt_list import jax_common


@optimizers.optimizer
def optimizer_for_idx(idx, training_steps):
  """Get a nadamw optimizer for the given configuration and training_steps.

  Args:
    idx: int
      The index into the learned optimizer list.
    training_steps: int
      total number of training steps that the model will be trained.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  config = common.get_optimizer_config(idx)
  config['training_steps'] = training_steps
  config['use_bias_correction'] = True  # always true for now.
  hyper_params = jax_common.NAdamWHyperParams(**config)

  def init(x0):
    return x0, jnp.zeros_like(x0), jnp.zeros_like(x0)

  def update(i, g, state):
    x = state[0]
    state = jax_common.NAdamWParamState(*state[1:])
    update, new_s = jax_common.nadamw_update(i, hyper_params, x, state, g)
    new_x = x + update
    return new_x, new_s[0], new_s[1]

  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params
