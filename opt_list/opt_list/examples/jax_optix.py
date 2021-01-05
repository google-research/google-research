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

# Lint as: python3
"""Example using opt_list with Jax and optix style optimizers."""

from absl import app

import jax
from jax import random
from jax.experimental import optix
from jax.experimental import stax
import jax.numpy as jnp

from opt_list import jax_optix_opt_list


def main(_):
  # Define the total number of training steps
  training_iters = 200

  rng = random.PRNGKey(0)

  rng, key = random.split(rng)

  init_random_params, model_apply = stax.serial(
      stax.Dense(256), stax.Relu, stax.Dense(256), stax.Relu, stax.Dense(2))

  # init the model
  _, params = init_random_params(rng, (-1, 2))

  # Create the optimizer corresponding to the 0th hyperparameter configuration
  # with the specified amount of training steps.
  # opt = optix.adam(1e-4)
  opt = jax_optix_opt_list.optimizer_for_idx(0, training_iters)

  opt_state = opt.init(params)

  @jax.jit
  def loss_fn(params, batch):
    x, y = batch
    y_hat = model_apply(params, x)
    return jnp.mean(jnp.square(y_hat - y))

  @jax.jit
  def train_step(params, opt_state, batch):
    """Train for a single step."""
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = value_and_grad_fn(params, batch)

    # Note this is not the usual optix api as we additionally need parameter
    # values.
    # updates, opt_state = opt.update(grad, opt_state)
    updates, opt_state = opt.update_with_params(grad, params, opt_state)

    new_params = optix.apply_updates(params, updates)
    return new_params, opt_state, loss

  for _ in range(training_iters):
    # make a random batch of fake data
    rng, key = random.split(rng)
    inp = random.normal(key, [512, 2]) / 4.
    target = jnp.tanh(1 / (1e-6 + inp))

    # train the model a step
    params, opt_state, loss = train_step(params, opt_state, (inp, target))
    print(loss)


if __name__ == '__main__':
  app.run(main)
