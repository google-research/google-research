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
"""Example using opt_list with Jax."""

from absl import app

from flax import nn
import jax
from jax import random
import jax.numpy as jnp

from opt_list import jax_opt_list


class SimpleModel(nn.Module):

  def apply(self, x):
    x = nn.Dense(x, features=256)
    x = nn.relu(x)
    x = nn.Dense(x, features=256)
    x = nn.relu(x)
    return nn.Dense(x, features=2)


def main(_):
  # Define the total number of training steps
  training_iters = 200

  rng = random.PRNGKey(0)

  rng, key = random.split(rng)
  # Construct the model
  _, model = SimpleModel.create_by_shape(key, [((256, 2), jnp.float32)])

  # Create the optimizer corresponding to the 0th hyperparameter configuration
  # with the specified amount of training steps.
  optimizer_def = jax_opt_list.optimizer_for_idx(0, training_iters)
  optimizer = optimizer_def.create(model)

  @jax.jit
  def train_step(optimizer, batch):
    """Train for a single step."""

    def loss_fn(model):
      """A fake loss function."""
      x, y = batch
      return jnp.mean(jnp.square(model(x) - y))

    optimizer, loss = optimizer.optimize(loss_fn)
    return optimizer, loss

  for _ in range(training_iters):
    # make a random batch of fake data
    rng, key = random.split(rng)
    inp = random.normal(key, [512, 2]) / 4.
    target = jnp.tanh(1 / (1e-6 + inp))

    # train the model a step
    optimizer, loss = train_step(optimizer, (inp, target))
    print(loss)


if __name__ == '__main__':
  app.run(main)
