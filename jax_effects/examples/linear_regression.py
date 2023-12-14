# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Linear regression examples."""

# Disable lint warnings related to effect operation definitions with @effect.
# pylint: disable=unused-argument
# pylint: disable=g-bad-todo

from collections.abc import Sequence
import functools
from typing import TypeVar

from absl import app
import jax
from jax import random
import jax.numpy as jnp
import rich

import jax_effects
from jax_effects import handlers

effect = jax_effects.effect
effectify = jax_effects.effectify
effectify_with_loss = jax_effects.effectify_with_loss
Handler = jax_effects.Handler
handle = jax_effects.handle

loss = jax_effects.loss
choose_enumerate = handlers.choose_enumerate
choose_grad = handlers.choose_grad

T = TypeVar('T')


@effect
def choose(args: T) -> T:
  """Choice effect operation."""  # pytype: disable=bad-return-type


@effect(abstract_eval=lambda xs: xs[0])
def choose_single(args: Sequence[T]) -> T:
  """Choose-one-from-many effect operation."""  # pytype: disable=bad-return-type


def linear_regression(params, x, y):
  """Performs one step of linear regression (using jax-effects)."""
  w_new, b_new = choose(params, name='choose_grad')
  prediction = w_new * x + b_new
  loss(jnp.mean((prediction - y) ** 2))
  return (w_new, b_new)


@jax.jit
def linear_regression_baseline(params, x, y, lr):
  """Performs one step of linear regression (using vanilla JAX)."""
  def loss_fn(params):
    w, b = params
    prediction = w * x + b
    return jnp.mean((prediction - y) ** 2)

  loss_value, grads = jax.value_and_grad(loss_fn)(params)
  new_params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
  return loss_value, new_params


def linear_regression_example(*, jit: bool = False, verbose: bool = False):
  """Linear regression example."""
  lr = 0.001
  w = 0.5
  b = -1.0
  x = jnp.arange(10, dtype=jnp.float32)
  y = jnp.arange(10, dtype=jnp.float32)

  linear_regression_transformed = handle(
      lr,
      choose_grad=choose_grad,
      return_fun=lambda _, x: x,
  )(linear_regression)

  linear_regression_transformed = effectify_with_loss(
      linear_regression_transformed, verbose=verbose
  )
  if jit:
    linear_regression_transformed = jax.jit(linear_regression_transformed)

  def run_linear_regression(params, linear_regression_fn):
    w, b = params
    print(f'w: {w}, b: {b}')
    loss_value = None
    for i in range(5):
      loss_value, (w, b) = linear_regression_fn((w, b), x, y)
      rich.print(f'[bold blue]iteration {i}, loss: {loss_value}')
      print(f'w: {w}, b: {b}')
    print()
    return loss_value, (w, b)

  linear_regression_baseline_with_lr = functools.partial(
      linear_regression_baseline, lr=lr
  )
  rich.print('[bold blue]Linear regression (baseline)')
  run_linear_regression((w, b), linear_regression_baseline_with_lr)

  rich.print('[bold blue]Linear regression (jax_effects)')
  result = run_linear_regression((w, b), linear_regression_transformed)
  return result


def hyperparameter_tuning_example():
  """Learning rate tuning example."""
  key = random.PRNGKey(0)
  d0, d1, d2 = [4, 5, 2]
  w = random.normal(key, (d2, d0), dtype=jnp.float32)
  b = random.normal(key, (1, d1), dtype=jnp.float32)
  x = random.normal(key, (d0, d1), dtype=jnp.float32)
  y = random.normal(key, (d2, d1), dtype=jnp.float32)

  # TODO(jax_effects-team): Support `effectify` on previously-effect-handled
  # functions. This "transformation order invariance" is desirable.
  #
  # Currently, we get an error if the line below is commented:
  # JaxprTypeError: Closed call in_avals mismatch
  # linear_regression = effectify(linear_regression)

  lrs = [0.001, 0.005, 0.00001]

  @jax.jit
  @effectify
  def run_hyperparameter_tuning(params, x, y):
    with Handler(choose_enumerate=choose_enumerate):
      lr = choose_single(lrs, name='choose_enumerate')
      with Handler(lr, choose_grad=choose_grad):
        return linear_regression(params, x, y)

  for i in range(5):
    rich.print(f'[bold blue]Step {i}')
    w, b = run_hyperparameter_tuning((w, b), x, y)
    print(f'w: {w}')
    print(f'b: {b}')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  rich.print('[bold green]Linear regression (JIT)')
  linear_regression_example(jit=True)

  # TODO(jax-effects-team): Requires batching support for call_k / call_lk
  # primitives.
  # rich.print('[bold green]Hyperparameter tuning')
  # hyperparameter_tuning_example()


if __name__ == '__main__':
  app.run(main)
