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

"""Interleaved effects example."""

# Disable lint warnings related to effect operation definitions with @effect.
# pylint: disable=unused-argument

from collections.abc import Sequence

from absl import app
import jax
import jax.numpy as jnp

import jax_effects

effect = jax_effects.effect
effectify = jax_effects.effectify
loss = jax_effects.loss
Handler = jax_effects.Handler

Bool = jax.core.ShapedArray(shape=(), dtype=jnp.bool_)


def interleaved_effects_example():
  """Interleaved effects example."""
  @effect
  def outer1(a: int):
    pass

  @effect
  def outer2(a: int):
    pass

  @effect
  def inner1(a: int):
    pass

  def outer1_handler(s, x, k, lk):
    loss(x)
    return k(s + 1, x + 1)

  def outer2_handler(s, x, k, lk):
    jax.debug.print("outer2: s is {}", s)
    jax.debug.print("outer2: x is {}", x)
    loss(x)
    return k(s + 3, x + 3)

  def inner1_handler(s, x, k, lk):
    loss(x)
    return k(s + 2, x + 2)

  def compute(x0):
    with Handler(
        0.0,
        outer1=outer1_handler,
        outer2=outer2_handler,
        parameterized=True,
    ) as outer_handler:
      with Handler(
          100.0,
          inner1=inner1_handler,
          parameterized=True,
      ) as inner_handler:
        x1 = outer1(x0)
        x2 = inner1(x1)
        x3 = outer2(x2)
        inner_handler.result = x3
      inner_state, inner_result = inner_handler.result
      del inner_state
      x5 = outer1(inner_result)
      outer_handler.result = x5
    return outer_handler.result

  result = effectify(compute)(42)
  return result


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(interleaved_effects_example())


if __name__ == "__main__":
  app.run(main)
