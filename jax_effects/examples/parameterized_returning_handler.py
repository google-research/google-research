# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Examples of parameterized handlers with return function."""

# Disable lint warnings related to effect operation definitions with @effect.
# pylint: disable=unused-argument

from collections.abc import Sequence

from absl import app
import jax
import jax.numpy as jnp

import jax_effects

effect = jax_effects.effect
effectify = jax_effects.effectify
ParameterizedHandler = jax_effects.ParameterizedHandler

Bool = jax.core.ShapedArray(shape=(), dtype=jnp.bool_)
State = jax.core.ShapedArray(shape=(), dtype=jnp.int32)


def state_example():
  """State effect example."""

  @effect(abstract_eval=lambda _: Bool)
  def state_set(s):
    pass

  @effect(abstract_eval=lambda: State)
  def state_get():
    pass

  def computation_simple(x):
    state_set(x)
    return state_get() + state_get()

  def get_handler(s, x, k, lk):
    return k(s, s)

  def set_handler(s, x, k, lk):
    return k(x, True)

  def simple_state_example(x):
    with ParameterizedHandler(
        0,
        state_get=get_handler,
        state_set=set_handler,
        return_fun=lambda s, x: s + 1,
    ) as state_handler:
      x = computation_simple(x)
      state_handler.result = x
    return state_handler.result

  result = effectify(simple_state_example)(42)
  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(state_example())


if __name__ == "__main__":
  app.run(main)
