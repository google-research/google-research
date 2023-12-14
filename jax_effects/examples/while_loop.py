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

"""While loop example."""

# Disable lint warnings related to effect operation definitions with @effect.
# pylint: disable=unused-argument

from collections.abc import Sequence

from absl import app
import jax

import jax_effects

effect = jax_effects.effect
effectify = jax_effects.effectify
handle = jax_effects.handle


def simple_while_loop():
  """Simple while loop example."""

  @effect
  def incr(x: int) -> int:
    """Increments `x`."""  # pytype: disable=bad-return-type

  def while_loop(init_state):
    def cond_fun(state):
      x, _ = state
      return x < 10

    @handle(incr=incr_handler)
    def body_fun(state):
      x, acc = state
      acc_new = x + acc
      jax.debug.print("acc: {}", acc_new)
      return (incr(x), acc_new)

    return jax.lax.while_loop(cond_fun, body_fun, init_state)

  def incr_handler(x, k, lk):
    return k(x + 1)

  result = effectify(while_loop)((0, 0))
  return result


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(simple_while_loop())


if __name__ == "__main__":
  app.run(main)
