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

"""Examples of handlers with return function."""

# Disable lint warnings related to effect operation definitions with @effect.
# pylint: disable=unused-argument

from collections.abc import Sequence

from absl import app

import jax_effects

effect = jax_effects.effect
effectify = jax_effects.effectify
Handler = jax_effects.Handler


def returning_handler_example():
  """Returning handler example."""

  @effect
  def double(x):
    pass

  @effect
  def triple(x):
    pass

  def id_handler(x, k, lk):
    return k(x)

  def times_six(x):
    with Handler(
        triple=id_handler, return_fun=lambda y: 3 * y
    ) as triple_handler:
      tmp = triple(x)
      with Handler(
          double=id_handler, return_fun=lambda z: 2 * z
      ) as double_handler:
        double_handler.result = double(tmp)
      triple_handler.result = double_handler.result
    return triple_handler.result

  result = effectify(times_six)(10)
  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(returning_handler_example())


if __name__ == "__main__":
  app.run(main)
