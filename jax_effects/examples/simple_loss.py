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

"""Example for simple, effect-free loss program."""

from collections.abc import Sequence

from absl import app

import jax_effects

loss = jax_effects.loss
effectify = jax_effects.effectify


def simple_loss_example():
  """Loss effect example."""

  def compute(x, y):
    loss(10)
    res = x + y
    loss(20)
    return res

  compute = effectify(compute)
  result = compute(3, 5)
  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(simple_loss_example())


if __name__ == "__main__":
  app.run(main)
