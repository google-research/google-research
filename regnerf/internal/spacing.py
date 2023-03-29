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

"""Helper functions for things like linspace."""
import jax.numpy as jnp
import oryx


def genspace(start, stop, num, fn=None):
  """A generalization of linspace(), geomspace(), and NeRF's "lindisp".

  Behaves like jnp.linspace(), except it allows an optional function that
  "curves" the values to make the spacing between samples not linear.
  If no `fn` value is specified, genspace() is equivalent to jnp.linspace().
  If fn=jnp.log, genspace() is equivalent to jnp.geomspace().
  If fn=jnp.reciprocal, genspace() is equivalent to NeRF's "lindisp".

  Args:
    start: float tensor. The starting value of each sequence.
    stop: float tensor. The end value of each sequence.
    num: int. The number of samples to generate for each sequence.
    fn: function. A jnp function handle used to curve `start`, `stop`, and the
      intermediate samples.

  Returns:
    A tensor of length `num` spanning [`start`, `stop`], according to `fn`.
  """
  fn = fn or (lambda x: x)

  # Linspace between the curved start and stop values.
  t = jnp.linspace(0., 1., num)
  s = fn(start) * (1. - t) + fn(stop) * t

  if fn == (lambda x: x):
    inv_fn = fn
  else:
    # Ask oryx for the inverse of `fn`.
    inv_fn = oryx.core.inverse(fn)

  # Apply `inv_fn` and clamp to the range of valid values.
  return jnp.clip(inv_fn(s), jnp.minimum(start, stop), jnp.maximum(start, stop))
