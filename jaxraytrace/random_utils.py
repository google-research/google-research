# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utilities for generating random numbers."""

import chex
import jax
import jax.numpy as jnp


def random_points_in_sphere(radius,
                            num,
                            rng):
  """Returns a random point sampled uniformly inside a sphere."""
  coords_rng, scale_rng = jax.random.split(rng)

  coords = random_points_on_sphere(radius, num, coords_rng)
  scale = jax.random.uniform(scale_rng, shape=(num,))
  scale = jnp.cbrt(scale)
  coords *= scale[:, jnp.newaxis]
  return coords


def random_points_on_sphere(radius,
                            num,
                            rng):
  """Returns a random point sampled uniformly on a sphere."""
  coords = jax.random.normal(rng, shape=(num, 3))
  magnitude = jnp.sqrt(jnp.sum(jnp.square(coords), axis=-1, keepdims=True))
  coords /= magnitude
  coords *= radius
  return coords
