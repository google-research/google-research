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

"""Initial Conditions for Wildfire Simulator."""

from typing import Tuple, Optional, Union

import jax
from jax import numpy as jnp
from jax import random
import numpy as np

_FieldShapeType = Union[Tuple[int, int, int], Tuple[int, int]]


def _abs_random_normal(prng,
                       field_shape,
                       mean = 1.0,
                       stdev = 0.25):
  norm_dist = random.normal(prng, field_shape) * stdev + mean
  return jnp.abs(norm_dist)


# Vegetation Density Generator
density_random_normal = _abs_random_normal


def density_bool(prng, field_shape,
                 burnable_p):
  """Generate boolean density."""
  return jnp.less_equal(random.uniform(prng, field_shape),
                        burnable_p).astype('float32')


def density_patchy(prng,
                   field_shape,
                   radius,
                   p_large,
                   p_small,
                   mean = 1.0,
                   stdev = 0.25):
  """Iteratively generate circular patches of vegetation.

  Creates binary patches in the field, by incrementally adding patches of radius
  `radius` until the fraction of cells having vegetation is greater than
  `p_large`. Then the binary patches are converted into gaussian density with
  `mean` and `stdev`. Finally, a mask is applied on the field to ensure the
  fraction of cells having vegetation is less than `p_small`.

  Args:
    prng: jax.random PRNGKeyArray.
    field_shape: Shape of the field.
    radius: Radius of the patches.
    p_large: Add patches to the field until the fraction of cells with
      vegetation crosses this threshold.
    p_small: Upper threshold for the final vegetation density.
    mean: Mean of the vegetation density.
    stdev: Standard Deviation of the vegetation density.

  Returns:
    Field with patchy vegetation.
  """

  density = jnp.zeros(field_shape)
  field_with_patches = jnp.zeros(field_shape, dtype=bool)
  p_actual = 0

  x = jnp.arange(field_shape[0])
  y = jnp.arange(field_shape[1])
  x_grid, y_grid = jnp.meshgrid(x, y)

  while p_actual <= p_large:
    prng, key = random.split(prng)
    loc = jnp.round(random.uniform(key, (2,)) * jnp.array(field_shape))

    x_dist = x_grid - loc[0]
    y_dist = y_grid - loc[1]
    r = jnp.sqrt(x_dist**2 + y_dist**2)

    patch_bool = jnp.less_equal(r, radius)
    field_with_patches = jnp.logical_or(field_with_patches, patch_bool)

    p_actual = jnp.mean(field_with_patches)

  prng, key1, key2 = random.split(prng, 3)
  density = (
      field_with_patches * random.normal(key1, field_shape) * stdev + mean)
  density = density * density_bool(key2, field_shape, p_small)

  return density


# Moisture
moisture_random_normal = _abs_random_normal


# Wind
def wind_uniform(field_shape,
                 components):
  """Generate wind field from constant components."""
  if len(field_shape) == 2:
    assert components.shape == (2,)

    return jnp.reshape(components, (1, 1, 2)) * jnp.ones(field_shape + (2,))
  else:
    assert (components.ndim == 2 and components.shape[1] == 2 and
            components.shape[0] == field_shape[0])

    return (jnp.reshape(components, (components.shape[0], 1, 1, 2)) *
            jnp.ones(field_shape + (2,)))


def location_random(prng,
                    bounds,
                    n_pts = 1,
                    batch_size = None):
  """Generate random coordinates within field."""
  key1, key2 = random.split(prng)
  if batch_size is None:
    return jnp.column_stack(
        (random.randint(key1, (n_pts, 1), bounds[0][0], bounds[0][1]),
         random.randint(key2, (n_pts, 1), bounds[1][0], bounds[1][1])))

  return jnp.concatenate(
      (random.randint(key1, (batch_size, n_pts, 1), bounds[0][0], bounds[0][1]),
       random.randint(key2,
                      (batch_size, n_pts, 1), bounds[1][0], bounds[1][1])),
      axis=-1)


def lit_from_pts(field_shape,
                 fire_pts):
  """Generates lit array from fire coordinates."""
  lit = jnp.zeros(field_shape)
  if len(field_shape) == 2:
    assert fire_pts.ndim == 2

    lit = lit.at[fire_pts[:, 0], fire_pts[:, 1]].set(1)
    return lit
  else:
    assert fire_pts.ndim == 3

    for batch in range(fire_pts.shape[0]):
      lit = lit.at[batch, fire_pts[batch, :, 0], fire_pts[batch, :, 1]].set(1)

    return lit


def terrain_slope(field_shape, azimuth,
                  elevation):
  """Generate terrain with constant given slope in given direction."""

  plane_norm = (-jnp.sin(elevation) * jnp.cos(azimuth),
                -jnp.sin(elevation) * jnp.sin(azimuth), jnp.cos(elevation))

  x = jnp.arange(field_shape[0])
  y = jnp.arange(field_shape[1])
  x_grid, y_grid = jnp.meshgrid(x, y)

  terrain = -(plane_norm[0] * x_grid + plane_norm[1] * y_grid) / plane_norm[2]

  return terrain.T - jnp.amin(terrain)


def terrain_diamond_step(prng, terrain,
                         temp_field_size, size, half, n,
                         roughness, height):
  """Diamond-square algorithm - diamond step."""

  x_, y_ = jnp.meshgrid(
      jnp.arange(half, temp_field_size - 1, size),
      jnp.arange(half, temp_field_size - 1, size))

  offset = ((random.normal(prng, x_.shape) - 0.5) * roughness * height) / (2**n)

  terrain = terrain.at[x_, y_].set(
      (terrain[x_ + half, y_ + half] + terrain[x_ + half, y_ - half] +
       terrain[x_ - half, y_ + half] + terrain[x_ - half, y_ - half]) / 4 +
      offset)

  return terrain


def terrain_square_step(prng, terrain,
                        temp_field_size, size, half, n,
                        roughness, height):
  """Diamond-square algorithm - square step."""
  i_s, j_s = [[] for _ in range(5)], [[] for _ in range(5)]

  for i in range(half, temp_field_size - 1, size):
    for j in range((i + half) % size, temp_field_size, size):
      if i == 0:
        idx = 0
      elif i == temp_field_size - 1:
        idx = 1
      elif j == 0:
        idx = 2
      elif j == temp_field_size - 1:
        idx = 3
      else:
        idx = 4
      i_s[idx].append(i)
      j_s[idx].append(j)

  i_s = [jnp.array(i) for i in i_s]
  j_s = [jnp.array(j) for j in j_s]

  for idx in range(5):
    l = len(i_s[idx])
    if not l:
      continue
    prng, key = random.split(prng)
    offset = ((random.normal(key, (l,)) - 0.5) * roughness * height) / (2**n)

    update_list = (
        terrain[i_s[idx] - half, j_s[idx]],
        terrain[i_s[idx] + half, j_s[idx]],
        terrain[i_s[idx], j_s[idx] - half],
        terrain[i_s[idx], j_s[idx] + half],
    )

    updates = np.mean(
        tuple(x for (i, x) in enumerate(update_list) if i != idx))
    updates = updates + offset
    terrain = terrain.at[i_s[idx], j_s[idx]].set(updates)

  return terrain


def terrain_ds(prng, field_shape,
               height, roughness):
  """Generate random terrain using diamond-square algorithm."""

  assert field_shape[0] == field_shape[1]

  next_pow2 = lambda x: jnp.ceil(jnp.log2(jnp.abs(x)))

  # Padded field size determined by next 2^n + 1
  temp_field_size = int(2**next_pow2(field_shape[0] - 1) + 1)

  # Maximum possible iterations
  iteration_count = int(next_pow2(temp_field_size - 1))

  # Initialize field
  terrain = jnp.zeros((temp_field_size, temp_field_size))

  # Seed corner values
  prng, key1, key2, key3, key4 = random.split(prng, 5)
  terrain = terrain.at[0, 0].set(random.normal(key1) * roughness * height)
  terrain = terrain.at[-1, 0].set(random.normal(key2) * roughness * height)
  terrain = terrain.at[0, -1].set(random.normal(key3) * roughness * height)
  terrain = terrain.at[-1, -1].set(random.normal(key4) * roughness * height)

  # Perform diamond-square algorithm
  size = temp_field_size - 1

  for n in range(iteration_count):
    half = size // 2
    prng, key = random.split(prng)
    terrain = terrain_diamond_step(key, terrain, temp_field_size, size, half, n,
                                   roughness, height)
    prng, key = random.split(prng)
    terrain = terrain_square_step(key, terrain, temp_field_size, size, half, n,
                                  roughness, height)
    size = half

  # Remove padding
  padding = temp_field_size - field_shape[0] - 1

  terrain = terrain[padding:(temp_field_size - 1),
                    padding:(temp_field_size - 1)]

  # Shift terrain vertically by setting lowest point to 0
  return terrain - np.amin(terrain)
