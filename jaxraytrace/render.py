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

"""Rendering library in JAX."""

import functools

from typing import Tuple
import chex
import jax
import jax.numpy as jnp
import ml_collections

from jaxraytrace import camera
from jaxraytrace import color
from jaxraytrace import materials
from jaxraytrace import objects
from jaxraytrace import vector


def correct_gamma(image, gamma):
  """Performs gamma correction."""
  if gamma == 1:
    return image

  if gamma == 2:
    return jnp.sqrt(image)

  return image**(1 / gamma)


def compute_color(
    ray,
    world,
    config,
    rng,
):
  """Computes a color for the ray."""

  # Background.
  def compute_color_at_background(ray):
    mixing_factor = (ray.unit().y() + 1) / 2
    background_1 = color.get_color(config.first_background_color)
    background_2 = color.get_color(config.second_background_color)
    return color.linear_interpolation(background_1, background_2, mixing_factor)

  # We avoid recursion because that leads to long JAX compilation times.
  # Instead, we use a JAX-special while loop,
  # with a terminating condition on recursive depth.
  max_depth = config.max_recursion_depth

  def check_depth(args):
    """Terminating condition on recursive depth."""
    depth = args[2]
    return depth < max_depth

  def ray_trace_at_depth(args):
    """Computes the next ray or final ray color at this depth."""

    def compute_next_ray(
        ray,
        ray_color,
        depth,
        rng,
        hit_record,
    ):
      """Computes the next ray after scattering."""
      depth_rng = jax.random.fold_in(rng, depth)
      ray, sampled_color, did_scatter = hit_record.material.scatter(
          ray, hit_record, rng=depth_rng)
      ray_color, depth = jax.lax.cond(
          did_scatter,
          lambda: (ray_color.elementwise_multiply(sampled_color), depth + 1),
          lambda: (color.Color(*[0, 0, 0]), max_depth),
      )
      return ray, ray_color, depth, rng

    def compute_background_color(
        ray,
        ray_color,
        depth,
        rng,
        hit_record,
    ):
      """Computes the color of a ray hitting the background."""
      del hit_record
      sampled_ray_color = compute_color_at_background(ray)
      ray_color = ray_color.elementwise_multiply(sampled_ray_color)
      depth = max_depth
      return ray, ray_color, depth, rng

    ray = args[0]
    hit_record, did_hit = world.hits(ray, t_min=0.001, t_max=jnp.inf)
    return jax.lax.cond(
        did_hit,
        compute_next_ray,
        compute_background_color,
        *args,
        hit_record,
    )

  init_depth = 0
  init_color = color.Color(*[1, 1, 1])
  ray_color = jax.lax.while_loop(
      check_depth,
      ray_trace_at_depth,
      (ray, init_color, init_depth, rng),
  )[1]
  return ray_color


def generate_image(
    height,
    width,
    scene_camera,
    world,
    config,
):
  """Generates an image of dimensions (height x width x 3) from the given camera."""

  def process_pixel(
      position,
      num_samples,
      rng,
  ):
    j, i = position

    def get_color_at_sample(u, v,
                            sample_rng):
      ray = scene_camera.get_ray(u, v)
      return compute_color_fn(ray, rng=sample_rng).array()

    pixel_rng = jax.random.fold_in(rng, width * i + j)
    pixel_rng, i_rng, j_rng = jax.random.split(pixel_rng, num=3)

    # Random samples for anti-aliasing.
    random_is = jax.random.uniform(i_rng, shape=(num_samples,))
    random_js = jax.random.uniform(j_rng, shape=(num_samples,))

    us = (j + random_js) / width
    vs = (i + random_is) / height
    sample_rngs = jax.random.split(pixel_rng, num=num_samples)

    colors = jax.vmap(get_color_at_sample)(us, vs, sample_rngs)
    colors = jnp.mean(colors, axis=0)
    return colors

  num_samples = config.num_antialiasing_samples
  rng = jax.random.PRNGKey(config.rng_seed)

  compute_color_fn = functools.partial(
      compute_color, world=world, config=config)
  process_pixel_fn = functools.partial(
      process_pixel, num_samples=num_samples, rng=rng)
  process_pixel_fn = jax.vmap(jax.vmap(process_pixel_fn))

  grid = jnp.dstack(jnp.meshgrid(jnp.arange(width), jnp.arange(height)))
  image = process_pixel_fn(grid)
  return image
