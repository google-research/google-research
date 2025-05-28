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

"""Entry point for rendering."""

from typing import Any, Mapping, Sequence

from absl import app
from absl import logging
import chex
import jax
import ml_collections
import tensorflow as tf

from jaxraytrace import camera
from jaxraytrace import configuration
from jaxraytrace import materials
from jaxraytrace import objects
from jaxraytrace import output
from jaxraytrace import render
from jaxraytrace import vector

_SUPPORTED_SHAPES = ("sphere",)


def build_camera(config,
                 aspect_ratio):
  """Builds the camera that views the world."""
  return camera.Camera(
      origin=vector.Point(*config.origin),
      view_direction=vector.Vector(*config.view_direction),
      view_up=vector.Vector(*config.view_up),
      vertical_field_of_view=config.vertical_field_of_view,
      aspect_ratio=aspect_ratio,
  )


def build_world(config):
  """Builds the world to be rendered."""

  def build_object(obj):
    if obj["shape"] not in _SUPPORTED_SHAPES:
      raise ValueError(f"Unsupported object: {obj['shape']}.")

    return objects.Sphere(
        center=vector.Point(*obj["center"]),
        radius=obj["radius"],
        material=materials.get_material(obj["material_type"],
                                        obj["material_color"],
                                        obj.get("material_fuzz")),
    )

  return objects.World(
      objects=[build_object(obj) for obj in config.objects.values()])


def render_and_save():
  """Renders the image according to the configuration and saves it to disk."""

  rendering_config = configuration.get_config()
  rendering_config = ml_collections.FrozenConfigDict(rendering_config)
  aspect_ratio = rendering_config.aspect_ratio
  height = rendering_config.height
  width = int(aspect_ratio * height)

  scene_camera = build_camera(rendering_config, aspect_ratio)
  world = build_world(rendering_config)

  # Render.
  logging.info("Tracing rays...")
  render_image_fn = jax.jit(
      render.generate_image,
      static_argnames=["height", "width", "config"])
  image = render_image_fn(height, width, scene_camera, world, rendering_config)

  image = render.correct_gamma(image, gamma=rendering_config.gamma_correction)

  logging.info("Saving to file...")
  output.export_as_ppm(image, rendering_config.output_file)

  return image


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  # Log available devices. This code only supports a single host currently.
  logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  render_and_save()


if __name__ == "__main__":
  app.run(main)
