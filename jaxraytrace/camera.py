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

"""Defines a generic Camera for viewing."""

from typing import Tuple
import jax
import jax.numpy as jnp

from jaxraytrace import vector


def _degrees_to_radians(deg):
  """Converts degrees to radians."""
  return jnp.pi * (deg / 180.0)


@jax.tree_util.register_pytree_node_class
class Camera:
  """Defines a generic camera."""

  def __init__(
      self,
      origin,
      view_direction,
      view_up,
      vertical_field_of_view,
      aspect_ratio,
  ):
    self.origin = origin
    self.view_direction = view_direction
    self.view_up = view_up
    self.vertical_field_of_view = vertical_field_of_view
    self.aspect_ratio = aspect_ratio

    viewport_height = 2 * jnp.tan(
        _degrees_to_radians(self.vertical_field_of_view) / 2)
    viewport_width = self.aspect_ratio * viewport_height

    horizontal_unit_vector = self.view_direction.cross(self.view_up).unit()
    vertical_unit_vector = horizontal_unit_vector.cross(
        self.view_direction).unit()

    self.horizontal = viewport_width * horizontal_unit_vector
    self.vertical = viewport_height * vertical_unit_vector
    self.lower_left_corner = (
        self.origin - self.horizontal / 2 - self.vertical / 2 +
        (self.view_direction.unit()))

  def get_ray(self, u, v):
    """Returns the ray emitted from the camera for the given (u, v) coordinates."""
    return vector.Ray(  # pytype: disable=wrong-arg-types  # jax-types
        self.origin,
        self.lower_left_corner + u * self.horizontal + v * self.vertical -
        self.origin,
    )

  def tree_flatten(
      self
  ):
    children = (self.origin, self.view_direction, self.view_up,
                self.vertical_field_of_view, self.aspect_ratio)
    return (children, None)

  @classmethod
  def tree_unflatten(
      cls, aux_data, children
  ):
    del aux_data
    return cls(*children)
