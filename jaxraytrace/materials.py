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

"""Defines materials for objects."""

import dataclasses
import enum
from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp

from jaxraytrace import color
from jaxraytrace import random_utils
from jaxraytrace import vector


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class HitRecord:
  """Keeps track of ray intersections with objects.

  This follows the implementation in Section 6.3 of
  https://raytracing.github.io/books/RayTracingInOneWeekend.html.
  """

  point: vector.Point
  normal: vector.Vector
  material: "Material"
  t: float

  def tree_flatten(
      self
  ):
    children = (self.point, self.normal, self.material, self.t)
    return (children, None)

  @classmethod
  def tree_unflatten(
      cls, aux_data, children
  ):
    del aux_data
    return cls(*children)


class MaterialTypes(enum.IntEnum):
  """Defines material types with different scattering behavior."""

  DIFFUSE = 0
  REFLECTIVE = 1
  FUZZY = 2


def reflect(direction, normal):
  """Reflects a vector at a point, given the surface normal at this point."""
  return direction - (2 * direction.dot(normal) * normal)


@jax.tree_util.register_pytree_node_class
class Material:
  """Defines a material that obeys certain scattering behavior.

  Unfortunately, having different materials inherit from an abstract Material
  class doesn"t work well with JAX.
  To avoid this, we keep a material_type attribute and select the scattering
  logic based on that.
  """

  def __init__(self,
               material_type,
               material_color,
               fuzz = 0.):
    self.material_type = material_type
    self.color = material_color
    self.fuzz = fuzz

  def diffuse_scatter(
      self,
      ray,
      hit_record,
      rng,
  ):
    """Models scattering off a diffuse surface."""

    del ray
    normal = hit_record.normal
    contact_point = hit_record.point
    random_point = random_utils.random_points_on_sphere(
        radius=1, num=1, rng=rng)
    random_point = random_point.reshape(-1)
    random_point = vector.Point(*random_point)
    scatter_direction = normal + random_point
    scatter_direction = jax.lax.cond(
        jnp.allclose(scatter_direction.array(), 0),
        lambda: normal,
        lambda: scatter_direction,
    )
    scattered_ray = vector.Ray(contact_point, scatter_direction)
    did_scatter = True
    return scattered_ray, self.color, did_scatter

  def reflective_scatter(
      self,
      ray,
      hit_record,
      rng,
  ):
    """Models scattering off a reflective surface."""

    del rng
    normal = hit_record.normal
    contact_point = hit_record.point
    reflected_direction = reflect(ray.direction, normal)
    scattered_ray = vector.Ray(contact_point, reflected_direction)
    did_scatter = True
    return scattered_ray, self.color, did_scatter

  def fuzzy_scatter(
      self,
      ray,
      hit_record,
      rng,
  ):
    """Models scattering off a fuzzy surface."""

    normal = hit_record.normal
    contact_point = hit_record.point
    fuzzyness_offset = random_utils.random_points_in_sphere(
        radius=self.fuzz, num=1, rng=rng)
    fuzzyness_offset = fuzzyness_offset.reshape(-1)
    fuzzyness_offset = vector.Point(*fuzzyness_offset)
    reflected_direction = reflect(ray.direction, normal) + fuzzyness_offset
    scattered_ray = vector.Ray(contact_point, reflected_direction)
    did_scatter = reflected_direction.dot(normal) > 0
    return scattered_ray, self.color, did_scatter

  def scatter(
      self,
      ray,
      hit_record,
      rng,
  ):
    """Models scattering off a surface."""

    return jax.lax.switch(
        self.material_type,
        [self.diffuse_scatter, self.reflective_scatter, self.fuzzy_scatter],
        ray,
        hit_record,
        rng,
    )

  def tree_flatten(
      self):
    children = (self.material_type, self.color, self.fuzz)
    return (children, None)

  @classmethod
  def tree_unflatten(
      cls, aux_data, children):
    del aux_data
    return cls(
        material_type=children[0], material_color=children[1], fuzz=children[2])


def get_material(material_type, material_color,
                 fuzz):
  """Convenience function for creating materials."""

  if material_type == "diffuse":
    material_type = MaterialTypes.DIFFUSE
    fuzz = 0.
  elif material_type == "reflective":
    material_type = MaterialTypes.REFLECTIVE
    fuzz = 0.
  elif material_type == "fuzzy":
    material_type = MaterialTypes.FUZZY
    if fuzz is None:
      raise ValueError("Fuzz value for fuzzy material cannot be None.")
  else:
    raise ValueError(f"Unsupported material: {material_type}.")
  material_color = color.get_color(material_color)
  return Material(material_type, material_color, fuzz)


def create_dummy_material():
  """Creates a dummy material."""
  return Material(MaterialTypes.DIFFUSE, color.get_color("black"), 0.)


def create_hit_record_at_infinity():
  """Returns a dummy HitRecord, at infinity."""
  return HitRecord(
      point=vector.Point(jnp.inf, jnp.inf, jnp.inf),
      normal=vector.Vector(0, 0, 1),
      t=jnp.inf,
      material=create_dummy_material(),
  )
