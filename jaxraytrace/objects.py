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

"""Defines objects in the scene."""

import abc
import dataclasses
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from jaxraytrace import materials
from jaxraytrace import vector


class Hittable(abc.ABC):
  """Defines a hittable object."""

  @abc.abstractmethod
  def hits(self, ray, t_min,
           t_max):
    pass


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class World(Hittable):
  """Defines a world with multiple hittable objects."""

  objects: Sequence[Hittable]

  def hits(self, ray, t_min,
           t_max):
    closest_hit_record = materials.create_hit_record_at_infinity()
    did_hit = False
    for obj in self.objects:
      t_max = closest_hit_record.t
      hit_record, did_hit_object = obj.hits(ray, t_min, t_max)
      closest_hit_record = jax.lax.cond(
          did_hit_object,
          lambda hit_record=hit_record: hit_record,
          lambda: closest_hit_record,
      )
      did_hit |= did_hit_object
    return closest_hit_record, did_hit

  def tree_flatten(self):
    children = (self.objects,)
    return (children, None)

  @classmethod
  def tree_unflatten(cls, aux_data,
                     children):
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Sphere(Hittable):
  """Defines a sphere."""

  center: vector.Point
  radius: float
  material: materials.Material

  def hits(self, ray, t_min,
           t_max):
    """Checks if the ray hits the sphere."""
    oc = ray.origin - self.center
    # We solve the quadratic at^2 + bt + c = 0, following Section 6.2
    # from https://raytracing.github.io/books/RayTracingInOneWeekend.html.
    a = ray.direction.dot(ray.direction)
    b = oc.dot(ray.direction)
    c = oc.dot(oc) - (self.radius**2)
    discriminant = (b**2) - a * c

    t = jax.lax.cond(
        discriminant > 0,
        lambda: -(b + jnp.sqrt(discriminant)) / a,
        lambda: jnp.inf,
    )

    t = jax.lax.cond(
        jax.lax.bitwise_and(t_min <= t, t <= t_max),
        lambda: t,
        lambda: -(b - jnp.sqrt(discriminant)) / a,
    )

    t = jax.lax.cond(
        jax.lax.bitwise_and(t_min <= t, t <= t_max),
        lambda: t,
        lambda: jnp.inf,
    )

    def create_hit_record(t):
      point = ray.at(t)
      normal = (point - self.center).unit()
      return materials.HitRecord(
          point=point, normal=normal, t=t, material=self.material)

    return jax.lax.cond(
        t == jnp.inf,
        lambda t: (materials.create_hit_record_at_infinity(), False),
        lambda t: (create_hit_record(t), True),
        t,
    )

  def tree_flatten(
      self):
    children = (self.center, self.radius, self.material)
    return (children, None)

  @classmethod
  def tree_unflatten(
      cls, aux_data, children):
    del aux_data
    return cls(*children)
