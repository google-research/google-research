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

"""Defines a generic Vector class."""

from typing import Union, Tuple
import chex
import jax
import jax.numpy as jnp

Numeric = Union[int, float, complex, jnp.number]


@jax.tree_util.register_pytree_node_class
class Vector:
  """Defines a generic 3D vector."""

  def __init__(self,
               x,
               y,
               z,
               dtype = jnp.float32):
    self.arr = jnp.array([x, y, z], dtype=dtype)

  def x(self):
    return self.arr[0]  # pytype: disable=bad-return-type  # jnp-type

  def y(self):
    return self.arr[1]  # pytype: disable=bad-return-type  # jnp-type

  def z(self):
    return self.arr[2]  # pytype: disable=bad-return-type  # jnp-type

  def array(self):
    return self.arr

  def __repr__(self):
    return f"Vector: {self.arr.__repr__()}"

  def __add__(self, other):
    return Vector(*(self.arr + other.arr))

  def __sub__(self, other):
    return Vector(*(self.arr - other.arr))

  def __mul__(self, other):
    return Vector(*(self.arr * other))

  def __truediv__(self, other):
    return Vector(*(self.arr / other))

  def __rmul__(self, other):
    return Vector(*(self.arr * other))

  def cross(self, other):
    return Vector(*jnp.cross(self.arr, other.arr))

  def dot(self, other):
    return jnp.dot(self.arr, other.arr)  # pytype: disable=bad-return-type  # jnp-type

  def length(self):
    return jnp.sqrt(jnp.sum(jnp.square(self.arr)))  # pytype: disable=bad-return-type  # jax-types

  def unit(self):
    return self / self.length()

  def tree_flatten(self):
    children = self.arr
    return (children, None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class Point(Vector):
  """Defines a point in 3D space."""


@jax.tree_util.register_pytree_node_class
class Ray:
  """Defines a ray."""

  def __init__(self, origin, direction):
    self.origin = origin
    self.direction = direction

  def at(self, t):
    coords = self.origin + t * self.direction
    coords = coords.array()
    return Point(*coords)

  def unit(self):
    return self.direction.unit()

  def tree_flatten(self):
    children = (self.origin, self.direction)
    return (children, None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)
