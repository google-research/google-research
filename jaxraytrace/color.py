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

"""Defines colors for objects."""

from typing import Tuple, Union
import chex
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

Numeric = Union[int, float, complex, jnp.number]


@register_pytree_node_class
class Color:
  """Defines an RGB color."""

  def __init__(
      self,
      r,
      g,
      b,
  ):
    self.arr = jnp.array([r, g, b], dtype=jnp.float32)

  def r(self):
    return self.arr[0]  # pytype: disable=bad-return-type  # jnp-type

  def g(self):
    return self.arr[1]  # pytype: disable=bad-return-type  # jnp-type

  def b(self):
    return self.arr[2]  # pytype: disable=bad-return-type  # jnp-type

  def array(self):
    return self.arr

  def __repr__(self):
    return f"Color (RGB): {self.arr.__repr__()}"

  def __add__(self, other):
    return Color(*(self.arr + other.arr))

  def __sub__(self, other):
    return Color(*(self.arr - other.arr))

  def __mul__(self, other):
    return Color(*(self.arr * other))

  def __truediv__(self, other):
    return Color(*(self.arr / other))

  def __rmul__(self, other):
    return Color(*(self.arr * other))

  def elementwise_multiply(self, other):
    return Color(*(self.arr * other.arr))

  def tree_flatten(self):
    children = self.arr
    return (children, None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


_NAMED_COLORS = {
    "red": Color(1, 0, 0),
    "green": Color(0, 1, 0),
    "blue": Color(0, 0, 1),
    "skyblue": Color(0.5, 0.7, 1),
    "white": Color(1, 1, 1),
    "gray": Color(0.5, 0.5, 0.5),
    "black": Color(0, 0, 0),
    "brown": Color(0.7, 0.3, 0.3),
    "lightgreen": Color(0.8, 0.8, 0),
}


def get_color(name):
  """Constructs a color by name."""
  return _NAMED_COLORS[name]


def linear_interpolation(c1, c2, mixing_factor):
  """Linear interpolation between two colors."""
  return mixing_factor * c1 + (1 - mixing_factor) * c2
