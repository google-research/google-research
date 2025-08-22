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

"""Helpers for defining concept specs.

Used to allow substrates to easily define concept specs within their
configs.
"""
from typing import Any, Dict, Iterable, Optional
import dm_env
import numpy as np

from concept_marl.utils.concept_types import ConceptType
from concept_marl.utils.concept_types import ObjectType


class ConceptArray(dm_env.specs.Array):
  """An array for non-categorical (binary, scalar, and position) concepts.
  """

  __slots__ = ('_shape', '_dtype', '_concept_type', '_object_type', '_name')
  __hash__ = None

  def __init__(self,
               shape,
               dtype,
               concept_type,
               object_type,
               name = None):
    """Initializes a new `Array` spec.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      concept_type: ConceptType held by the array (e.g. binary vs. scalar).
      object_type: Type of the object that this concept pertains to (e.g. agent
        vs. environment object).
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      TypeError: If `shape` is not an iterable of elements convertible to int,
      or if `dtype` is not convertible to a numpy dtype.
    """
    super().__init__(shape=shape, dtype=dtype, name=name)
    self._concept_type = concept_type
    self._object_type = object_type

  @property
  def concept_type(self):
    """Returns a type specifying concept type for the array."""
    return self._concept_type

  @property
  def object_type(self):
    """Returns a type specifying object type for the array."""
    return self._object_type

  def __repr__(self):
    return 'ConceptArray(shape={}, dtype={}, name={})'.format(
        self.shape, repr(self.dtype), repr(self.name))


class CategoricalConceptArray(dm_env.specs.BoundedArray):
  """An array specifically for categorical concepts.

  To keep track of the number of categories the concept holds, this
  overwrites BoundedArray instead of Array.
  """

  def __init__(self,
               shape,
               num_categories,
               object_type,
               dtype,
               name = None):
    """Initializes a new `CategoricalConceptArray` spec.

    Args:
      shape: An iterable specifying the array shape.
      num_categories: number of objects this concept holds (determines the shape
        of the tensor).
      object_type: Type of the object that this concept pertains to (e.g. agent
        vs. environment object).
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.
    """

    super().__init__(
        shape=shape,
        dtype=dtype,
        minimum=0,
        maximum=num_categories - 1,
        name=name)

    self._concept_type = ConceptType.CATEGORICAL
    self._object_type = object_type
    self._num_categories = num_categories

  @property
  def concept_type(self):
    """Returns the type of this concept."""
    return self._concept_type

  @property
  def object_type(self):
    """Returns a type specifying object type for the array."""
    return self._object_type

  @property
  def num_categories(self):
    """Returns the type of this concept."""
    return self._num_categories

  def replace(self, **kwargs):
    """Returns a new copy of `self` with specified attributes replaced.

    Args:
      **kwargs: Optional attributes to replace.

    Returns:
      A new copy of `self`.
    """
    all_kwargs = self._get_constructor_kwargs()
    all_kwargs.update(kwargs)
    return type(self)(**all_kwargs)


def binary_concept(num_agents,
                   num_objects,
                   is_agent,
                   name = None):
  """Returns the spec for a binary (i.e.

  np.int32 tensor with 0/1 values).

  Args:
    num_agents: Number of agents to which these objects pertain.
    num_objects: number of objects this concept holds (determines the shape of
      the tensor).
    is_agent: Whether or not this concept pertains to an agent (vs. env object)
    name: optional name for the spec.
  """
  obj_type = ObjectType.AGENT if is_agent else ObjectType.ENVIRONMENT_OBJECT
  return ConceptArray(  # pytype: disable=wrong-arg-types  # numpy-scalars
      shape=(num_agents, num_objects,),
      dtype=np.int32,
      concept_type=ConceptType.BINARY,
      object_type=obj_type,
      name=name)


def scalar_concept(num_agents,
                   num_objects,
                   is_agent,
                   name = None):
  """Returns the spec for a scalar (i.e.

  np.float32 tensor).

  Args:
    num_agents: Number of agents to which these objects pertain.
    num_objects: number of objects this concept holds (determines the shape of
      the tensor).
    is_agent: Whether or not this concept pertains to an agent (vs. env object)
    name: optional name for the spec.
  """
  obj_type = ObjectType.AGENT if is_agent else ObjectType.ENVIRONMENT_OBJECT
  return ConceptArray(  # pytype: disable=wrong-arg-types  # numpy-scalars
      shape=(num_agents, num_objects,),
      dtype=np.float32,
      concept_type=ConceptType.SCALAR,
      object_type=obj_type,
      name=name)


def categorical_concept(num_agents,
                        num_objects,
                        num_values,
                        is_agent,
                        name = None):
  """Returns the spec for categorical concepts.

  Args:
    num_agents: Number of agents to which these objects pertain.
    num_objects: number of objects this concept holds (determines the shape of
      the tensor).
    num_values: number of objects this concept holds (determines the shape of
      the tensor).
    is_agent: Whether or not this concept pertains to an agent (vs. env object)
    name: optional name for the spec.
  """
  obj_type = ObjectType.AGENT if is_agent else ObjectType.ENVIRONMENT_OBJECT
  return CategoricalConceptArray(  # pytype: disable=wrong-arg-types  # numpy-scalars
      shape=(num_agents, num_objects,),
      num_categories=num_values,
      object_type=obj_type,
      dtype=np.int32,
      name=name)


def position_concept(num_agents,
                     num_objects,
                     is_agent,
                     name = None):
  """Returns the spec for a position (i.e.

  np.int32 tensor with 2 coordinates).

  Args:
    num_agents: Number of agents to which these objects pertain.
    num_objects: number of object positions this concept holds (determines the
      shape of the tensor).
    is_agent: Whether or not this concept pertains to an agent (vs. env object)
    name: optional name for the spec.
  """
  obj_type = ObjectType.AGENT if is_agent else ObjectType.ENVIRONMENT_OBJECT
  return ConceptArray(  # pytype: disable=wrong-arg-types  # numpy-scalars
      shape=(num_agents, num_objects, 2),
      dtype=np.int32,
      concept_type=ConceptType.POSITION,
      object_type=obj_type,
      name=name)
