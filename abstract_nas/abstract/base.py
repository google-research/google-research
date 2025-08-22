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

"""Base class for abstract property inference.

Abstract properties are properties defined for computation (sub)graphs. They are
abstract in the sense that non-equivalent graphs (as defined by input-output
behavior) may have the same value for a given abstract property.

An illustrative example of an abstract property is the shape of the output
tensor (given the shape of the input tensor).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Type, TypeVar

from abstract_nas.model import Model
from abstract_nas.model.subgraph import SubgraphModel

Tensor = Any
AGP = TypeVar("AGP", bound="AbstractGraphProperty")
AP = TypeVar("AP", bound="AbstractProperty")


class AbstractGraphProperty(abc.ABC):
  """Base class for graph abstract property."""

  @classmethod
  @abc.abstractmethod
  def _infer_abstract(cls,
                      model,
                      input_values,
                      state = None,
                      input_intermediate_values = None,
                      intermediates = False):
    """Infers the property of a model using an abstract interpreter."""
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def _infer_concrete(cls,
                      model,
                      input_values,
                      state,
                      input_intermediate_values = None,
                      intermediates = False):
    """Infers the property of a model using a concrete execution."""
    raise NotImplementedError

  @classmethod
  def infer(cls,
            model,
            input_values,
            state = None,
            intermediates = False,
            input_intermediate_values = None,
            abstract = True,
            **kwargs):
    """Infers the property of a model given input values (and optional state)."""
    if abstract:
      return cls._infer_abstract(
          model=model,
          input_values=input_values,
          state=state,
          intermediates=intermediates,
          input_intermediate_values=input_intermediate_values,
          **kwargs)
    else:
      return cls._infer_concrete(
          model=model,
          input_values=input_values,
          state=state,
          intermediates=intermediates,
          input_intermediate_values=input_intermediate_values,
          **kwargs)


class AbstractProperty(abc.ABC):
  """Base class for abstract property."""

  def __init__(self,
               p = 0.0,
               safety_only = False,
               input_values = None):
    if p < 0 or p > 1:
      raise ValueError(f"p must be between 0 and 1 (inclusive), but got {p}.")
    self.p = p
    # safety_only can be used if synthesis does not need to satisfy a property,
    # but does need to ensure that the property is still feasible. For instance,
    # consider synthesizing two subgraphs A and B which will eventually be
    # joined by a binary op A + B. If we are synthesizing A before B, we need
    # to ensure that the output shape of A remains feasible from the input shape
    # of B. Note that this setting is currently not used.
    self.safety_only = safety_only
    self.input_values = input_values

  @abc.abstractmethod
  def infer(self,
            subgraph_model,
            abstract = True):
    """Infers the abstract property.

    Args:
      subgraph_model: The (sub)graph for which to infer the property.
      abstract: Whether to perform the inference using abstract or concrete
        values.

    Returns:
      The inferred abstract property.

    Raises:
      NotImplementedError: the child class must implement this abstract method.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def mutate(self):
    """Mutates the abstract property.

    Raises:
      NotImplementedError: the child class must implement this abstract method.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def distance_from(self, other):
    """Returns the distance to self from the other value.

    The quantity returned is a signed measure of how much more strict this
    value of the abstract property is than the value. For instance, if the
    abstract property represents the devation of a positive integer from 0,
    then a natural distance_from would have:
      3.distance_from(5) = 2
      5.distance_from(3) = -2
      3.distance_from(3) = 0
    Note that this is, in general, neither symmetric nor positive definite.

    See also self.is_satisfied_by(other).

    Args:
      other: The other abstract property.

    Raises:
      NotImplementedError: the child class must implement this abstract method.
    """
    raise NotImplementedError

  def is_satisfied_by(self, other):
    """Returns whether the other value satisfies the abstract property."""
    return self.distance_from(other) <= 0

  def verify(self,
             subgraph_model,
             abstract = True):
    """Returns whether subgraph satisfies the current abstract property."""
    prop = self.infer(subgraph_model, abstract=abstract)
    try:
      return self.is_satisfied_by(prop)
    except Exception:  # pylint: disable=broad-except
      return False
