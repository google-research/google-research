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

"""The shape abstract property."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, Sequence

import jax

from abstract_nas.abstract import base
from abstract_nas.model import Model
from abstract_nas.model.concrete import Op
from abstract_nas.model.subgraph import SubgraphModel

Tensor = Any


class _ShapeModel(Model):
  """Model class for shape abstract inference."""

  max_size: int = 0

  def exec_op(self, op, input_values,
              deterministic, training, **_):
    output_values = super().exec_op(op, input_values, deterministic, training)
    for idx, output_value in enumerate(output_values):
      output_size = output_value.size
      output_name = f"{op.op_kwargs['name']}:{idx}"
      if self.max_size and output_size > self.max_size:
        raise RuntimeError(f"Output {output_name} has size {output_size}, "
                           f"max_size {self.max_size} exceeded")
    return output_values


class ShapeModel():
  """Wrapper class for shape abstract inference."""

  def __init__(self, model, max_size = 0):
    self.model = _ShapeModel(model.graph, model.constants, max_size=max_size)

  def apply(self,
            input_values,
            state = None):
    if state is None:
      output_values, _ = jax.eval_shape(self.model.init_with_output,
                                        jax.random.PRNGKey(0), input_values)
    else:
      output_values = jax.eval_shape(self.model.apply, state, input_values)
    return output_values


class GraphShapes(base.AbstractGraphProperty):
  """Data structure for tracking the shapes in a graph."""

  def __init__(self, input_shapes,
               output_shapes):
    self.input_shapes = input_shapes
    self.output_shapes = output_shapes

  @classmethod
  def _infer_abstract(cls,
                      model,
                      input_values,
                      state = None,
                      intermediates = False,
                      input_intermediate_values = None,
                      max_size = 0):
    """Infers the shapes of a model given input values (and optional state).

    Args:
      model: model for execution.
      input_values: inputs (for shape inference).
      state: the state of the model.
      intermediates: whether to infer the intermediate tensor shapes as well.
      input_intermediate_values: Any tensors to help with constant resolution.
      max_size: the maximum size of any intermediate tensor.

    Returns:
      The inferred GraphShapes.

    Raises:
      RuntimeError: if the model cannot be executed (or is otherwise malformed).
    """
    if intermediates:
      old_output_names = list(model.graph.output_names)
      model.graph.output_names = []  # get all intermediate shapes

    input_values = dict(input_values)
    if input_intermediate_values:
      input_values.update(input_intermediate_values)

    input_shapes = {
        input_name: input_values[input_name].shape
        for input_name in input_values.keys()
    }

    shape_model = ShapeModel(model, max_size=max_size)
    try:
      output_shapes = shape_model.apply(input_values, state)
    except Exception as e:  # pylint: disable=broad-except
      if "max_size" in str(e):
        raise e
      else:
        raise RuntimeError(f"Malformed graph ({type(e).__name__}: {e}).") from e

    if intermediates:
      model.graph.output_names = old_output_names
      for output_name in old_output_names:
        if ":" not in output_name:
          output_shapes[f"{output_name}"] = output_shapes[f"{output_name}:0"]
          del output_shapes[f"{output_name}:0"]

    output_shapes = {k: v.shape for k, v in output_shapes.items()}
    return GraphShapes(input_shapes, output_shapes)

  @classmethod
  def _infer_concrete(cls,
                      model,
                      input_values,
                      state,
                      intermediates = False,
                      input_intermediate_values = None,
                      max_size = 0):
    # The abstract inference is exact
    return GraphShapes._infer_abstract(model, input_values, state,
                                       intermediates, input_intermediate_values,
                                       max_size)

  @classmethod
  def infer(
      cls,
      model,
      input_values,
      state = None,
      intermediates = False,
      input_intermediate_values = None,
      abstract = True,  # pylint: disable=unused-argument
      max_size = 0):
    """Infers the shapes of the model."""
    # The abstract inference is exact
    return cls._infer_abstract(model, input_values, state, intermediates,
                               input_intermediate_values, max_size)


class ShapeProperty(base.AbstractProperty):
  """Specifies the shapes of the input / output tensor(s) of a computation graph.

  This property specifies the shapes of the input and output tensor(s) for a
  specific instantiation of a computation graph, i.e., even if the computation
  graph supports variable input shapes (like batch dimension), this property
  only specifies the input and output shapes for a specific batch of inputs.

  Attributes:
    graph_shapes: the shape property of the graph.
  """

  def __init__(self,
               graph_shapes = None,
               p = 0.0,
               safety_only = False,
               input_values = None):
    super().__init__(p=p, safety_only=safety_only, input_values=input_values)
    self._graph_shapes: Optional[GraphShapes] = graph_shapes

  @property
  def input_shapes(self):
    assert self._graph_shapes is not None
    return self._graph_shapes.input_shapes

  @property
  def output_shapes(self):
    assert self._graph_shapes is not None
    return self._graph_shapes.output_shapes

  @classmethod
  def infer_inputs(
      cls,
      subgraph_model,
      abstract = True,
      input_values = None
  ):
    """Infers the input shapes of the subgraph."""
    input_values = input_values if input_values else subgraph_model.inputs
    if subgraph_model.subg_inputs_model is not None:
      input_shapes = GraphShapes.infer(
          subgraph_model.subg_inputs_model,
          input_values,
          abstract=abstract)
      input_shapes = input_shapes.output_shapes
    else:
      # no subgraph
      input_shapes = {
          input_name: input_tensor.shape
          for input_name, input_tensor in input_values.items()
      }
    return input_shapes

  @classmethod
  def infer_outputs(
      cls,
      subgraph_model,
      max_size = 0,
      intermediates = False,
      abstract = True,
      input_values = None
  ):
    """Infers the output shapes of the subgraph."""
    input_values = input_values if input_values else subgraph_model.inputs
    output_shapes = GraphShapes.infer(
        subgraph_model.subg_outputs_model,
        input_values,
        intermediates=intermediates,
        abstract=abstract,
        max_size=max_size)
    return output_shapes.output_shapes

  def infer(self,
            subgraph_model,
            max_size = 0,
            intermediates = False,
            abstract = True):
    """Infers the shape property of a subgraph, given some inputs."""
    input_shapes = self.infer_inputs(subgraph_model, abstract,
                                     self.input_values)
    output_shapes = self.infer_outputs(subgraph_model, max_size, intermediates,
                                       abstract, self.input_values)

    # the rewiring should also be reflected
    for node in subgraph_model.subgraph:
      if not node.output_names:
        continue
      for idx, output_name in enumerate(node.output_names):
        if output_name in output_shapes.keys():
          continue
        node_output_name = f"{node.op.name}:{idx}"
        if node_output_name in output_shapes.keys() and node.output_names[idx]:
          output_shapes[output_name] = output_shapes[node_output_name]

    graph_shapes = GraphShapes(input_shapes, output_shapes)
    return ShapeProperty(graph_shapes, self.p, self.safety_only,
                         self.input_values)

  def mutate(self):
    """Mutates the shape property."""
    new_prop = copy.deepcopy(self)
    for key in list(new_prop.output_shapes.keys()):
      if random.random() < self.p:
        del new_prop.output_shapes[key]
    return new_prop

  def distance_from(self, other):
    """Returns the distance to self from the other ShapeProperty.

    In general, we will assume that all tensors have the form:
      (batch_dim, [s1, s2, ...], feature_dim)
    where s1, s2, etc. are spatial dimensions. In particular, all tensors are
    at least 2 dimensional (though the dimensions may be singular).

    We make the assumption that the spatial dimensions can only decrease,
    and also only decrease by integer multiplicative factors. That is, there is
    no "super-resolution" occurring.

    The distance is defined as:
      - UNDEFINED if the input and output names do not match,
      - UNDEFINED if the input shapes do not match,
      - UNDEFINED if the number of dimensions do not match,
      - UNDEFINED if there is a pair of dimensions such that self dimension is
        not an integer multiplicative factor of other dimension
      - the number of mismatched dimensions otherwise.

    Note that this definition makes it difficult to support operations that
    change the number of dimensions (e.g., reshape, or generalized einsum-type
    operations).

    Args:
      other: The other ShapeProperty property.

    Returns:
      The distance.

    Raises:
      ValueError if the distance is undefined.
    """
    # Inputs must match exactly in name and shapes.
    if len(self.input_shapes) != len(other.input_shapes):
      raise ValueError

    for input_name, input_shape in self.input_shapes.items():
      if input_name not in other.input_shapes:
        raise ValueError
      if input_shape != other.input_shapes[input_name]:
        raise ValueError

    if not self.output_shapes:
      return 0

    dist = 0

    # Check all outputs in self.
    # Note that other is allowed to produce extraneous outputs.
    for output_name, output_shape in self.output_shapes.items():
      # All tensors are at least dim 2.
      assert len(output_shape) >= 2

      # Other must produce all the outputs that self does.
      if output_name not in other.output_shapes:
        raise ValueError
      other_shape = other.output_shapes[output_name]

      # These outputs must also have the same number of dims.
      if len(other_shape) != len(output_shape):
        raise ValueError

      # Batch dimensions must be equal.
      if output_shape[0] != other_shape[0]:
        raise ValueError

      # Check spatial dims.
      # 1. If all spatial dims are 1, then can use a squeeze pool.
      check_spatial = False
      for b in output_shape[1:-1]:
        if b != 1:
          check_spatial = True
          break
      # 2. Otherwise, make sure there exists a pooling operation with a square
      # receptive field.
      if check_spatial:
        factor = None
        for a, b in zip(other_shape[1:-1], output_shape[1:-1]):
          if a % b != 0:
            raise ValueError
          if factor and a // b != factor:
            raise ValueError
          factor = a // b

      # Count mismatched dims.
      dist += sum(
          [a != b for a, b in zip(other_shape, output_shape)]
      ) / len(output_shape)

    if self.safety_only: return 0
    return dist / len(self.output_shapes)
