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

"""The linear operator abstract property.

This treats the subgraph as a linear operation and infers several properties.
Roughly speaking, this means that if the operator is in fact linear, then the
properties suffice to uniquely identify the operator.
"""

from __future__ import annotations

import copy
import enum
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from abstract_nas.abstract import base
from abstract_nas.abstract import shape
from abstract_nas.model import Model
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import SubgraphModel

Tensor = Any


class Pairing():
  """Data structure for keeping track of pairing between input / output."""

  class Mapping(enum.IntEnum):
    NONE = 0
    ONE_TO_ONE = 1
    MANY_TO_ONE = 2
    ALL_TO_ONE = 3

  def __init__(self, in_dims, out_dims):
    self.mappings = np.zeros((
        in_dims,
        out_dims,
    ), dtype=int)

  def __setitem__(self, key, value):
    self.mappings[key] = value

  def __getitem__(self, key):
    return self.mappings[key]

  def __str__(self):
    return str(self.mappings)

  def __repr__(self):
    return repr(self.mappings)

  def join(self, other):
    """Joins two pairings.

    Given two paths in a graph between an input and output, this operation joins
    the two properties to give a graph-wide property. This is a symmetric
    operation that returns a new Pairing object.

    Args:
      other: The other pairing with which to join.

    Returns:
      A new Pairing object that represents the joined properties.

    Raises:
      ValueError: If the two pairings do not have the same dimensions.
    """
    if other.mappings.shape != self.mappings.shape:
      raise ValueError("Pairings with different shapes cannot be joined.")
    new_mappings = np.maximum(other.mappings, self.mappings)
    new_pairing = Pairing(self.in_dims, self.out_dims)
    new_pairing.mappings = new_mappings
    return new_pairing

  @property
  def in_dims(self):
    return self.mappings.shape[0]

  @property
  def out_dims(self):
    return self.mappings.shape[1]


class OpPairings():
  """Cache for pairing property of single ops."""

  def __init__(self):
    # {op_hash: {out_dim: {in_dim: Pairing}}}
    self.pairings: Dict[int, Dict[int, Dict[int, Pairing]]] = {}

  @classmethod
  def hash(cls, op, in_shapes):
    """Hashes relevant information for pairings.

    Args:
      op: Op to hash.
      in_shapes: The shapes of the input tensors.

    Returns:
      Hash value.
    """

    kwargs = {}

    def copy_kwargs(from_kv,
                    keys,
                    defaults = None):
      if isinstance(keys, str):
        keys = [keys]
      if not isinstance(defaults, str) and defaults is not None:
        try:
          iter(defaults)
        except TypeError:
          defaults = [defaults]
      if defaults is not None and len(defaults) != len(keys):
        raise ValueError(f"Number of defaults ({len(defaults)}) does not equal "
                         f"number of keys ({len(keys)}).")

      for idx, key in enumerate(keys):
        if defaults is not None:
          kwargs[key] = from_kv.get(key, defaults[idx])
        else:
          kwargs[key] = from_kv[key]

    op_type = op.type
    op_kwargs = op.op_kwargs
    input_kwargs = op.input_kwargs
    if op_type == OpType.DENSE:
      pass

    elif op_type == OpType.DENSE_GENERAL:
      copy_kwargs(op_kwargs, ["axis", "batch_dims"], [-1, ()])

    elif op_type == OpType.CONV:
      if isinstance(op_kwargs["kernel_size"], int):
        kernel_size = [op_kwargs["kernel_size"]]
      else:
        kernel_size = op_kwargs["kernel_size"]
      kernel_size = [1 if ks > 1 else 0 for ks in kernel_size]
      kwargs["kernel_size"] = kernel_size

      feature_group_count = op_kwargs.get("feature_group_count", 1)
      feature_group_count = 1 if feature_group_count > 1 else 0
      kwargs["feature_group_count"] = feature_group_count

    # others

    elif op_type == OpType.ADD:
      pass

    elif op_type == OpType.MUL:
      pass

    elif op_type == OpType.SCALAR_ADD:
      copy_kwargs(op_kwargs, ["const"], [0])

    elif op_type == OpType.SCALAR_MUL:
      copy_kwargs(op_kwargs, ["const"], [1])

    elif op_type == OpType.DOT_GENERAL:
      copy_kwargs(input_kwargs, ["dimension_numbers"])

    elif op_type == OpType.EINSUM:
      copy_kwargs(input_kwargs, ["sum"])

    # nn.attention

    elif op_type == OpType.SELF_ATTENTION:
      pass

    # nn.activation

    elif op_type in [OpType.RELU, OpType.GELU, OpType.SWISH, OpType.SIGMOID]:
      pass

    elif op_type == OpType.SOFTMAX:
      copy_kwargs(input_kwargs, ["axis"], [-1])

    # nn.normalization

    elif op_type == OpType.BATCH_NORM:
      copy_kwargs(input_kwargs, ["axis"], [-1])

    elif op_type == OpType.LAYER_NORM:
      pass

    elif op_type == OpType.GROUP_NORM:
      copy_kwargs(input_kwargs, ["num_groups"], [32])

    # reshape operators

    elif op_type == OpType.RESHAPE:
      copy_kwargs(input_kwargs, ["new_shape"])

    elif op_type == OpType.FLATTEN:
      pass

    elif op_type == OpType.TRANSPOSE:
      copy_kwargs(input_kwargs, ["axes"], [None])

    # nn.stochastic

    elif op_type == OpType.DROPOUT:
      pass

    elif op_type == OpType.STOCH_DEPTH:
      pass

    # nn.pooling

    elif op_type == OpType.AVG_POOL or op_type == OpType.MAX_POOL:
      window_shape = input_kwargs["window_shape"]
      in_shape = in_shapes[0]
      if isinstance(window_shape, int):
        window_shape = [window_shape] * (len(in_shape) - 2)
      new_window_shape = []
      for ws, s in zip(window_shape, in_shape[1:]):
        # We only need to be able to distinguish between all-to-one,
        # many-to-one, and one-to-one behaviors.
        if ws == 0 or ws == s:
          # Global pooling.
          new_window_shape.append(2)
        elif ws == 1:
          # No pooling.
          new_window_shape.append(0)
        else:
          # Local pooling.
          new_window_shape.append(1)
      kwargs["window_shape"] = new_window_shape

    elif op_type == OpType.MEAN:
      copy_kwargs(input_kwargs, ["axis"], [None])

    # new param

    elif op_type == OpType.PARAM:
      pass

    in_dims = tuple(len(in_shape) for in_shape in in_shapes)
    return hash((op.type, in_dims, frozenset(kwargs)))

  def get(self,
          op,
          in_shapes,
          output_idx = None,
          input_idx = None):
    """Returns the Pairing for an Op.

    If output_idx and input_idx are both None, returns all the pairings.
    Otherwise, returns the pairing between the corresponding output and input
    tensors of the op.

    Args:
      op: Op to get pairings for.
      in_shapes: The shapes of the input tensors.
      output_idx: the index of the output tensor to pair.
      input_idx: the index of the input tensor to pair.

    Returns:
      The pairing.

    Raises:
      ValueError: if the op is not recorded in op_pairings.
      ValueError: if only one of output_idx or input_idx is specified.
    """
    key = self.hash(op, in_shapes)
    if key not in self.pairings:
      self._update_pairings(op, in_shapes)

    if output_idx is None or input_idx is None:
      if output_idx is not None or input_idx is not None:
        raise ValueError("output_idx and input_idx must both be either "
                         "specified or None.")
      return self.pairings[key]
    return self.pairings[key][output_idx][input_idx]

  def _update_pairings(self, op, in_shapes):
    """Updates pairings with the property for a single op.

    Args:
      op: The op for which to infer the pairing property.
      in_shapes: the shapes of the input tensors.
    """
    assert len(op.input_names) == len(in_shapes)
    input_values = {
        input_name: jnp.ones(in_shape)
        for input_name, in_shape in zip(op.input_names, in_shapes)
    }
    output_names = [f"{op.name}:{i}" for i in range(op.num_outputs)]
    graph = new_graph(op.input_names, output_names, [op])
    model = Model(graph)
    state = model.init(jax.random.PRNGKey(0), input_values)
    pairings = GraphPairings.infer(
        model, input_values, state, abstract=False).pairings

    new_pairings = {}
    for output_idx, output_name in enumerate(output_names):
      new_pairings[output_idx] = {}
      for input_idx, input_name in enumerate(op.input_names):
        new_pairings[output_idx][input_idx] = pairings[output_name][input_name]

    key = self.hash(op, in_shapes)
    self.pairings[key] = new_pairings


_OP_PAIRINGS = OpPairings()


class _PairingModel(Model):
  """Model class for pairing abstract inference."""

  op_pairings = _OP_PAIRINGS

  def exec_init(self, key,
                value,
                **_):
    if isinstance(value, dict):
      return value
    pairing = Pairing(value.ndim, value.ndim)
    for i in range(value.ndim):
      pairing[i, i] = Pairing.Mapping.ONE_TO_ONE
    return {key: pairing}

  def _exec_pairing(self, input_pairing,
                    op_pairing):
    """Implements the abstract semantics of the pairing property.

    Each pairing refers to a specific input and output slot. For instance,
    input_pairing might refer to A -> B and op_pairing would refer to B -> C.
    Then this method computes the pairing A -> C.

    Args:
      input_pairing: The input properties
      op_pairing: The pairing property of the op

    Returns:
      The pairing property after applying the op.

    Raises:
      ValueError: if the op cannot be applied to the input.
    """
    in_dims = input_pairing.in_dims
    out_dims = op_pairing.out_dims
    if input_pairing.out_dims != op_pairing.in_dims:
      raise ValueError(f"out_dims={input_pairing.out_dims} of input_pairing "
                       f"does not equal in_dims={op_pairing.in_dims} of "
                       "op_pairing")

    new_pairing = Pairing(in_dims, out_dims)
    for in_dim in range(in_dims):
      input_dim_pairing = input_pairing[in_dim]
      for out_dim in range(out_dims):
        output_dim_pairing = op_pairing[:, out_dim]
        for in_dep, out_dep in zip(input_dim_pairing, output_dim_pairing):
          if in_dep == Pairing.Mapping.ALL_TO_ONE:
            new_pairing[in_dim, out_dim] = Pairing.Mapping.ALL_TO_ONE
            break
          if out_dep == Pairing.Mapping.NONE or in_dep == Pairing.Mapping.NONE:
            continue
          dep = max(in_dep, out_dep)
          new_pairing[in_dim, out_dim] = max(new_pairing[in_dim, out_dim], dep)
    return new_pairing

  def exec_op(self, op, input_values,
              deterministic, training,
              **context):
    """Executes an op in the abstract semantics of the pairing property.

    Args:
      op: The op to execute.
      input_values: A sequence of Pairing properties, one for each input slot of
        the op.
      deterministic: Whether to execute the op deterministically (for stochastic
        ops, e.g., dropout).
      training: Whether to execute in training or inference mode (e.g., for
        batchnorm).
      **context: A dictionary of context values.

    Returns:
      A sequence of Pairing properties, one for each output slot of the op.

    Raises:
      ValueError: if the context does not include shapes information.
    """
    if "shapes" not in context:
      raise ValueError("Must pass shapes of intermediate values via context.")
    shapes = context["shapes"]
    input_shapes = [shapes[input_name] for input_name in op.input_names]
    output_shapes = [
        shapes[f"{op.name}:{idx}"] for idx in range(op.num_outputs)
    ]

    if op.type == OpType.IDENTITY:
      return input_values
    if op.type == OpType.NONE:
      assert len(input_shapes) == 1
      assert len(output_shapes) == 1
      output_values = []
      for input_value in input_values:
        output_value = {}
        for input_name, pairing in input_value.items():
          output_value[input_name] = Pairing(pairing.in_dims, pairing.out_dims)
        output_values.append(output_value)
      return output_values

    op_pairings = self.op_pairings.get(op, input_shapes)

    input_names = []
    input_dims = {}
    for input_value in input_values:
      input_names.extend(input_value.keys())
      for input_name, pairing in input_value.items():
        input_dims[input_name] = pairing.in_dims
    input_names = set(input_names)

    output_values = []
    for output_idx, output_shape in enumerate(output_shapes):
      pairings = {}
      for input_idx, input_value in enumerate(input_values):
        for input_name, input_pairing in input_value.items():
          pairing = pairings.get(input_name,
                                 Pairing(pairing.in_dims, len(output_shape)))
          op_pairing = op_pairings[output_idx][input_idx]
          new_pairing = self._exec_pairing(input_pairing, op_pairing)
          pairing = pairing.join(new_pairing)
          pairings[input_name] = pairing
      output_values.append(pairings)
    return output_values


class PairingModel():
  """Wrapper class for pairing abstract inference."""

  def __init__(self, model, max_size = 0):
    self.model = model
    self.pairing_model = _PairingModel(model.graph, model.constants)
    self.max_size = max_size

  def apply(
      self,
      input_values,
      state = None,
      input_intermediate_values = None
  ):
    """Applies the PairingModel to the inputs.

    This API is designed to match flax.module.apply.

    Args:
      input_values: the dictionary of input tensors. These need not be concrete,
        but they must have shape information (e.g., ShapeArray).
      state: An optional dictionary of parameters. These should be concrete if
        supplied.
      input_intermediate_values: Any tensors to help with constant resolution.

    Returns:
      A dictionary of {input_name: {output_name: Pairing}}
    """
    graph_shapes = shape.GraphShapes.infer(
        self.model,
        input_values,
        state,
        abstract=True,
        intermediates=True,
        input_intermediate_values=input_intermediate_values,
        max_size=self.max_size)
    shape_context = {}
    for s in [graph_shapes.input_shapes, graph_shapes.output_shapes]:
      for k, v in s.items():
        if ":" not in k:
          shape_context[f"{k}:0"] = v
        else:
          shape_context[k] = v
    return self.pairing_model.apply({}, input_values, shapes=shape_context)


class GraphPairings(base.AbstractGraphProperty):
  """Data structure for keeping track of pairing for a graph.

  Pairing relationship

  Given an input tensor and an output tensor, we say there is a pairing between
  input_dim and output_dim if for each position in the input_dim, there is at
  least one position in the output_dim which depends on this input position.

  We compute this relationship approximately by taking a 1-d slice of the output
  tensor along output_dim, computing the preimage of these values in the input
  tensor, and checking if the preimage contains at least one element from every
  position of the input_dim. The preimage calculation is performed by taking a
  gradient of the output values with respect to the input values. The actual
  gradient values do not matter, only whether the gradient is non-zero.

  For example, consider a 2d input tensor of shape
    [batch_size, input_features].
  We pass this through a dense layer to get an output tensor of shape
    [batch_size, output_features].

  We need to consider the following possible pairings:
  - input batch_size to output batch_size. Given a 1-d slice along the output
    batch dimension, there is an element in every input batch dimension which
    affects the value of this slice, so these dimensions are paired.
  - input batch_size to output_features. Given a 1-d slice along the output
    feature dimension, there is exactly one element in the input batch dimension
    which affects the value of this slice, namely, the corresponding batch along
    which we took the output slice. These dimensions are not paired.
  - input input_features to output batch_size: similar to input batch_size to
    output_features, these are not paired.
  - input input_features to output output_features: paired, since for any slice
    along the output_features dimension, at least one element for all positions
    along the input_features dimension contributes to the computation.

  Pairing.Mapping

  For paired dimensions, we further distinguish between ALL_TO_ONE, MANY_TO_ONE,
  and ONE_TO_ONE relationships. These are computed by taking the gradient with
  respect to a single element of the output, and seeing how many elements along
  the input_dim contribute to the value. Following the previous example,
  - input batch_size to output batch_size is ONE_TO_ONE because only one element
    along the input batch dimension contributes to any single element of the
    output.
  - input_features to output_features is ALL_TO_ONE, since every element along
   the input features dimension contributes jointly to a single element of the
   output.
  Finally, an example of a MANY_TO_ONE relationship is the pairing of a
  convolution betweeen the input spatial dimensions and the output feature
  dimension.
  """

  def __init__(self, pairings = None):
    # {output_name: {input_name: Pairing}}}
    self.pairings: Dict[str, Dict[str, Pairing]] = pairings if pairings else {}

  @classmethod
  def _infer_concrete(cls,
                      model,
                      input_values,
                      state,
                      intermediates = False,
                      input_intermediate_values = None,
                      max_size = 0):
    """Infers the pairing properties of a function, given some inputs."""
    graph_shapes = shape.GraphShapes.infer(
        model,
        input_values,
        state,
        intermediates=intermediates,
        input_intermediate_values=input_intermediate_values,
        abstract=False,
        max_size=max_size)
    output_shapes = graph_shapes.output_shapes

    # Use the jacobian instead of taking the gradient of the sum of the slice
    # because the sum (hence the gradient) along the feature dim is a constant
    # for softmax. In general, it is too error-prone to design an appropriate
    # function that guarantees the correct behavior for all ops, so the jacobian
    # is just safer.
    def slice_along_dim(inputs, output_name, dim):
      outputs = model.apply(state, inputs, training=False, deterministic=True)
      output_tensor = outputs[output_name]
      output_shape = output_tensor.shape
      idxs = tuple([dim // 2 for dim in output_shape])
      slice_sizes = [1] * len(output_shape)
      slice_sizes[dim] = output_shape[dim]
      sliced = jax.lax.dynamic_slice(output_tensor, idxs, tuple(slice_sizes))
      return jnp.squeeze(sliced)

    grad_along_dim = jax.jacobian(slice_along_dim)

    def center_element(inputs, output_name):
      outputs = model.apply(state, inputs, training=False, deterministic=True)
      output_tensor = outputs[output_name]
      dims = tuple([dim // 2 for dim in output_tensor.shape])
      return output_tensor[dims]

    grad_center = jax.grad(center_element)

    pairings = {}
    for output_name, output_shape in output_shapes.items():
      pairings[output_name] = {}
      for input_name, input_tensor in input_values.items():
        pairings[output_name][input_name] = Pairing(input_tensor.ndim,
                                                    len(output_shape))

    for output_name in output_shapes.keys():
      # grads_center contains the gradient of the center element of output_name,
      # with respect to each of the input_values.
      grads_center: Dict[str, Tensor] = grad_center(input_values, output_name)
      for output_dim in range(len(output_shapes[output_name])):
        # grads_dim contains the gradient of a 1d slice taken from the center of
        # output_name along output_dim, with respect to each of the
        # input_values.
        grads_dim: Dict[str, Tensor] = grad_along_dim(input_values, output_name,
                                                      output_dim)
        for input_name in input_values.keys():
          # a non-zero gradient indicates a dependence
          grads_dim_input: Tensor = grads_dim[input_name] != 0
          grads_dim_input = grads_dim_input.sum(axis=0)
          grads_center_input: Tensor = grads_center[input_name] != 0
          input_dims = list(range(grads_dim_input.ndim))
          for input_dim in input_dims:
            new_dims = list(input_dims)
            del new_dims[input_dim]

            # dim_reduced is a 1d array of length = input_dim. Each entry counts
            # the number of elements along output_dim which depends on the
            # corresponding (n-1)d slice, where n = len(input_dims).
            dim_reduced = jnp.sum(grads_dim_input, axis=tuple(new_dims))

            # all() => for each position in the input_dim
            # dim_reduced > 0 => there is at least one position in the
            #   output_dim which depends on the position in the input_dim.
            if (dim_reduced > 0).all():
              center_reduced = jnp.sum(grads_center_input, axis=tuple(new_dims))
              center_reduced = center_reduced > 0
              if center_reduced.all():
                pairings[output_name][input_name][input_dim][
                    output_dim] = Pairing.Mapping.ALL_TO_ONE
              elif jnp.sum(center_reduced) > 1:
                pairings[output_name][input_name][input_dim][
                    output_dim] = Pairing.Mapping.MANY_TO_ONE
              else:
                pairings[output_name][input_name][input_dim][
                    output_dim] = Pairing.Mapping.ONE_TO_ONE
    return GraphPairings(pairings)

  @classmethod
  def _infer_abstract(cls,
                      model,
                      input_values,
                      state = None,
                      intermediates = False,
                      input_intermediate_values = None,
                      max_size = 0):
    if intermediates:
      old_output_names = list(model.graph.output_names)
      model.graph.output_names = []  # get all intermediate pairings

    pairing_model = PairingModel(model, max_size=max_size)
    input_values = dict(input_values)
    if input_intermediate_values:
      input_values.update(input_intermediate_values)
    output_pairings = pairing_model.apply(input_values, state)

    if intermediates:
      model.graph.output_names = old_output_names
      for output_name in output_pairings:
        if ":" not in output_name:
          output_pairings[f"{output_name}"] = output_pairings[
              f"{output_name}:0"]
          del output_pairings[f"{output_name}:0"]
    return GraphPairings(output_pairings)


class LinopProperty(shape.ShapeProperty):
  """Specifies the properties of a subgraph assuming it is a linear operator."""

  def __init__(self,
               pairings = None,
               p = 0.0,
               safety_only = False,
               input_values = None):
    super().__init__(p=p, safety_only=safety_only, input_values=input_values)
    self._pairings: Optional[GraphPairings] = pairings

  @property
  def pairings(self):
    assert self._pairings is not None
    return self._pairings.pairings

  def infer(self,
            subgraph_model,
            max_size = 0,
            intermediates = False,
            abstract = True):
    """Infers the linear operator property of a subgraph, given some inputs."""
    if not self.input_values:
      if subgraph_model.subg_inputs_model is not None:
        if abstract:
          shape_model = shape.ShapeModel(subgraph_model.subg_inputs_model)
          inputs = shape_model.apply(subgraph_model.inputs)

          old_output_names = subgraph_model.subg_inputs_model.graph.output_names
          subgraph_model.subg_inputs_model.graph.output_names = []
          shape_model = shape.ShapeModel(subgraph_model.subg_inputs_model)
          input_intermediate_values = shape_model.apply(subgraph_model.inputs)
          subgraph_model.subg_inputs_model.graph.output_names = old_output_names
        else:
          inputs = subgraph_model.get_default_subg_inputs()
      else:
        inputs = subgraph_model.inputs
        input_intermediate_values = subgraph_model.get_subg_inputs(
            subgraph_model.inputs, intermediates=True)
      input_intermediate_values.update(subgraph_model.inputs)
    else:
      inputs = self.input_values
      input_intermediate_values = None

    prop = GraphPairings.infer(
        subgraph_model.subg_model,
        inputs,
        state=subgraph_model.state,
        max_size=max_size,
        intermediates=intermediates,
        input_intermediate_values=input_intermediate_values,
        abstract=abstract)

    # the rewiring should also be reflected
    for node in subgraph_model.subgraph:
      if not node.output_names:
        continue
      for idx, output_name in enumerate(node.output_names):
        if output_name in prop.pairings:
          continue
        node_output_name = f"{node.op.name}:{idx}"
        if node_output_name in prop.pairings and node.output_names[idx]:
          prop.pairings[output_name] = prop.pairings[node_output_name]

    return LinopProperty(prop, self.p, self.safety_only, self.input_values)

  def mutate(self):
    """Mutates the linear operator property."""
    new_prop = copy.deepcopy(self)
    pairings = new_prop.pairings
    for output_name in pairings.keys():
      for input_name in list(pairings[output_name].keys()):
        if np.random.rand() < self.p:
          del pairings[output_name][input_name]
    return new_prop

  def distance_from(self, other):
    """Returns the distance to self from the other LinopProperty.

    The distance is defined as the sum over the number of pairwise unsatisfied
    Pairing.Mapping values.

    Args:
      other: The other LinopProperty property.

    Returns:
      The distance.
    """
    if self.safety_only: return 0

    pairings = self.pairings
    others = other.pairings

    dist = 0
    count = 0
    for output_name in pairings.keys():
      for input_name in pairings[output_name].keys():
        mapping = pairings[output_name][input_name].mappings
        if output_name not in others or input_name not in others[output_name]:
          other = 0
        else:
          other = others[output_name][input_name].mappings
        norm = np.sum(mapping > 0)
        norm = norm if norm else 1
        dist += np.sum(mapping > other) / norm
        count += 1
    return dist / (count if count else 1)
