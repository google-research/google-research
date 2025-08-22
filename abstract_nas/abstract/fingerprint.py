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

"""A fingerprint for checking the functional equivalence of two models."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Sequence

import flax.linen as nn
from jax import lax
import jax.numpy as jnp

from abstract_nas.abstract.shape import GraphShapes
from abstract_nas.model import Model
from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.utils import canonicalize_tensor_name

Tensor = Any

OP_FNS = {
    OpType.DENSE: nn.Dense,
    OpType.DENSE_GENERAL: nn.DenseGeneral,
    OpType.CONV: nn.Conv,
    OpType.ADD: None,
    OpType.MUL: None,
    OpType.SCALAR_ADD: None,
    OpType.SCALAR_MUL: None,
    OpType.DOT_GENERAL: lax.dot_general,
    OpType.EINSUM: jnp.einsum,
    OpType.SELF_ATTENTION: nn.SelfAttention,
    OpType.RELU: nn.relu,
    OpType.GELU: nn.gelu,
    OpType.SWISH: nn.swish,
    OpType.SIGMOID: nn.sigmoid,
    OpType.SOFTMAX: nn.softmax,
    OpType.BATCH_NORM: nn.BatchNorm,
    OpType.LAYER_NORM: nn.LayerNorm,
    OpType.GROUP_NORM: nn.GroupNorm,
    OpType.RESHAPE: jnp.reshape,
    OpType.FLATTEN: None,
    OpType.TRANSPOSE: jnp.transpose,
    OpType.DROPOUT: nn.Dropout,
    OpType.STOCH_DEPTH: None,
    OpType.AVG_POOL: nn.avg_pool,
    OpType.MAX_POOL: nn.max_pool,
    OpType.MEAN: jnp.mean,
    OpType.PARAM: None,
    OpType.NONE: None,
    OpType.IDENTITY: None,
}


def _get_default_args(f):
  signature = inspect.signature(f)
  return {
      k: v.default
      for k, v in signature.parameters.items()
      if v.default is not inspect.Parameter.empty
  }


def get_full_kwargs(op):
  """Gets the full op_kwargs and input_kwargs for the op."""

  op_type = op.type
  if op_type not in OP_FNS:
    raise ValueError(f"op_type {op_type} not supported...")
  op_fn = OP_FNS[op_type]

  if op_fn is None:
    op_kwargs = {}
    input_kwargs = {}
  elif inspect.isclass(op_fn):
    assert issubclass(op_fn, nn.Module)
    op_kwargs = _get_default_args(op_fn)
    input_kwargs = _get_default_args(op_fn.__call__)
  else:
    op_kwargs = {}
    input_kwargs = _get_default_args(op_fn)

  op_kwargs.update(op.op_kwargs)
  del op_kwargs["name"]
  input_kwargs.update(op.input_kwargs)
  return op_kwargs, input_kwargs


class _FingerprintModel(Model):
  """Model class for fingerprinting the computation graph.

  To compute the fingerprint, instead of actually executing ops, we instead hash
  a functional representation of the op. Tensors in the regular computation
  graph are then replaced by hashes representing the computation thus far. The
  possibility of collision should be extremely low. Computation graphs which are
  equivalent up to a reordering should always return the same hash, i.e., it
  does not matter the order in which the ops are computed in the flattened
  graph.

  Note however, that we do NOT check for any functional rewrites, e.g.,
    fingerprint((a + b) + c) != fingerprint((c + a) + b).
  In fact, this does not even know that the "+" operator is commutative, so
    fingerprint((a + b)) != fingerprint((b + a))

  Finally, the names that matter are those of the inputs and outputs of the
  graph; in particular, the names of the ops do NOT matter.

  More explicitly,
  - inputs to the fingerprint model are hashes of the respective input names
  - instead of executing an op, we instead hash a tuple consisting of the op
  type, the kwargs passed to the op, and the input values (hashes), preserving
  the order of the inputs.
  - the hashes corresponding to the output tensors of the graph are then hashed
  together with their names to produce a single fingerprint for the entire
  graph.
  """

  def exec_op(self, op, input_values, *_,
              **__):
    """Hash an op with the given inputs."""
    if op.type == OpType.NONE or op.type == OpType.IDENTITY:
      return input_values

    op_kwargs, input_kwargs = get_full_kwargs(op)
    outputs = [
        hash((op.type, frozenset(op_kwargs.items()),
              frozenset(input_kwargs.items()), frozenset(input_values), idx))
        for idx in range(op.num_outputs)
    ]
    return outputs

  def resolve_op(self, op, *args, **kwargs):
    op = super().resolve_op(op, *args, **kwargs)
    op.name = f"hashed/{op.name}"
    op.input_names = [f"hashed/{input_name}" for input_name in op.input_names]
    return op


def fingerprint_graph(graph,
                      constants,
                      input_values,
                      state = None):
  """Returns a fingerprint for functional equivalence."""

  # Get shape info for resolving ops.
  shapes = GraphShapes.infer(
      Model(graph, constants),
      input_values=input_values,
      state=state,
      intermediates=True,
      abstract=True)

  # Save original output names.
  output_names = graph.output_names

  # Augment constants with shapes.
  constants = dict(constants) if constants else {}
  for input_name in graph.input_names:
    input_name = canonicalize_tensor_name(input_name)
    constants[f"hashed/{input_name}"] = hash(input_name)
  constants.update(shapes.input_shapes)
  constants.update(shapes.output_shapes)

  # Output hash values.
  graph.output_names = [f"hashed/{output_name}" for output_name in output_names]

  # Get fingerprints for each output tensor.
  fingerprints = _FingerprintModel(graph, constants).apply({}, input_values)

  # Restore original output names.
  graph.output_names = output_names

  # Return hash of outputs.
  return hash(frozenset(fingerprints.items()))
