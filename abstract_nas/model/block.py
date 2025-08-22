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

"""A parameterized block."""

from __future__ import annotations

from collections import abc
from typing import Any, Dict, Optional, Sequence

from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.utils import canonicalize_tensor_name

Tensor = Any


def _prefix_symbolic(maybe_iter, prefix,
                     constants,
                     updated_names):
  """Recursively prefixes named symbolic constants."""
  if not prefix: return maybe_iter

  if not isinstance(maybe_iter, str) and isinstance(maybe_iter, abc.Iterable):
    return tuple([
        _prefix_symbolic(a, prefix, constants, updated_names)
        for a in maybe_iter
    ])
  v = maybe_iter
  if not (isinstance(v, str) and (v.startswith("K:") or v.startswith("S:"))):
    return v

  k = v.split("*")[0].split("%")[0]
  if k.startswith("K:"):
    # before    "K:T" => constants[T]
    # after     "K:{prefix}T" => constants[{prefix}T]
    if k[2:] in constants:
      v = f"K:{prefix}{v[2:]}"
  else:
    # before    "S:T:#" => intermediate_value[T].shape[#]
    # after     "S:{prefix}T:#" => intermediate_value[{prefix}T].shape[#]
    assert v.startswith("S:")

    v = v[2:]
    if ":" in v:
      arr = v.split(":")
      key = ":".join(arr[:-1])
      try:
        key = int(key)
      except ValueError:
        key = updated_names.get(key, f"{prefix}{key}")
      v = f"{key}:{arr[-1]}"
    v = f"S:{v}"
  return v


class Block():
  """A (sub)graph which is a parameterized block."""

  def __init__(self,
               name,
               graph,
               constants = None,
               base_graph = None,
               base_constants = None):
    self.name = name
    self.graph = graph
    self.constants = constants if constants else {}
    self.base_graph = base_graph if base_graph else self.graph
    self.base_constants = base_constants if base_constants else self.constants

  def instantiate(self,
                  input_names,
                  instance_id = None,
                  constants = None):
    """Instantiates a version of the block with unique names.

    This method uses the names of graph and constants from the initial
    definition of the block (__init__) , so that one can instantiate from any
    derived block with same effect, e.g., if we have:
      init_block = block.__init__(name="conv_layer", ...)
      block0 = init_block.instantiate(instance_id=0, ...)
    then:
      block1 = init_block.instantiate(instance_id=1, ...)
    will have the same effect as:
      block1 = block0.instantiate(instance_id=1, ...)
    The one caveat is that the default values for unspecified constants are
    inherited from the instantiating block (instead of the initial definition).

    Args:
      input_names: The input tensor names the instantiated block will consume.
      instance_id: An id to make the names in the instantiated block unique.
        The id should be unique within a graph.
      constants: Updated parameters for the instantiated block.

    Returns:
      An instantiated block.

    Raises:
      ValueError: if the number of input names provided does not equal the
        number of inputs consumed by the graph.
    """
    if len(input_names) != len(self.base_graph.input_names):
      raise ValueError("Wrong number of inputs provided.")

    prefix = ""
    if self.name: prefix += self.name
    if instance_id is not None: prefix += str(instance_id)
    if prefix: prefix += "/"

    if not constants: constants = dict(self.base_constants)

    new_input_names = input_names
    updated_names = {
        o: n for o, n in zip(self.base_graph.input_names, new_input_names)
    }
    inputs_names = [
        canonicalize_tensor_name(n) for n in self.base_graph.input_names
    ]
    updated_names.update({o: n for o, n in zip(inputs_names, new_input_names)})

    # Update ops.
    new_ops = []
    for op in self.base_graph.ops:
      # Update all input tensor names.
      # Any internal inputs (i.e., anything that is not a graph input) needs to
      # be updated with the prefix.
      new_inputs = []
      for inp in op.input_names:
        try:
          idx = inputs_names.index(inp)
          new_inputs.append(new_input_names[idx])
        except ValueError:
          new_inputs.append(f"{prefix}{inp}")

      # Update symbolic constant names in input_kwargs and op_kwargs.
      new_kwargs = []
      for kwargs in [op.input_kwargs, op.op_kwargs]:
        nk = {
            k: _prefix_symbolic(v, prefix, constants, updated_names)
            for k, v in kwargs.items()
        }
        new_kwargs.append(nk)

      new_ops.append(
          new_op(op_name=f"{prefix}{op.name}",
                 op_type=op.type,
                 input_names=new_inputs,
                 input_kwargs=new_kwargs[0],
                 op_kwargs=new_kwargs[1],
                 num_outputs=op.num_outputs))

    # Update constants and prefix symbolic constant names.
    old_constants = dict(self.base_constants)
    if constants: old_constants.update(constants)
    new_constants = {f"{prefix}{k}": v for k, v in old_constants.items()}

    # Prefix graph output names.
    new_output_names = [f"{prefix}{on}" for on in self.base_graph.output_names]

    graph = new_graph(
        ops=new_ops, input_names=new_input_names, output_names=new_output_names)
    return Block(name=self.name, graph=graph, constants=new_constants,
                 base_graph=self.base_graph, base_constants=old_constants)
