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

"""Class for a random mutator."""

import math
import random
from typing import Sequence

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.evolution.mutator.base import AbstractMutator
from abstract_nas.model.concrete import Graph
from abstract_nas.model.subgraph import SubgraphNode
from abstract_nas.model.subgraph import SubgraphSpec
from abstract_nas.utils import canonicalize_tensor_name


class RandomMutator(AbstractMutator):
  """Class for random mutator.

  This class selects a random connected subgraph, i.e., every op has either
    - at least one input consumed by another op in the graph, or
    - at least one output produced by another op in the graph.
  Furthermore, the subgraph consists of a single component (in the usual sense).
  """

  def __init__(self,
               properties,
               p,
               max_len = 0,
               max_perc = 0):
    """Initializes the RandomSequentialMutator.

    Args:
      properties: The properties to mutate.
      p: The probability of increasing the size of the subgraph.
      max_len: The maximum length of the subgraph. If <= 0, then no maximum.
      max_perc: The maximum length of the subgraph as a percentage of the input
        graph length. If <= 0, then no maximum.

    Raises:
      ValueError: if p is not a probability between 0 and 1.
      ValueError: if max_perc is not a probability between 0 and 1.
      ValueError: if both max_len and max_perc are set.
    """
    super().__init__(properties)
    if p < 0 or p > 1:
      raise ValueError("p must be between 0 and 1 (inclusive), "
                       f"but had value {p}.")
    if max_len > 0 and max_perc > 0:
      raise ValueError("Only one of max_len and max_perc may be set.")
    if max_perc < 0 or max_perc > 1:
      raise ValueError("max_perc must be between 0 and 1 (inclusive), "
                       f"but had value {max_perc}.")

    self.p = p
    self.max_len = max_len
    self.max_perc = max_perc

  def select_subgraph(self, graph):
    """Selects a subgraph for mutation."""
    ops_to_add = list(graph.ops)
    subgraph_ops = []
    produced_outputs = []
    consumed_inputs = []

    max_len = self.max_len
    if max_len <= 0:
      max_len = math.ceil(self.max_perc * len(graph.ops))
    max_len = max(max_len, 1)

    # Sampling loop.
    while True:
      # Select a random op and add it to the current subgraph.
      op = random.choice(ops_to_add)
      subgraph_ops.append(op)

      # Update the outputs produced and the inputs consumed by the current
      # subgraph.
      for idx in range(op.num_outputs):
        produced_outputs.append(f"{op.name}:{idx}")
      for input_name in op.input_names:
        consumed_inputs.append(input_name)

      ops_to_add = []
      # Find all ops which are neighbors of subgraph_ops (which is the current
      # subgraph).
      for op in graph.ops:
        # Skip any ops which are already in the subgraph.
        if op in subgraph_ops:
          continue

        # If any of the outputs of op are consumed by the subgraph, it is a
        # neighbor.
        for idx in range(op.num_outputs):
          if f"{op.name}:{idx}" in consumed_inputs:
            ops_to_add.append(op)
            break
        if op in ops_to_add: continue

        # If any of the inputs of op are produced by the subgraph, it is a
        # neighbor.
        for input_name in op.input_names:
          if input_name in produced_outputs:
            ops_to_add.append(op)
            break

      # Break if maximum length of subgraph has been reached.
      if max_len > 0 and len(subgraph_ops) >= max_len:
        break

      # Break with probability (1-p).
      if random.random() > self.p:
        break

    # Get all externally visible outputs, i.e., all tensors which are produced
    # by ops inside the subgraph, and consumed by ops outside the subgraph.
    externally_visible_outputs = {
        canonicalize_tensor_name(n) for n in graph.output_names
    }
    for op in graph.ops:
      if op in subgraph_ops: continue
      for input_name in op.input_names:
        if input_name in produced_outputs:
          externally_visible_outputs.add(input_name)

    # Create the subgraph spec.
    # N.B. adding the subgraph_ops in order by graph.ops preserves the
    # topological sort.
    subgraph_spec = []
    for op in graph.ops:
      if op not in subgraph_ops: continue
      output_names = []
      for idx in range(op.num_outputs):
        if f"{op.name}:{idx}" in externally_visible_outputs:
          output_names.append(f"{op.name}:{idx}")
        else:
          output_names.append(None)
      subg_node = SubgraphNode(op, output_names=output_names)
      subgraph_spec.append(subg_node)

    return subgraph_spec
