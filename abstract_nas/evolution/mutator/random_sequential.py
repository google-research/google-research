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

"""Class for a random sequential mutator."""

import math
import random
from typing import Sequence

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.evolution.mutator.sequential import AbstractSequentialMutator
from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import SubgraphNode
from abstract_nas.model.subgraph import SubgraphSpec


UNSUPPORTED_OP_TYPES = [
    # Do not support ops which change the number of dimensions.
    OpType.RESHAPE, OpType.FLATTEN,
    # Do not support ops which could potentially increase spatial resolution.
    OpType.DENSE_GENERAL, OpType.TRANSPOSE,
    # Do not support ops which have tricky constants.
    OpType.SCALAR_MUL,
]


class RandomSequentialMutator(AbstractSequentialMutator):
  """Base class for random sequential mutator."""

  def __init__(self,
               properties,
               p,
               allow_multi_outputs = False,
               max_len = 0,
               max_perc = 0):
    """Initializes the RandomSequentialMutator.

    If allow_multi_outputs is set to True, then we may select ops which produce
    multi outputs, provided exactly one output is actually consumed. The other
    outputs are "dangling" in the sense that they are computed but not used.

    Args:
      properties: The properties to mutate.
      p: The probability of increasing the size of the subgraph.
      allow_multi_outputs: Whether to select ops with multiple outputs.
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
    self.allow_multi_outputs = allow_multi_outputs
    self.max_len = max_len
    self.max_perc = max_perc

  def select_sequential_subgraph(self, graph):
    max_len = self.max_len
    if max_len <= 0:
      max_len = math.ceil(self.max_perc * len(graph.ops))
    max_len = max(max_len, 1)

    if self.p == 0:
      # uniform random from [1, max_len]
      max_len = random.randrange(max_len) + 1

    # first, filter for ops with a single input and a single output
    # or, ops with a single input and multi outputs (if allowed)
    ops = [
        op for op in graph.ops
        if (len(op.input_names) == 1 and
            (op.num_outputs == 1 or self.allow_multi_outputs))
    ]
    ops = [op for op in ops if op.type not in UNSUPPORTED_OP_TYPES]
    if not ops:
      raise ValueError("No sequential subgraphs exist.")

    # start with a random op
    op = random.choice(ops)
    subgraph_ops = [op]

    idx = ops.index(op)
    next_idx = idx + 1
    next_ok = next_idx < len(ops)
    prev_idx = idx - 1
    prev_ok = prev_idx >= 0

    # sampling loop
    while True:
      # break if maximum length of subgraph has been reached
      if max_len > 0 and len(subgraph_ops) >= max_len:
        break

      # break with probability (1-p)
      if self.p > 0 and random.random() > self.p:
        break

      # check if output of previous node is input to first node of subgraph
      if prev_ok and prev_idx >= 0:
        if f"{ops[prev_idx].name}:0" != subgraph_ops[0].input_names[0]:
          prev_ok = False
      if prev_idx < 0: prev_ok = False

      # check if output of last node of subgraph node is input to next node
      if next_ok and next_idx < len(ops):
        if f"{subgraph_ops[-1].name}:0" != ops[next_idx].input_names[0]:
          next_ok = False
      if next_idx >= len(ops): next_ok = False

      if not prev_ok and not next_ok:
        break

      # either previous node is not an option, or
      # both nodes are an option, then we randomly select the next node
      if (not prev_ok) or (prev_ok and next_ok and random.random() < 0.5):
        subgraph_ops.append(ops[next_idx])
        next_idx += 1
      # otherwise, we take the previous node
      else:
        subgraph_ops.insert(0, ops[prev_idx])
        prev_idx -= 1

    # now create the subgraph spec
    subgraph_spec = [SubgraphNode(op) for op in subgraph_ops]
    output_op = subgraph_ops[-1]

    # select one of the outputs of the output_op to rewire
    # we need to make sure that the output selected is actually consumed
    idxs = set()

    # add outputs that are consumed by other ops
    for op in graph.ops:
      for idx in range(output_op.num_outputs):
        for input_name in op.input_names:
          if input_name == f"{output_op.name}:{idx}":
            idxs.add(idx)

    # add outputs that are externally visible (i.e., outputs of the graph)
    for output_name in graph.output_names:
      if output_name == output_op.name:
        idxs.add(0)
      else:
        for idx in range(output_op.num_outputs):
          if output_name == f"{output_op.name}:{idx}":
            idxs.add(idx)

    output_idx = random.choice(list(idxs))
    subgraph_spec[-1].output_names = [f"{output_op.name}:{output_idx}"]

    return subgraph_spec
