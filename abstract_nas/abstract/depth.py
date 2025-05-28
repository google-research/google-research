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

"""The depth abstract property.

For every input / output pair, we compute the maximum number of alternating
linear / non linear ops.

Inputs / outputs from residual connections are ignored.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, Optional

from abstract_nas.abstract import base
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import SubgraphModel


def is_nonlinear(op):
  """Returns whether op is non linear.

  We could infer this property by using some well-selected inputs.. But in
  practice it is easier to hand-specify for each op.

  Args:
    op: The op to check for non-linearity.

  Returns:
    Whether the op is non-linear.

  Raises:
    ValueError: if the op type is not supported.
  """
  if op.type in [
      OpType.DENSE, OpType.DENSE_GENERAL, OpType.CONV, OpType.ADD, OpType.MUL,
      OpType.SCALAR_ADD, OpType.SCALAR_MUL, OpType.DOT_GENERAL,
      OpType.BATCH_NORM, OpType.LAYER_NORM, OpType.GROUP_NORM, OpType.FLATTEN,
      OpType.RESHAPE, OpType.TRANSPOSE, OpType.DROPOUT, OpType.STOCH_DEPTH,
      OpType.AVG_POOL, OpType.MAX_POOL, OpType.MEAN, OpType.PARAM, OpType.EINSUM
  ]:
    return False
  elif op.type in [
      OpType.SELF_ATTENTION, OpType.RELU, OpType.GELU, OpType.SWISH,
      OpType.SIGMOID, OpType.SOFTMAX
  ]:
    return True
  elif op.type == OpType.IDENTITY:
    return None
  else:
    raise ValueError(f"{op.type} not recognized.")


class DepthProperty(base.AbstractProperty):
  """Specifies the depth between every pair of inputs and outputs.

  We define the depth between an input and output to be the maximum length path
  between the two, where the length is the number of alternative linear and
  non-linear layers on the path.

  Attributes:
    depth_map: the map from {input_name: {output_name: depth}}
  """

  def __init__(self,
               depth_map = None,
               p = 0.0,
               delta_max = 2,
               safety_only = False):
    """Initializes the depth property.

    Args:
      depth_map: The depth map of input name to output depths.
      p: The probability of mutating the depth by 1.
      delta_max: The maximum amount of change when mutating.
      safety_only: Whether this property is safety-only mode.
    """
    super().__init__(p=p, safety_only=safety_only, input_values=None)
    # {input_name: {output_name: depth}}
    self.depth_map = depth_map
    assert 0 <= delta_max
    self.delta_max = delta_max

  def infer(
      self,
      subgraph_model,
      abstract = True):  # pylint: disable=unused-argument
    """Infers the depth property of a subgraph, given some inputs."""

    # {op_name: {output_name: depth}}
    depth_map = {}
    # {input_name: {graph_output: {parent_is_nonlinear: depth}}}
    input_to_depth = {}

    # go through ops in reverse topological order
    for op in subgraph_model.subg_graph.ops[::-1]:
      depth = {}
      if op.type == OpType.NONE:
        continue
      op_is_nonlinear = is_nonlinear(op)

      # if single op output is a graph output, the depth is 0
      if op.num_outputs == 1 and op.name in subgraph_model.output_names:
        depth[op.name] = 0
      else:
        # We want the max depth of the op to each graph output, so we need to
        # take the max over each op output
        for i in range(op.num_outputs):
          output_name = f"{op.name}:{i}"
          # if any op output is a graph output, the depth is 0
          if output_name in subgraph_model.output_names:
            depth[output_name] = 0
            continue

          # this happens if there are extra ops in the graph
          # i.e., the graph is A -> B -> C -> D, but output_names is ["C"]
          if output_name not in input_to_depth:
            continue

          # compute depth to each graph output
          output_to_depth = input_to_depth[output_name]
          for graph_output, parent_dict in output_to_depth.items():
            for parent_is_nonlinear, parent_depth in parent_dict.items():
              if op_is_nonlinear is not None:
                if (op_is_nonlinear and not parent_is_nonlinear or
                    not op_is_nonlinear and parent_is_nonlinear):
                  parent_depth += 1
              cur_depth_to_output = depth.get(graph_output, 0)
              depth[graph_output] = max(parent_depth, cur_depth_to_output)

      depth_map[op.name] = depth

      # update input_to_depth
      for input_name in op.input_names:
        if ":" not in input_name:
          input_name = f"{input_name}:0"
        output_to_depth = input_to_depth.get(input_name, {})
        for graph_output, depth_to_output in depth.items():
          nonlinear_to_depth = output_to_depth.get(graph_output, {})
          cur_depth = nonlinear_to_depth.get(op_is_nonlinear, 0)
          nonlinear_to_depth[op_is_nonlinear] = max(cur_depth, depth_to_output)
          output_to_depth[graph_output] = nonlinear_to_depth
        input_to_depth[input_name] = output_to_depth

    filtered = {}
    for input_name in subgraph_model.input_names:
      op_name = input_name.split(":")[0]
      if op_name in depth_map:
        filtered[input_name] = depth_map[op_name]
      else:
        if ":" not in input_name:
          key = f"{input_name}:0"
        else:
          key = input_name
        filtered[input_name] = {}

        parent_linear = False
        for op in subgraph_model.graph.ops:
          found = False
          for idx in range(op.num_outputs):
            if key == f"{op.name}:{idx}":
              parent_linear = not is_nonlinear(op)
              found = True
              break
          if found:
            break

        for output_name in input_to_depth[key].keys():
          filtered[input_name][output_name] = max(
              input_to_depth[key][output_name].get(parent_linear, -1) + 1,
              input_to_depth[key][output_name].get(not parent_linear, 0))

    # the rewiring should also be reflected
    for node in subgraph_model.subgraph:
      if not node.output_names:
        continue
      for idx, output_name in enumerate(node.output_names):
        for input_name, output_to_depth in filtered.items():
          if output_name in output_to_depth:
            continue
          node_output_name = f"{node.op.name}:{idx}"
          if node_output_name in output_to_depth and node.output_names[idx]:
            output_to_depth[output_name] = output_to_depth[node_output_name]
    return DepthProperty(filtered, self.p, self.delta_max, self.safety_only)

  def mutate(self):
    """Mutates the depth property."""
    if self.depth_map is None:
      raise ValueError("self.depth_map not set.")
    new_prop = copy.deepcopy(self)
    for _, output_to_depth in new_prop.depth_map.items():
      for output_name, depth in output_to_depth.items():
        delta = 0
        while delta < self.delta_max:
          if random.random() < self.p:
            delta += 1
          else:
            break
        if random.random() > 0.5:
          delta = -delta
        output_to_depth[output_name] = max(0, depth + delta)
    return new_prop

  def distance_from(self, other):
    """Returns the distance to self from the other DepthProperty.

    The distance is defined as the total excess depth of the current property
    over the other property.

    Args:
      other: The other DepthProperty property.

    Returns:
      The distance.
    """
    if self.safety_only: return 0
    if self.depth_map is None:
      raise ValueError("self.depth_map not set.")

    count = 0
    dist = 0
    for input_name, output_to_depth in self.depth_map.items():
      for output_name, depth in output_to_depth.items():
        if (input_name not in other.depth_map or
            output_name not in other.depth_map[input_name]):
          dist += 1
        else:
          dist += max(
              0, depth - other.depth_map[input_name][output_name]
          ) / (depth if depth else 1)
        count += 1
    return dist / (count if count else 1)
