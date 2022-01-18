# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tracing functionality for Python control flow graph execution."""

import copy

import dataclasses
from typing import Dict, List, NewType, Text

from absl import logging  # pylint: disable=unused-import

# Branch decisions:
TRUE_BRANCH = 0
FALSE_BRANCH = 1
NO_BRANCH_DECISION = 2

Values = NewType("Values", Dict[Text, int])


@dataclasses.dataclass
class Trace:
  """The sequence of statements that were run during a program execution.

  A Trace represents the sequence of statements that were run during the
  execution of a program and the branch decisions made along the way.

  cfg_node_index_values and statement_branch_decisions are indexed by
  statement index. cfg_node_index_values[3] gives the values of all variables
  after each execution of statement with index 3.

  Attributes:
   trace_cfg_node_indexes: index in cfg.nodes of statements run (in order of
     execution)
   trace_line_indexes: line indexes of statements run (in order of execution)
   cfg_node_index_values: values after each execution of the control flow
     node
   cfg_node_index_branch_decisions: branch decisions after each execution of
     the control flow node
   line_index_values: values after each execution of the given line
   line_index_branch_decisions: branch decisions after each execution of the
     given line
   basic_block_branch_decisions: branch decisions taken after each basic block
   trace_values: values after each statement (in order of execution)
   trace_branch_decisions: branch decisions (in order of execution)
   trace_block_indexes: The index of the statement's block (in order of
     execution)
   trace_instruction_indexes: The index of the statement's instruction index in
     the current block. (in order of execution)
  """
  trace_cfg_node_indexes: List[int]
  trace_line_indexes: List[int]
  cfg_node_index_values: List[List[Values]]
  cfg_node_index_branch_decisions: List[List[int]]
  line_index_values: List[List[Values]]
  line_index_branch_decisions: List[List[int]]
  basic_block_branch_decisions: List[List[int]]
  trace_values: List[List[Values]]
  trace_branch_decisions: List[List[int]]
  trace_block_indexes: List[int]
  trace_instruction_indexes: List[int]


def make_trace(python_source, cfg):
  num_cfg_nodes = len(cfg.nodes)
  num_lines = len(python_source.strip().split("\n"))
  num_basic_blocks = len(cfg.blocks)
  return Trace(
      trace_cfg_node_indexes=[],
      trace_line_indexes=[],
      cfg_node_index_values=[[] for _ in range(num_cfg_nodes)],
      cfg_node_index_branch_decisions=[[] for _ in range(num_cfg_nodes)],
      line_index_values=[[] for _ in range(num_lines)],
      line_index_branch_decisions=[[] for _ in range(num_lines)],
      basic_block_branch_decisions=[[] for _ in range(num_basic_blocks)],
      trace_values=[],
      trace_branch_decisions=[],
      trace_block_indexes=[],
      trace_instruction_indexes=[],
  )


def make_trace_fn(python_source, cfg):
  """Creates a trace function `trace_fn` to be used by `eval_statements`."""
  # The trace will be available as trace_fn.trace.
  trace = make_trace(python_source, cfg)
  def trace_fn(control_flow_node, values,
               branch_decision=NO_BRANCH_DECISION):
    """Records the execution of a control_flow_node in the trace.

    The trace is available as trace_fn.trace.

    Args:
      control_flow_node: The control_flow.ControlFlowNode that was just run.
      values: The resultant values of all variables in the program.
      branch_decision: The branch decision, if any, taken after running the
        node's instruction.
    """
    values = copy.copy(values)
    basic_block = control_flow_node.block
    cfg_node_index = cfg.nodes.index(control_flow_node)
    instruction_index = basic_block.index_of(control_flow_node)
    line_index = control_flow_node.instruction.node.lineno - 1
    block_index = cfg.blocks.index(basic_block)
    trace.trace_cfg_node_indexes.append(cfg_node_index)
    trace.trace_line_indexes.append(line_index)
    trace.cfg_node_index_values[cfg_node_index].append(values)
    trace.cfg_node_index_branch_decisions[cfg_node_index].append(
        branch_decision)
    trace.line_index_values[line_index].append(values)
    trace.line_index_branch_decisions[line_index].append(branch_decision)
    trace.trace_block_indexes.append(block_index)
    trace.trace_instruction_indexes.append(instruction_index)
    if branch_decision != NO_BRANCH_DECISION:
      trace.basic_block_branch_decisions[block_index].append(branch_decision)
    trace.trace_values.append([values])
    trace.trace_branch_decisions.append([branch_decision])
  trace_fn.trace = trace
  return trace_fn
