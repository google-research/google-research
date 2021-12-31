# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Python interpreter that operates on control flow graphs."""

import math
from absl import logging  # pylint: disable=unused-import
import astunparse
import gast as ast
import tree

from ipagnn.datasets.control_flow_programs import python_interpreter_trace


class ExecExecutor(object):
  """A Python executor that uses exec.

  Potentially unsafe; use only with trusted code.
  """

  def __init__(self):
    self.locals = {}

  def execute(self, code):
    exec(code,  # pylint:disable=exec-used
         {'__builtins__': {'True': True, 'False': False, 'range': range,
                           'sqrt': math.sqrt, 'AssertionError': AssertionError,
                           'len': len,
                           }},
         self.locals)

  def get_values(self, mod=None):
    """Gets the values (mod `mod`, if applicable) of the executor."""
    values = self.locals.copy()
    if mod is not None:
      values = tree.map_structure(lambda x: x % mod, values)
    return values




def evaluate_cfg(executor, cfg, mod=None, initial_values=None, trace_fn=None,
                 timeout=None):
  """Evaluates a Python program given its control flow graph.

  Args:
    executor: The executor with which to perform the execution.
    cfg: The control flow graph of the program to execute.
    mod: The values are computed mod this.
    initial_values: Optional dictionary mapping variable names to values.
    trace_fn: A function for collecting the execution trace.
    timeout: Optional maximum number of basic blocks to evaluate before
        raising a timeout RuntimeError.
  Returns:
    A values dictionary mapping variable names to their final values.
  Raises:
    RuntimeError: If timeout is given and the program runs for more than
      `timeout` blocks, a RuntimeError is raised.
  """
  executor.locals = {}
  block = cfg.start_block
  values = initial_values or {}  # Default to no initial values.
  blocks_evaluated = 0
  while block:
    if timeout and blocks_evaluated > timeout:
      raise RuntimeError('Evaluation of CFG has timed out.')
    block, values = evaluate_until_next_basic_block(
        executor, block, mod=mod, values=values, trace_fn=trace_fn)
    blocks_evaluated += 1
  return values


def evaluate_until_next_basic_block(executor, basic_block, mod, values,
                                    trace_fn=None):
  """Takes a single step of control flow graph evaluation.

  Evaluates a single basic block starting from the provided values. Returns
  the correct next basic block to step to and the new values of all the
  variables.

  Args:
    executor: The executor with which to take a step of execution.
    basic_block: (control_flow.BasicBlock) A single basic block from the control
      flow graph.
    mod: The values are computed mod this.
    values: A dict mapping variable names to literal Python values.
    trace_fn: A function for collecting the execution trace.
  Returns:
    The next basic block to execute and the new mapping from variable names to
    values.
  """
  values = evaluate_basic_block(executor, basic_block, mod=mod, values=values,
                                trace_fn=trace_fn)
  if not basic_block.exits_from_end:
    # This is the end of the program.
    return None, values
  elif len(basic_block.exits_from_end) == 1:
    # TODO(dbieber): Modify control_flow.BasicBlock API to have functions
    # `has_only_one_exit` and `get_only_exit`.
    basic_block = next(iter(basic_block.exits_from_end))
  else:
    assert len(basic_block.exits_from_end) == 2
    assert len(basic_block.branches) == 2, basic_block.branches
    assert 'vBranch' in values
    branch_decision = bool(values['vBranch'])
    basic_block = basic_block.branches[branch_decision]
    # TODO(dbieber): Trace the branch decision too.
  return basic_block, values


def evaluate_until_branch_decision(executor, basic_block, mod, values,
                                   trace_fn=None):
  """Evaluates a Python program until reaching a branch decision.

  Evaluates one basic block at a time until a branch decision is reached.
  Returns the resulting values of the variables, the instructions executed,
  and the branch decision. The branch decision is represented as a dict mapping
  True/False to the next basic block after the branch decision.

  Args:
    executor: The executor with which to perform the execution.
    basic_block: A single basic block from the control flow graph.
    mod: The values are computed mod this.
    values: A dict mapping variable names to literal Python values.
    trace_fn: A function for collecting the execution trace.
  Returns:
    A triple (values, instructions, branches). `values` is the resulting values
    of the variables. `instructions` is a list of the instructions executed,
    and `branches` is the branch decision reached, represented as a dict mapping
    True/False to the next basic block after the branch decision.
  """
  instructions = []

  done = False
  branches = None
  while not done:
    # Collect the instructions from the current block.
    nodes = basic_block.control_flow_nodes
    for node in nodes:
      instructions.append(node.instruction)

    # Evaluate the current block.
    values = evaluate_basic_block(executor, basic_block, mod=mod,
                                  values=values, trace_fn=trace_fn)

    # Determine next block to evaluate or whether to stop.
    # TODO(dbieber): Refactor to reduce redundancy with
    # evaluate_until_next_basic_block.
    if not basic_block.exits_from_end:
      # The program has terminated.
      done = True
      branches = None
    elif len(basic_block.exits_from_end) == 1:
      # There is no branch decision at this point. Keep evaluating.
      basic_block = next(iter(basic_block.exits_from_end))
    else:
      # Evaluation has reached a branch decision.
      assert len(basic_block.exits_from_end) == 2
      assert len(basic_block.branches) == 2
      assert 'vBranch' in values
      done = True
      branches = basic_block.branches

  return values, instructions, branches


def evaluate_basic_block(executor, basic_block, mod, values, trace_fn=None):
  """Evaluates a single basic block of Python with an executor.

  Args:
    executor: The executor with which to perform the execution.
    basic_block: A control_flow.BasicBlock of Python statements.
    mod: The values are computed mod this.
    values: A dictionary mapping variable names to their Python literal values.
    trace_fn: (optional) A function to record the statements executed and the
      sequence of values the variables took on during execution.
  Returns:
    A dictionary mapping variable names to their final values at the end of
    evaluating the basic block.
  """

  for var, value in values.items():
    python_source = f'{var} = {value}'
    executor.execute(python_source)

  nodes = basic_block.control_flow_nodes
  for index, node in enumerate(nodes):
    instruction = node.instruction
    ast_node = instruction.node
    python_source = astunparse.unparse(ast_node, version_info=(3, 5))

    make_branch_decision = (index == len(nodes) - 1 and basic_block.branches)
    if make_branch_decision:
      python_source = 'vBranch = ' + python_source

    executor.execute(python_source)

    if trace_fn is not None:
      values = executor.get_values(mod=mod)

      # Note: This records the correct branch decision based on the values
      # provided, i.e. the value taken on by vBranch. An agent may choose to
      # take a branch inconsistent with vBranch.
      if make_branch_decision:
        branch_decision = (python_interpreter_trace.TRUE_BRANCH
                           if values['vBranch']
                           else python_interpreter_trace.FALSE_BRANCH)
      else:
        branch_decision = python_interpreter_trace.NO_BRANCH_DECISION

      trace_fn(control_flow_node=node, values=values,
               branch_decision=branch_decision)

  # Extract the values of the v0, v1... variables.
  values = executor.get_values(mod=mod)
  return values
