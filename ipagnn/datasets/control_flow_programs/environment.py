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

"""The control flow programs reinforcement learning environment."""

from absl import logging  # pylint: disable=unused-import

import astunparse

from ipagnn.datasets.control_flow_programs import control_flow_programs_features
from ipagnn.datasets.control_flow_programs import python_interpreter
from ipagnn.datasets.control_flow_programs import python_interpreter_trace
from ipagnn.datasets.control_flow_programs import python_programs


def init(python_object, info, config):
  """python_object -> state, reward."""
  base = info.program_generator_config.base
  tokens_per_statement = info.program_encoder.tokens_per_statement
  target_output_length = info.program_generator_config.num_digits
  mod = info.program_generator_config.mod
  output_mod = info.program_generator_config.output_mod

  executor = python_interpreter.ExecExecutor()
  if isinstance(python_object, tuple):
    python_source, partial_python_source = python_object
    cfg = python_programs.to_cfg(partial_python_source)
  else:
    python_source = python_object
    cfg = python_programs.to_cfg(python_source)

  # Run until branch decision, collecting simple statements.
  # TODO(dbieber): This should occur in exactly one location.
  # (also in control_flow_programs_features.py)
  initial_values = {'v0': 1}
  values, instructions, branches = (
      python_interpreter.evaluate_until_branch_decision(
          executor, cfg.start_block, mod=base ** target_output_length,
          values=initial_values))

  state = dict(
      initial_values=initial_values,
      values=values,
      instructions=instructions,
      branches=branches,
      base=base,
      tokens_per_statement=tokens_per_statement,
      target_output_length=target_output_length,
      mod=mod,
      output_mod=output_mod,
      executor=executor,
      config=config,
      step=0,
  )
  reward = 0.0
  return state, reward


def step(state, action, log_prob_branch_decision_fn):
  """state, action -> state', reward."""
  # Takes the given action, then executes until an unknown branch decision.
  config = state['config']
  executor = state['executor']
  block = state['branches'][action]
  base = state['base']
  target_output_length = state['target_output_length']
  mod = state['mod']

  initial_values = state['values']
  values, instructions, branches = (
      python_interpreter.evaluate_until_branch_decision(
          executor=executor,
          basic_block=block,
          mod=mod,
          values=initial_values))

  state = dict(
      initial_values=initial_values,
      values=values,
      instructions=instructions,
      branches=branches,
      base=base,
      tokens_per_statement=state['tokens_per_statement'],
      target_output_length=target_output_length,
      mod=state['mod'],
      output_mod=state['output_mod'],
      executor=executor,
      config=config,
      step=state['step'] + 1,
  )

  if config.train.use_intermediate_outputs:
    # Reward for correct branch predictions:
    correct_action = bool(initial_values['vBranch'])
    if config.reinforce.discrete_reward:
      # action dependent discrete reward.
      reward = 1.0 if correct_action == action else 0.0
    else:
      # action_logit dependent continuous reward.
      correct_action_index = (
          python_interpreter_trace.TRUE_BRANCH if correct_action
          else python_interpreter_trace.FALSE_BRANCH)
      reward = log_prob_branch_decision_fn(correct_action_index)

    # TODO(dbieber): Reward for correct intermediate value predictions.
  else:
    reward = 0.0
  return state, reward


def is_terminal(state):
  """state -> is_terminal."""
  return (state['branches'] is None
          or state['step'] >= state['config'].reinforce.max_steps)


def state_as_example(state):
  """state -> example dict."""
  base = state['base']
  tokens_per_statement = state['tokens_per_statement']
  target_output_length = state['target_output_length']
  mod = state['mod']
  output_mod = state['output_mod']
  python_source_lines = []
  for instruction in state['instructions']:
    ast_node = instruction.node
    python_source_line = astunparse.unparse(ast_node, version_info=(3, 5))
    python_source_line = python_source_line.strip()
    python_source_lines.append(python_source_line)
  python_source = '\n'.join(python_source_lines)
  return control_flow_programs_features.generate_example_from_python_object(
      executor=state['executor'],
      base=base,
      python_object=python_source,
      tokens_per_statement=tokens_per_statement,
      target_output_length=target_output_length,
      mod=mod,
      output_mod=output_mod)


def make_env():
  """Creates an environment for executing programs by making branch decisions.

  Returns:
    The environment, consisting of init, step, is_terminal, and state_as_example
    functions.
  """
  return {
      'init': init,
      'step': step,
      'is_terminal': is_terminal,
      'state_as_example': state_as_example,
  }
