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

"""Symbolic instructions visualization with graphviz."""

import graphviz

from symbolic_functionals.syfes.symbolic import instructions


INSTRUCTION_LABEL = {
    'AdditionBy1Instruction': 'a+1',
    'Power2Instruction': 'a^2',
    'Power3Instruction': 'a^3',
    'Power4Instruction': 'a^4',
    'Power6Instruction': 'a^6',
    'SquareRootInstruction': 'a^1/2',
    'CubeRootInstruction': 'a^1/3',
    'Log1PInstruction': 'log(1+a)',
    'ExpInstruction': 'e^a',
    'AdditionInstruction': 'a+b',
    'SubtractionInstruction': 'a-b',
    'MultiplicationInstruction': 'a*b',
    'MultiplicationAdditionInstruction': '+=a*b',
    'DivisionInstruction': 'a/b',
    'UTransformInstruction': 'UTransform',
    'PBEXInstruction': 'PBEX',
    'PBECInstruction': 'PBEC',
    'RPBEXInstruction': 'RPBEX',
    'B88XInstruction': 'B88X',
}


def create_graph(feature_names,
                 shared_parameter_names,
                 bound_parameter_names,
                 variable_names,
                 instruction_list):
  """Creates graph for instruction_list.

  Args:
    feature_names: List of strings, the name of features.
    shared_parameter_names: List of strings, the name of shared parameters.
    bound_parameter_names: List of strings, the name of bound parameters.
    variable_names: List of strings, the name of variables.
    instruction_list: List of lists of strings. The first string is the
        instruction name, the second string is the output name and the rest are
        input names and bound parameters.

  Returns:
    graphviz.Digraph.
  """
  def _get_node(name, variable_reuse=True):
    """Gets node for the symbol.

    Args:
      name: String, the name of the symbol.
      variable_reuse: Boolean, whether to reuse the existing node in graph
          if it exists. Otherwise, create new node.

    Returns:
      String, the id of the node.
    """
    if name in variable_names:
      if name not in counter:
        counter[name] = 0
      if variable_reuse and counter[name] > 0:
        return f'{name}--{counter[name]}'
      else:
        counter[name] += 1
        output_id = f'{name}--{counter[name]}'
        graph.node(output_id, name)
        return output_id
    elif name in feature_names:
      graph.node(name, name, fillcolor='#0072b2', style='filled')
      return name
    elif name in shared_parameter_names:
      graph.node(name, name, fillcolor='#de8f05', style='filled')
      return name
    elif name in bound_parameter_names:
      graph.node(name, name, fillcolor='#cc79a7', style='filled')
      return name
    else:
      counter[name] = counter.get(name, -1) + 1
      output_id = f'{name}--{counter[name]}'
      graph.node(
          output_id, INSTRUCTION_LABEL[name],
          fillcolor='#009e73', style='filled', shape='square')
      return output_id

  feature_names = set(feature_names)
  shared_parameter_names = set(shared_parameter_names)
  bound_parameter_names = set(bound_parameter_names)
  variable_names = set(variable_names)
  counter = {}
  graph = graphviz.Digraph()
  for name in feature_names:
    _get_node(name)
  for name in shared_parameter_names:
    _get_node(name)
  for name in bound_parameter_names:
    _get_node(name)

  for instruction in instruction_list:
    instruction_id = _get_node(instruction[0])
    instruction_class = instructions.INSTRUCTION_CLASSES[instruction[0]]

    for input_index in range(instruction_class.get_num_inputs()):
      # first 2 are instruction id and output. The inputs start from index 2.
      input_id = _get_node(instruction[input_index + 2])
      graph.edge(input_id, instruction_id)

    if instruction[0] == 'MultiplicationAdditionInstruction':
      # for a += b * c operation, set a also as an input node and connect it
      # with instruction node with double arrow
      graph.edge(_get_node(instruction[1]), instruction_id, color='black:black')

    for bound_parameter_name in instruction_class.get_bound_parameters():
      bound_parameter_id = _get_node(bound_parameter_name)
      graph.edge(bound_parameter_id, instruction_id)

    output_id = _get_node(instruction[1], variable_reuse=False)
    graph.edge(instruction_id, output_id)
  return graph
