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

"""Base class for a sequential mutator."""

import abc

from abstract_nas.evolution.mutator.base import AbstractMutator
from abstract_nas.model.concrete import Graph
from abstract_nas.model.subgraph import SubgraphSpec


class AbstractSequentialMutator(AbstractMutator):
  """Base class for sequential mutator.

  A sequential mutator only selects sequential subgraphs for mutation, which are
  subgraphs with exactly one input and one output, consisting of a sequence of
  ops, each of which consumes exactly one input that must be an output of the
  previous op.
  """

  @abc.abstractmethod
  def select_sequential_subgraph(self, graph):
    raise NotImplementedError

  def select_subgraph(self, graph):
    subgraph_spec = self.select_sequential_subgraph(graph)

    # ensure the subgraph is sequential
    input_names = subgraph_spec[0].op.input_names
    output_names = []
    for node in subgraph_spec:
      # each node can only have one input
      assert len(node.op.input_names) == 1
      # the input must be an output of the previous node
      for input_name in node.op.input_names:
        assert input_name in input_names
      input_names = [
          f"{node.op.name}:{idx}" for idx in range(node.op.num_outputs)
      ]
      output_names.extend(node.output_names)
    # the graph must have exactly one output
    assert len(output_names) == 1

    return subgraph_spec
