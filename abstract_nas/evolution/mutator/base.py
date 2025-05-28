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

"""Base class for mutator.

A mutator takes a graph as input, selects a subgraph, mutates the subgraph's
properties, then returns a specification of the subgraph and mutated properties.
"""

import abc
import random
from typing import Any, Dict, Optional, Sequence, Tuple

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.abstract.shape import ShapeProperty
from abstract_nas.model.concrete import Graph
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphSpec

Tensor = Any


class AbstractMutator(abc.ABC):
  """Base class for mutator."""

  def __init__(self, properties):
    self.properties = properties

  @abc.abstractmethod
  def select_subgraph(self, graph):
    """Selects a subgraph for mutation."""
    raise NotImplementedError

  def mutate(
      self,
      graph,
      contexts,
      abstract = True
  ):
    """Selects a subgraph and mutates its abstract properties.

    Note that this method does not mutate the graph, only the properties of a
    subgraph. Therefore the caller is responsible for synthesizing a subgraph
    satisfying the mutated properties.

    Args:
      graph: The input graph to mutate.
      contexts: A list of (constants, state, inputs) tuples, each specifying
        a different instantiation of the graph.
      abstract: Whether to infer the subgraph properties abstractly or
        concretely.

    Returns:
      For each instantiation, a subgraph model specifying the selected subgraph,
      and a sequence of abstract properties specifying the mutates properties.

    Raises:
      NotImplementedError: for concrete inference of properties.
    """

    subgraph_spec = self.select_subgraph(graph)
    new_graph = replace_subgraph(graph, subgraph_spec)
    if not abstract:
      # need to initialize the model state if not abstract
      raise NotImplementedError

    models_and_props = []
    to_mutate = random.randrange(len(contexts))
    for idx, (constants, _, inputs) in enumerate(contexts):
      subgraph_model = SubgraphModel(new_graph, constants, state=None,
                                     inputs=inputs, subgraph=subgraph_spec)
      # Only mutate the properties of one randomly selected instance. The other
      # instances simply need to satisfy the shape property (which is never
      # mutated).
      # The alternative would be to make sure that all the properties are
      # mutated "in the same way" for every instance of the graph, but that
      # requires overly complex logic matching input and output names.
      if idx == to_mutate:
        new_properties = [
            prop.infer(subgraph_model, abstract=abstract).mutate()
            for prop in self.properties
        ]
      else:
        new_properties = []
        for prop in self.properties:
          if type(prop) is ShapeProperty:  # pylint: disable=unidiomatic-typecheck
            new_properties.append(ShapeProperty().infer(
                subgraph_model, abstract=abstract))
      models_and_props.append((subgraph_model, new_properties))

    # match mutated shapes
    for prop in models_and_props[to_mutate][1]:
      if type(prop) is not ShapeProperty: continue  # pylint: disable=unidiomatic-typecheck
      output_shapes = prop.output_shapes
      for _, other_props in models_and_props:
        for other_prop in other_props:
          if type(other_prop) is not ShapeProperty: continue  # pylint: disable=unidiomatic-typecheck
          for k in list(other_prop.output_shapes.keys()):
            if k not in output_shapes:
              del other_prop.output_shapes[k]
    return models_and_props
