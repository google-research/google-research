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

"""Abstract base class for a synthesizer.

At a high level, a synthesizer is given a set of abstract properties, and
returns a (sub)graph satisfying the properties. We also provide the enclosing
context into which the subgraph will be embedded, since some of the abstract
properties require this context to evaluate.
"""

import abc
import sys
import traceback
from typing import Optional, Sequence, Tuple

from absl import logging
import flax
from jax import random

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.model import Model
from abstract_nas.model import subgraph
from abstract_nas.model.concrete import Graph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphSpec


class AbstractSynthesizer(abc.ABC):
  """Base class for synthesizer."""

  def __init__(self,
               subgraphs_and_props,
               generation,
               abstract = True):
    """Initializes a synthesizer.

    Args:
      subgraphs_and_props: A list of tuples providing subgraphs providing the
        enclosing contexts into which the synthesized subgraph will be embedded,
        and the properties for each instantiation to satisfy. Note that each
        subgraph will have the same graph, but differing constants, state, and
        inputs.
      generation: The generation of the new subgraph, the value of which must be
        incremented at least once every time a graph is selected to have a
        subgraph in it replaced. This is primarily to ensure that the
        synthesizer is easily able to generate unique names for any newly
        inserted nodes.
      abstract: Whether the properties of the synthesize subgraph need to to
        satisfy the properties abstractly or concretely (i.e., using actual
        values).
    """
    self.subgraphs_and_props = subgraphs_and_props
    self.generation = generation
    self.abstract = abstract

  @abc.abstractmethod
  def synthesize(self):
    raise NotImplementedError

  def verify(self, subg_models):
    if len(subg_models) != len(self.subgraphs_and_props):
      raise ValueError("Proposed models has the wrong number of models.")
    for subg_model, (_, properties) in zip(subg_models,
                                           self.subgraphs_and_props):
      for prop in properties:
        if not prop.verify(subg_model, self.abstract):
          return False
    return True

  def get_subg_distance(self, subg_models):
    """Gets the total distance for all properties and models."""
    if len(subg_models) != len(self.subgraphs_and_props):
      raise ValueError("Proposed models has the wrong number of models.")

    dist = {}
    for subg_model, (_, properties) in zip(subg_models,
                                           self.subgraphs_and_props):
      cur_dist = {}
      for prop in properties:
        new_prop = prop.infer(subg_model, abstract=self.abstract)
        cur_dist[type(prop)] = prop.distance_from(new_prop)
      for key in cur_dist:
        if key not in dist:
          dist[key] = cur_dist[key]
        else:
          assert dist[key] == cur_dist[key]
    total_dist = 0
    for _, d in dist.items():
      total_dist += d
    return total_dist

  def make_subgraph_models(
      self,
      subgraph_spec,
      graphs = None):
    """Inserts the new subgraph_spec into the subgraph_models.

    The graphs argument can be used to pass in intermediate results of synthesis
    rather than completing synthesis all at once, e.g., we can call
    make_subgraph_models twice and pass the output of the first call as an
    argument to the second call. This allows us to break the synthesis down into
    multiple steps.

    Args:
      subgraph_spec: The new ops to insert.
      graphs: The graphs into which to insert the new ops.

    Returns:
      A sequence of subgraphs with subgraph_spec inserted.
    """
    new_subgraph_models = []
    if not graphs:
      graphs = [None] * len(self.subgraphs_and_props)
    for graph, (subgraph_model, _) in zip(graphs, self.subgraphs_and_props):
      graph = graph if graph else subgraph_model.graph
      constants = subgraph_model.constants
      state = subgraph_model.state
      inputs = subgraph_model.inputs

      new_graph = subgraph.replace_subgraph(graph, subgraph_spec)
      if not self.abstract:
        # concrete synthesis initializes the state while inheriting the parent
        # params
        new_model = Model(new_graph, constants)
        try:
          new_state = new_model.init(random.PRNGKey(0), inputs)
        except Exception as e:  # pylint: disable=broad-except
          # catch everything else for now... this is the safest way to filter
          # out malformed subgraphs which will not initialize
          exc_type, exc_value, exc_traceback = sys.exc_info()
          logging.info(
              "%s", "".join(
                  traceback.format_exception(exc_type, exc_value,
                                             exc_traceback)))
          raise ValueError("Could not initialized malformed subgraph "
                           f"({type(e).__name__}: {e}).") from e

        new_state = flax.core.unfreeze(new_state)
        inherited, frozen = subgraph.inherit_params(new_state["params"],
                                                    state["params"])
        new_state = {**inherited, **frozen}
      else:
        new_state = None
      new_subgraph_model = SubgraphModel(new_graph, constants, new_state,
                                         inputs, subgraph_spec)
      new_subgraph_models.append(new_subgraph_model)
    return new_subgraph_models
