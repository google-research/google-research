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

import abc
from strategic_exploration.hrl.graph import DirectedEdge


class GraphUpdate(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def update(self, graph):
    """Updates the AbstractGraph.

        Args: graph (AbstractGraph)
    """
    raise NotImplementedError()


class Traverse(GraphUpdate):
  EDGE_EXPANSION_COEFF = None

  def __init__(self, edge, success, teleport=None):
    """Safe across multi-process.

        Args: edge (DirectedEdge)
            success (bool): if edge was traversed
            teleport (Teleport | None): teleport action to the next_state that
            was reached, only used if success is True
    """
    # May be from a different process, so you need to copy the underlying
    # state, not the DirectedEdge!
    self._edge = [edge.start.abstract_state, edge.end.abstract_state]
    self._edge_degree = edge.degree
    self._success = success
    self._teleport = teleport

  @classmethod
  def configure(cls, edge_expansion_coeff):
    """Configures classmethods.

        Args:
            edge_expansion_coeff (int): configures
              Traverse.edge_expansion_attempts
    """
    cls.EDGE_EXPANSION_COEFF = edge_expansion_coeff

  @classmethod
  def edge_expansion_attempts(cls, degree):
    """Returns number of wasted attempts that triggers edge expansion."""
    return cls.EDGE_EXPANSION_COEFF * (degree**2)

  def update(self, graph):
    """Updates the graph by marking the edge as traverse(success).

    Does
        edge expansion and adds teleport as necessary

        Args: graph (AbstractGraph)
    """
    edge = graph.get_edge(self._edge[0], self._edge[1], self._edge_degree)
    edge.traverse(self._success)
    if not self._success:
      self._edge_expansion(edge, graph)

    # Gets hit first time an edge becomes reliable
    if self._success and edge.reliable() and edge.end.teleport is None:
      edge.end.set_teleport(self._teleport)

  def _edge_expansion(self, edge, graph):
    wasted_attempts = edge.train_count - edge.total_successes
    if wasted_attempts != Traverse.edge_expansion_attempts(edge.degree) or \
            not edge.training():
      return

    # Make sure that edge expansion doesn't get triggered multiple times
    edge.traverse(False)

    # Expand forwards outside of the feasible set
    for neighbor_edge in edge.end.neighbors:
      neighbor = neighbor_edge.end
      degree = edge.degree + neighbor_edge.degree
      reward = edge.reward + neighbor_edge.reward
      life_lost = edge.life_lost or neighbor_edge.life_lost
      if not neighbor.contains_parent(edge.start):
        graph.get_edge(edge.start.abstract_state, neighbor.abstract_state,
                       degree, reward, life_lost)

    # Expand backwards into feasible set
    for parent_edge in edge.start.parents:
      if not parent_edge.reliable():
        continue

      parent = parent_edge.start
      degree = edge.degree + parent_edge.degree
      reward = edge.reward + parent_edge.reward
      life_lost = edge.life_lost or parent_edge.life_lost
      if not parent.contains_neighbor(edge.end):
        graph.get_edge(parent.abstract_state, edge.end.abstract_state, degree,
                       reward, life_lost)

    if edge.degree >= 2:
      # Add the edge again with higher degree (allow more steps
      # and use pixel-worker)
      new_edge = graph.get_edge(edge.start.abstract_state,
                                edge.end.abstract_state, edge.degree + 2,
                                edge.reward)
      #logging.info("Adding a new edge: {}".format(new_edge))


class Visit(GraphUpdate):

  def __init__(self, node):
    """Safe across multi-process.

        Args:
            node (AbstractNode): doesn't necessarily modify *this* node, since
              it may be a multi-process copy
    """
    # Copy abstract state, not the node for multi-process
    self._abstract_state = node.abstract_state

  def update(self, graph):
    """Updates the graph by marking the node.visit().

        Args: graph (AbstractGraph)
    """
    # TODO: In parallel mode, remove this.
    return
    graph.get_node(self._abstract_state).visit()
