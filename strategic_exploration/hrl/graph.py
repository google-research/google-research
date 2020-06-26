# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
import copy
import numpy as np
import logging
from ast import literal_eval
from collections import deque
from strategic_exploration.hrl import abstract_state as AS
from strategic_exploration.hrl import data
from strategic_exploration.hrl.utils import mean_with_default


class DirectedEdge(object):
  """A directed edge between two abstract nodes representing that it is

    possible to traverse from the start node to the end node.
  """
  # Enum for four different types of state, see state property
  TRAINING = "training"
  EVALUATING = "evaluating"
  RELIABLE = "reliable"

  def __init__(self,
               start,
               end,
               eval_window_size,
               train_window_size,
               traverse_threshold,
               callbacks,
               degree=1,
               reward=0.,
               life_lost=False):
    """
        Args:
            start (AbstractNode): can traverse from start node to end node end
              (AbstractNode)
            eval_window_size (int): number of past traverse attempts to keep
              track of
            traverse_threshold (float): above this threshold, state is reliable
            callbacks (dict): whenever this edge transitions from state A to
              state B, callbacks[(A, B)](self) is called (A != B).
            degree (int): a degree k + 1 edge is gotten by bypassing a degree k
              edge
            reward (float): reward received from traversing this edge
            life_lost (bool): True if traversing this edge loses a life
    """
    self._start = start
    self._end = end
    self._degree = degree
    self._train_count = 0
    self._train_window = deque(maxlen=train_window_size)
    self._train_window_size = train_window_size
    self._eval_window = deque(maxlen=eval_window_size)
    self._eval_window_size = eval_window_size
    self._past_window = deque(maxlen=train_window_size)
    self._traverse_threshold = traverse_threshold
    self._total_successes = 0
    self._callbacks = callbacks
    self._reward = reward
    self._state = DirectedEdge.EVALUATING
    self._dead = False
    self._life_lost = life_lost

  @property
  def start(self):
    """Returns AbstractNode, see constructor."""
    return self._start

  @property
  def end(self):
    """Returns AbstractNode, see constructor."""
    return self._end

  @property
  def state(self):
    """An edge can take on three different states:

            DirectedEdge.TRAINING: edge has not been trained > threshold times
                yet
            DirectedEdge.UNRELIABLE: edge has been evaluated > threshold times
                and cannot be reliably traversed
            DirectedEdge.RELIABLE: edge has been evaluated > threshold times
                and can be reliably traversed

        Returns:
            string: one of these three states
        """
    return self._state

  def training(self):
    """Returns True if in the TRAINING state."""
    return self.state == DirectedEdge.TRAINING

  def evaluating(self):
    """Returns True if in the EVALUATING state."""
    return self.state == DirectedEdge.EVALUATING

  def reliable(self):
    """Returns True if in the RELIABLE state."""
    return self.state == DirectedEdge.RELIABLE

  @property
  def degree(self):
    """Returns the degree (int). See constructor argument docs."""
    return self._degree

  @property
  def success_rate(self):
    """Returns the success rate over the past N attempts"""
    return mean_with_default(self._past_window, 0.)

  @property
  def train_count(self):
    """Returns the number of times this edge has been selected to

        traverse.

        Returns:
            int
        """
    return self._train_count

  @property
  def dead(self):
    return self._dead

  def kill(self):
    self._dead = True

  def start_evaluation(self):
    """Forces the state to be DirectedEdge.EVALUATING"""
    self._train_window.clear()
    self._eval_window.clear()
    self._state = DirectedEdge.EVALUATING

  def traverse(self, success):
    """Marks an attempt to traverse this edge.

    The traverse was
        successful if success is True.

        Args:
            success (bool): if the test attempt was successful
    """
    # Possible state transitions:
    # Enough successes: TRAINING --> EVALUATING
    # Enough successes: EVALUATING --> RELIABLE
    # Enough failures:  EVALUATING --> TRAINING
    if self.reliable() and not success:
      logging.error("Failed reliable edge: {}".format(self))

    if success:
      self._total_successes += 1
    self._past_window.append(success)

    if self.training():
      self._train_window.append(success)
      self._train_count += 1
      successes = sum(self._train_window)
      evaluation_threshold = \
          self._train_window_size * self._traverse_threshold
      if successes >= evaluation_threshold:
        self.start_evaluation()
        self._callback(DirectedEdge.TRAINING, DirectedEdge.EVALUATING)
    elif self.evaluating():
      self._eval_window.append(success)

      successes = sum(self._eval_window)
      reliable_threshold = \
          self._eval_window_size * self._traverse_threshold
      max_successes = successes + self._eval_window_size - \
          len(self._eval_window)
      if successes >= reliable_threshold:
        self._state = DirectedEdge.RELIABLE
        self._callback(DirectedEdge.EVALUATING, DirectedEdge.RELIABLE)
      elif max_successes < reliable_threshold:
        self._state = DirectedEdge.TRAINING
        self._eval_window.clear()
        self._callback(DirectedEdge.EVALUATING, DirectedEdge.TRAINING)
    elif self.reliable():
      # For some reason, not working
      successes = sum(self._past_window)
      if successes == 0:
        self._state = DirectedEdge.TRAINING
        self._eval_window.clear()
        self._callback(DirectedEdge.RELIABLE, DirectedEdge.TRAINING)

  @property
  def state_difference(self):
    return self.end.abstract_state.numpy - \
            self.start.abstract_state.numpy

  @property
  def reward(self):
    """The reward associated with traversing this edge (float)"""
    return self._reward

  def update_reward(self, reward, force=False):
    """Sets edge's reward if reward is positive.

    Ignores the reward if it
        is negative, unless force=True.
        """
    if force or reward >= 0:
      self._reward = reward

  @property
  def total_successes(self):
    return self._total_successes

  @property
  def life_lost(self):
    return self._life_lost

  def set_life_lost(self, life_lost):
    self._life_lost = life_lost

  def partial_state_dict(self):
    """Returns part of a state_dict.

    Can be used with missing information
        to recreate the DirectedEdge in from_partial_state_dict.

        Returns:
            state_dict (dict)
        """
    return {
        "degree": self._degree,
        "train_count": self._train_count,
        "train_window": self._train_window,
        "train_window_size": self._train_window_size,
        "eval_window": self._eval_window,
        "eval_window_size": self._eval_window_size,
        "past_window": self._past_window,
        "traverse_threshold": self._traverse_threshold,
        "total_successes": self._total_successes,
        "reward": self._reward,
        "state": self._state,
        "dead": self._dead,
        "life_lost": self._life_lost,
    }

  @classmethod
  def from_partial_state_dict(cls, state_dict, start, end, callbacks):
    """Creates a DirectedEdge from an partial state dict and the missing

        information.

        Args:
            state_dict (dict): the partial state dict from partial_state_dict()
            start (AbstractNode): see constructor
            end (AbstractNode): see constructor
            callbacks (AbstractNode): see constructor

        Returns:
            DirectedEdge
        """
    edge = cls(start, end, state_dict["eval_window_size"],
               state_dict["train_window_size"],
               state_dict["traverse_threshold"], callbacks,
               state_dict["degree"], state_dict["reward"],
               state_dict["life_lost"])
    edge._train_count = state_dict["train_count"]
    edge._train_window = state_dict["train_window"]
    edge._eval_window = state_dict["eval_window"]
    edge._past_window = state_dict["past_window"]
    edge._total_successes = state_dict["total_successes"]
    edge._reward = state_dict["reward"]
    edge._state = state_dict["state"]
    edge._dead = state_dict["dead"]
    edge._life_lost = state_dict["life_lost"]
    return edge

  def summary(self):
    """Returns a summary of this edge such that two edges edge1 and edge2

        represent the same information if edge1.summary == edge2.summary

        Returns:
            tuple
        """
    return (self.start.uid, self.end.uid, self.degree)

  def _callback(self, old_state, new_state):
    callback = self._callbacks.get((old_state, new_state))
    if callback is not None:
      callback(self)

  def __str__(self):
    return ("{} --> {} (deg={}, evaluate={}/{} [{:.2f}],"
            " train={}/{} [{:.2f}], past={}/{} [{:.2f}], status={},"
            " train count={}, reward={}, total successes={},"
            " dead={}, life lost={}").format(
                self.start.uid, self.end.uid, self.degree,
                sum(self._eval_window), len(self._eval_window),
                mean_with_default(self._eval_window,
                                  0.), sum(self._train_window),
                len(self._train_window),
                mean_with_default(self._train_window, 0.),
                sum(self._past_window), len(self._past_window),
                mean_with_default(self._past_window,
                                  0.), self.state, self.train_count,
                self.reward, self.total_successes, self.dead, self.life_lost)

  __repr__ = __str__


class AbstractNode(object):
  """A node in the AbstractGraph representing an AbstractState"""

  def __init__(self, abstract_state, min_visit_count, uid, start=False):
    """
        Args: abstract_state (AbstractState)
            min_visit_count (int): number of times a node must be visited before
            all neighbors are assumed to be discovered
            uid (int): a unique identifier
            start (bool): True if this is the starting node (the state that
            reseting starts on)
    """
    self._abstract_state = abstract_state
    self._visit_count = 0  # Number of times this node has been explored at
    # {(Neighbor, degree): (DirectedEdge (this --> neighbor)}
    self._neighbors = {}
    # {(Parent, degree): DirectedEdge (parent --> this)}
    self._parents = {}
    self._min_visit_count = min_visit_count
    self._uid = uid
    self._path_to_start = None
    self._start = start
    if start:
      self._path_to_start = []
    self._teleport = None
    self._path_reward = None
    self._distance_from_start = None

  def add_neighbor(self, edge_to_neighbor):
    """Adds an edge to a neighbor.

    Edges to itself are ignored as are
        duplicated neighbors.

        Args:
            edge_to_neighbor (DirectedEdge): edge starting at this node going to
              a neighbor
    """
    key = (edge_to_neighbor.end, edge_to_neighbor.degree)
    assert edge_to_neighbor.start == self
    if edge_to_neighbor.end != self and \
            key not in self._neighbors:
      self._neighbors[key] = edge_to_neighbor

  def add_parent(self, parent_to_self):
    """Adds an edge that ends in self as an edge starting at a parent.

        Edges to itself are ignored, as are duplicated parents.

        Args:
            parent_to_self (DirectedEdge): starts at a parent, ends at this node
    """
    key = (parent_to_self.start, parent_to_self.degree)
    assert parent_to_self.end == self
    if parent_to_self.start != self and \
            key not in self._parents:
      self._parents[key] = parent_to_self

  def distance_from_start(self):
    if self._distance_from_start is None:
      self._distance_from_start = len(self.path_to_start())
    return self._distance_from_start

  def path_reward(self):
    """Returns reward for path to get here (float), if there is a feasible

        path. Raises ValueError if no such path exists.
        """
    if self._path_reward is None:
      path_to_start = self.path_to_start()
      self._path_reward = sum(edge.reward for edge in path_to_start)
    return self._path_reward

  def path_to_start(self):
    if self._path_to_start is None:
      for parent_edge in self.parents:
        parent = parent_edge.start
        if parent_edge.reliable():
          self._path_to_start = \
                  parent.path_to_start() + [parent_edge]
          break
      else:
        raise ValueError("Called on non-feasible node {}".format(self))
    # Needs to return a copy, otherwise gets destructively modified
    return copy.copy(self._path_to_start)

  def get_neighbor_edge(self, neighbor, degree):
    """Returns an edge (self, neighbor) if this is a neighbor, otherwise

        returns None.

        Args: neighbor (AbstractNode) degree (int)

        Returns:
            DirectedEdge | None
        """
    return self._neighbors.get((neighbor, degree))

  def set_teleport(self, teleport):
    if teleport is None:
      logging.warning(
          ("Teleport set to None. Indicates that a [likely safe] timing"
           " issue happened."))
    logging.info("On edge: {}, setting teleport to: {}".format(self, teleport))
    if self._teleport is not None:
      logging.warning("Teleport already set to: {}. Ignoring.".format(self))
    self._teleport = teleport

  @property
  def teleport(self):
    # (Teleport action)
    return self._teleport

  @property
  def neighbors(self):
    """Returns list[DirectedEdge] of all edges to neighbors."""
    # Sort here for determinism
    # return sorted(self._neighbors.values(), key=lambda edge: str(edge))
    return list(self._neighbors.values())

  def contains_neighbor(self, neighbor):
    neighbors = set([n for (n, degree) in self._neighbors])
    return neighbor in neighbors

  @property
  def parents(self):
    """Returns list[DirectedEdge] of all edges to parents."""
    # Sort here for determinism
    # return sorted(self._parents.values(), key=lambda edge: str(edge))
    return list(self._parents.values())

  def contains_parent(self, parent):
    parents = set([p for (p, degree) in self._parents])
    return parent in parents

  @property
  def uid(self):
    return self._uid

  @property
  def abstract_state(self):
    return self._abstract_state

  def active(self):
    """Returns True if this node must be visited (explored)  more times."""
    return self._visit_count < self._min_visit_count

  @property
  def visit_count(self):
    """Number of times this node has been visited: int"""
    return self._visit_count

  @property
  def min_visit_count(self):
    return self._min_visit_count

  def visit(self):
    """Marks a single visit of this node."""
    self._visit_count += 1

  def clear_path_reward_cache(self):
    """Marks path_reward for recalculation."""
    self._path_reward = None

  def partial_state_dict(self):
    """Returns a representation of the AbstractNode with everything minus

        neighbor information. Can be used to reconstruct an AbstractNode with
        from_partial_state_dict.

        Returns:
            state_dict (dict)
        """
    # distance_from_start and derivatives can be recomputed.
    return {
        "abstract_state": self._abstract_state,
        "min_visit_count": self._min_visit_count,
        "visit_count": self._visit_count,
        "uid": self._uid,
        "start": self._start,
        "teleport": self._teleport,
    }

  @classmethod
  def from_partial_state_dict(cls, state_dict):
    """Reloads everything except for neighbor information.

        Args:
            state_dict (dict): partial state dict from partial_state_dict.

        Returns:
            AbstractNode
        """
    node = cls(state_dict["abstract_state"], state_dict["min_visit_count"],
               state_dict["uid"], state_dict["start"])
    node._visit_count = state_dict["visit_count"]
    node.set_teleport(state_dict["teleport"])
    return node

  def __str__(self):
    if self._path_to_start is not None:
      dist_from_start = self.distance_from_start()
      path_reward = self.path_reward()
    else:
      dist_from_start = None
      path_reward = None

    return ("[{}] {} (visit count={}, neighbors={}, parents={},"
            " dist from start: {}, path reward={})".format(
                self.uid, self._abstract_state, self.visit_count,
                len(self._neighbors), len(self._parents), dist_from_start,
                path_reward))

  __repr__ = __str__


class AbstractGraph(object):
  """Collection of AbstractNodes arranged into a graph."""

  @classmethod
  def from_config(cls, config, start_state, edge_callbacks, new_edge_callback,
                  domain):
    """Construct an AbstractGraph from a Config object.

    See constructor for
        documentation of what should be in the Config.
    """
    if config.type == "abstract":
      graph_factory = cls
    elif config.type == "oracle":
      graph_factory = OracleGraph
    else:
      raise ValueError("{} not a supported AbstractGraph type".format(
          config.type))

    graph = graph_factory(start_state, config.edge_window_size,
                          config.traverse_threshold, config.min_visit_count,
                          config.max_edge_degree, edge_callbacks,
                          new_edge_callback)

    if config.whitelist:
      return Whitelist(graph, domain)
    return graph

  def __init__(self, start_state, edge_window_size, traverse_threshold,
               min_visit_count, max_edge_degree, edge_callbacks,
               new_edge_callback):
    """
        Args:
            start_state (AbstractState): the starting state of the environmnet
            edge_window_size (int): number of episodes to keep track of for
              evaluation
            traverse_threshold (float): threshold for traverse_prob of edges to
              be considered feasible
            min_visit_count (int): see AbstractNode constructor
            max_edge_degree (int): maximum allowable degree for edges in graph
            edge_callbacks (dict): see DirectedEdge constructor
            new_edge_callback (Callable): called whenever a new edge is added to
              the graph (via get_edge), with the new edge as an argument
    """
    self._nodes = {}  # AbstractState --> AbstractNode
    self._start_node = self._nodes[start_state] = AbstractNode(
        start_state, min_visit_count, 0, True)
    self._edge_window_size = edge_window_size
    self._traverse_threshold = traverse_threshold
    self._uid_count = 1
    self._min_visit_count = min_visit_count
    self._max_edge_degree = max_edge_degree
    self._feasible_set = set([self._start_node])

    # feasible set union non-feasible deg-1 neighbors of feasible set
    self._neighbors_of_feasible = set([self._start_node])

    # Update the callback to update the feasible set.
    callback_key = (DirectedEdge.EVALUATING, DirectedEdge.RELIABLE)
    reliable_callback = edge_callbacks.get(callback_key, lambda edge: None)

    def eval_to_reliable(edge):
      self._feasible_set.add(edge.end)
      self._neighbors_of_feasible.add(edge.end)
      for neighbor_edge in edge.end.neighbors:
        if neighbor_edge.degree == 1:
          self._neighbors_of_feasible.add(neighbor_edge.end)
      reliable_callback(edge)

    edge_callbacks[callback_key] = eval_to_reliable
    self._callbacks = edge_callbacks
    self._new_edge_callback = new_edge_callback

  @property
  def nodes(self):
    return list(self._nodes.values())

  @property
  def edges(self):
    """Returns a list[DirectedEdge] of all edges in the graph."""
    return [edge for node in self.nodes for edge in node.neighbors]

  @property
  def feasible_set(self):
    """Returns the set of AbstractNodes that can be reached from the start

        node along reliable edges (set(AbstractNode))
        """
    return self._feasible_set

  @property
  def neighbors_of_feasible(self):
    """Returns set of AbstractNodes = feasible set union degree 1 neighbors

        of feasible set.
    """
    return self._neighbors_of_feasible

  def get_node(self, abstract_state):
    """Returns the AbstractNode associated with this state if it exists,

        otherwise creates and node, adds it to the graph and returns it.

        Args: abstract_state (AbstractState)
    """
    if abstract_state not in self._nodes:
      self._nodes[abstract_state] = AbstractNode(abstract_state,
                                                 self._min_visit_count,
                                                 self._uid_count)
      self._uid_count += 1
    return self._nodes[abstract_state]

  def get_edge(self,
               edge_start,
               edge_end,
               degree=1,
               reward=0.,
               life_lost=False):
    """Returns the DirectedEdge (edge_start --> edge_end).

    Adds
        edge_start and edge_end to the graph if they don't already exist and
        updates their neighbors / parents. Returns None if degree exceeds the
        max allowable or start == end

        Args: edge_start (AbstractState) edge_end (AbstractState)

        Returns:
            DirectedEdge | None
        """
    if degree > self._max_edge_degree or edge_start == edge_end:
      return None

    start = self.get_node(edge_start)
    end = self.get_node(edge_end)
    edge = start.get_neighbor_edge(end, degree)
    if edge is None:
      edge = DirectedEdge(start, end, self._edge_window_size,
                          self._edge_window_size, self._traverse_threshold,
                          self._callbacks, degree, reward, life_lost)
      start.add_neighbor(edge)
      end.add_parent(edge)
      self._on_new_edge(edge)
    return edge

  @property
  def traverse_threshold(self):
    """See constructor for documentation. Returns float"""
    return self._traverse_threshold

  def state_dict(self):
    """Returns all necessary information to serialize AbstractGraph.

    Can be
        reloaded with an AbstractGraph instance (constructed with the same
        configs) using load_state_dict.

        Returns:
            dict
        """
    # edge_window_size, traverse_threshold, min_visit_count,
    # max_edge_degree, callbacks reconstructed from Config.
    nodes = {
        abstract_state: node.partial_state_dict()
        for (abstract_state, node) in self._nodes.items()
    }
    feasible_set = set(node.uid for node in self._feasible_set)
    start_node = self._start_node.uid
    neighbors_of_feasible = set(
        node.uid for node in self._neighbors_of_feasible)
    edges = [(edge.partial_state_dict(), edge.start.uid, edge.end.uid)
             for node in self.nodes
             for edge in node.neighbors]
    return {
        "nodes": nodes,
        "feasible_set": feasible_set,
        "start_node": start_node,
        "neighbors_of_feasible": neighbors_of_feasible,
        "edges": edges,
        "uid_count": self._uid_count,
    }

  def load_state_dict(self, state_dict):
    """Reloads all information in the state_dict."""
    uid_to_node = {}
    for abstract_state, partial_node in state_dict["nodes"].items():
      node = AbstractNode.from_partial_state_dict(partial_node)
      self._nodes[abstract_state] = node
      uid_to_node[node.uid] = node

    self._feasible_set.clear()
    for uid in state_dict["feasible_set"]:
      self._feasible_set.add(uid_to_node[uid])

    self._uid_count = state_dict["uid_count"]
    assert self._uid_count == len(self._nodes)

    self._start_node = uid_to_node[state_dict["start_node"]]

    self._neighbors_of_feasible.clear()
    for uid in state_dict["neighbors_of_feasible"]:
      self._neighbors_of_feasible.add(uid_to_node[uid])

    for partial_edge, start_uid, end_uid in state_dict["edges"]:
      start = uid_to_node[start_uid]
      end = uid_to_node[end_uid]
      edge = DirectedEdge.from_partial_state_dict(partial_edge, start, end,
                                                  self._callbacks)
      start.add_neighbor(edge)
      end.add_parent(edge)
      # No need to call callbacks

  def _on_new_edge(self, edge):
    # Call client callback
    self._new_edge_callback(edge)

    # Update neighbor set of feasible set
    if edge.degree == 1 and edge.start in self._feasible_set:
      self._neighbors_of_feasible.add(edge.end)

  def __str__(self):
    s = ""
    # Sorted for determinism
    for node in sorted(self.nodes, key=lambda x: x.uid):
      s += "=" * 30 + "\n"
      s += str(node) + "\n"
      for edge in sorted(node.neighbors, key=lambda x: (x.end.uid, x.degree)):
        s += str(edge) + "\n"
    return s

  __repr__ = __str__


class OracleGraph(AbstractGraph):
  """Fixes the graph to only be the nodes along the path to solve the first

    room.
  """

  def __init__(self, start_state, edge_window_size, traverse_threshold,
               min_visit_count, max_edge_degree, edge_callbacks,
               new_edge_callback):
    super(OracleGraph,
          self).__init__(start_state, edge_window_size, traverse_threshold,
                         min_visit_count, max_edge_degree, edge_callbacks,
                         new_edge_callback)

    # Oracle nodes and edges for the first room
    # AbstractStates in order
    ROOM_NUMBER = 1
    path = [
        (70, 220),  # down ladder (x, y)
        (70, 210),  # down ladder
        (70, 200),  # down ladder
        (70, 190),  # bottom of ladder
        (80, 190),  # go right
        (90, 190),  # jump across rope
        (100, 190),  # on rope
        (110, 190),  # jump across rope
        (120, 190),  # on ledge
        (130, 180),  # down ladder
        (130, 170),  # down ladder
        (130, 160),  # down ladder
        (130, 150),  # bottom of ladder
        (120, 150),  # left
        (110, 150),  # left
        (100, 150),  # left
        (90, 150),  # left
        (80, 150),  # left
        (70, 150),  # left
        (60, 150),  # left
        (50, 150),  # left
        (40, 150),  # left
        (30, 150),  # left
        (20, 150),  # bottom of ladder
        (20, 160),  # up ladder
        (20, 170),  # up ladder
        (20, 180),  # up ladder
        (10, 180),  # left
        (10, 190),  # jump
        (10, 200),  # jump
    ]

    key_state = (10, 210, ROOM_NUMBER, 2, 14)

    # revisit the states in order except getting key
    backward_path = list(reversed(path)) + [
        (80, 240),  # right
        (90, 240),  # jump gap
        (100, 240),  # right gap
        (110, 240),  # right
        (120, 240),  # right
    ]

    open_door = (130, 240, ROOM_NUMBER, 0, 10)

    # Add (room #, inv mask, inv)
    for i, s in enumerate(path):
      path[i] = path[i] + (ROOM_NUMBER, 0, 15)

    for i, s in enumerate(backward_path):
      backward_path[i] = backward_path[i] + (ROOM_NUMBER, 2, 14)

    path = path + [key_state] + backward_path + [open_door]

    prev_node = self._start_node
    for i, state in enumerate(path):
      ram = np.zeros(128).astype(np.uint8)
      ram[np.array((42, 43, 3, 65, 66))] = state
      ram[42] += 10
      abstract_state = AS.AbstractState(State(ram_state=ram))
      curr_node = self._nodes[abstract_state] = AbstractNode(
          abstract_state, self._min_visit_count, self._uid_count)
      self._uid_count += 1
      edge = DirectedEdge(
          prev_node,
          curr_node,
          self._edge_window_size,
          self._edge_window_size,
          self._traverse_threshold,
          self._callbacks,
          degree=1)
      prev_node.add_neighbor(edge)
      curr_node.add_parent(edge)
      prev_node = curr_node

  def get_node(self, abstract_state):
    """If the state is along the oracle path, returns the associated

        AbstractNode, otherwise returns a dummy AbstractNode.

        Args: abstract_state (AbstractState)
    """
    if abstract_state in self._nodes:
      node = self._nodes[abstract_state]
      return node
    else:
      node = AbstractNode(abstract_state, 0, self._uid_count)
      return node

  def get_edge(self,
               edge_start,
               edge_end,
               degree=1,
               reward=0.,
               life_lost=False):
    """Adds an edge if the degree > 1, otherwise does nothing.

        Args: edge_start (AbstractState) edge_end (AbstractState)

        Returns:
            DirectedEdge
        """
    if edge_start in self._nodes and edge_end in self._nodes and \
            edge_start != edge_end and degree > 1:
      return super(OracleGraph, self).get_edge(edge_start, edge_end, degree,
                                               reward, life_lost)
    return None


class GraphWrapper(object):
  """Wraps a graph and overrides the get_node and get_edge

    functions
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, graph):
    self._graph = graph

  @property
  def graph(self):
    return self._graph

  @property
  def nodes(self):
    return self.graph.nodes

  @property
  def feasible_set(self):
    return self.graph.feasible_set

  @property
  def neighbors_of_feasible(self):
    return self.graph.neighbors_of_feasible

  @abc.abstractmethod
  def get_node(self, abstract_state):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_edge(self,
               edge_start,
               edge_end,
               degree=1,
               reward=0.,
               life_lost=False):
    raise NotImplementedError()

  @property
  def traverse_threshold(self):
    return self.graph.traverse_threshold

  def state_dict(self):
    return self._graph.state_dict()

  def load_state_dict(self, state_dict):
    self._graph.load_state_dict(state_dict)

  @property
  def edges(self):
    return self._graph.edges

  def __str__(self):
    return "{}:\n{}".format(type(self).__name__, self.graph)

  __repr__ = __str__


class Whitelist(GraphWrapper):

  def __init__(self, graph, domain):
    super(Whitelist, self).__init__(graph)

    # Whitelist (x, y, room)
    self._whitelist = set()
    with open(data.whitelist_file(domain), "r") as f:
      for line in f:
        self._whitelist.add(literal_eval(line))

    for node in self.nodes:
      for edge in node.neighbors:
        if self._x_y_room(edge.start) not in self._whitelist or \
                self._x_y_room(edge.end) not in self._whitelist:
          edge.kill()

  def get_node(self, abstract_state):
    return self.graph.get_node(abstract_state)

  def get_edge(self,
               edge_start,
               edge_end,
               degree=1,
               reward=0.,
               life_lost=False):
    """Adds the edge to the graph if it doesn't exist. If either
        endpoint is not whitelisted, the edge automatically gets a
        train count of 10000

        Args:
            edge_start (AbstractState)
            edge_end (AbstractState)

        Returns:
            DirectedEdge
        """
    edge = self.graph.get_edge(edge_start, edge_end, degree, reward, life_lost)
    if edge is None:
      return edge
    if self._x_y_room(edge.start) not in self._whitelist or \
            self._x_y_room(edge.end) not in self._whitelist:
      edge.kill()
    return edge

  def _x_y_room(self, node):
    abstract_state = node.abstract_state
    return (abstract_state.ram_x, abstract_state.ram_y,
            abstract_state.room_number)
