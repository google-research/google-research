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

"""Python object types for graphs and their corresponding POMDPs.

Section 3.1 of the paper describes a transformation from ordinary directed
graphs into POMDPs with specific observation and action spaces. The objects
defined in this file specify the Python representation of these graph POMDPs
and their action and observation spaces.

Notes on terminology:
- An "incoming edge type" represents what an agent observes when it enters a
  node by following a specific edge. Together, the set of pairs of node types
  and incoming edge types is the observation space of the agent.
- An "outgoing edge type" represents an action that an agent can take in order
  to leave a node. The set of outgoing edge types for each node type, combined
  with the set of special actions, is the action space of the agent.
- Each edge has both an incoming edge type and an outgoing edge type in addition
  to a source and destination node. This represents a single possible transition
  in the POMDP: when located at the source node and taking the action denoted by
  the out-edge type, the agent will move to the destination node and receive
  an observation based on the out-edge type.

As an example, for a graph representing an AST where B is the left child of A,
the edge from A to B might have outgoing type "left child" and incoming type
"parent"; then an agent can choose to move to the left child, and upon doing so
observes that it has arrived at that node from its parent.
"""

from typing import Dict, List, NewType
import dataclasses

# A category of nodes (i.e. AST `If` nodes). Part of the agent's observation.
NodeType = NewType("NodeType", str)

# A node in a graph (i.e. the root node)
# (Node IDs are strings instead of ints so that they can have semantic meaning.
# For instance, for abstract syntax trees the node IDs are string
# representations of paths from the root node, so that it's easier to convert
# from locations in the AST to graph IDs and vice versa.)
NodeId = NewType("NodeId", str)

# Incoming edge types, observed by the agent after a transition.
InEdgeType = NewType("InEdgeType", str)
# Outgoing edge types, representing the action space of the agent.
OutEdgeType = NewType("OutEdgeType", str)


@dataclasses.dataclass
class NodeSchema:
  """Specifies the set of possible input edges and output edges for a node.

  A node schema specifies the "local" observation space and action space for
  nodes of a particular type; the incoming edges are the observations that are
  possible when entering nodes of this type, and the outgoing edges are the
  movement actions that the agent can take at a node of this type.

  Note that, for each edge type in `out_edges`, a node conforming to this schema
  must have *at least one* outgoing edge of that type, so that the agent can
  take any of the allowed actions at this node. It is possible, however, to
  have loop edges that point back to this node, which results in no movement.

  Attributes:
    in_edges: Collection of edge types that can be used to enter this node.
    out_edges: Collection of edge types that can be used to exit this node.
  """
  in_edges: List[InEdgeType]
  out_edges: List[OutEdgeType]


@dataclasses.dataclass(frozen=True)
class InputTaggedNode:
  """A specific node tagged with the type of edge used to enter the node.

  InputTaggedNodes are used to represent the destinations of directed edges.
  Due to the Markov property, the state of the agent in the environment can be
  fully represented by the current input-tagged node, which encapsulates the
  agent's position and it's current observation, along with the agent's
  finite-state memory (stored separately).

  This class is immutable so it can be used as a key in dictionaries.

  Attributes:
    node_id: Specifies a node in a given graph.
    in_edge: The edge type that was used to arrive at this node; equivalently,
      the next observation that the agent will receive.
  """
  node_id: NodeId
  in_edge: InEdgeType


@dataclasses.dataclass
class GraphNode:
  """Represents a node in a graph.

  Attributes:
    node_type: The node type of this node.
    out_edges: Specifies, for each output edge type, a list of other nodes that
      can be reached by following an outgoing edge of that type. Each of those
      nodes is tagged with the incoming representation of the edge. This encodes
      the entire transition function of the POMDP when starting at this node.
  """
  node_type: NodeType
  out_edges: Dict[OutEdgeType, List[InputTaggedNode]]


# Graphs are collections of nodes
Graph = Dict[NodeId, GraphNode]

# A graph schema specifies all of the node and edge types that can appear in
# a graph. This defines a family of POMDPs with the same action and observation
# spaces; a single policy can be used to traverse any graph as long as the
# graphs all conform to a fixed schema.
GraphSchema = Dict[NodeType, NodeSchema]
