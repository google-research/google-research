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

# Lint as: python3
"""Finite-state automaton builder and parameter logic.

This module contains functions and objects to handle conversion of graphs into
ndarray datastructures, parameterizing automaton layers based on a schema, and
using automaton parameters to instantiate a concrete transition matrix for a
given converted graph.
"""

import enum
import itertools
from typing import Dict, Optional, Tuple, Union

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np

from gfsa import graph_types
from gfsa import jax_util
from gfsa import linear_solvers
from gfsa import schema_util
from gfsa import sparse_operator

# Sentinel incoming edge type, indicating that this is the initial position of
# the automaton in the graph (since it does not have a real incoming edge type
# until it moves for the first time).
SOURCE_INITIAL = graph_types.InEdgeType("INITIAL")


class SpecialActions(enum.Enum):
  """Special actions for the automaton, which are valid in any state."""
  FINISH = "FINISH"
  BACKTRACK = "BACKTRACK"
  FAIL = "FAIL"


@dataclasses.dataclass(frozen=True)
class InRouteType:
  """A type of node along with how that node is entered.

  Each InRouteType is an observation for the automaton agent, combining the
  current node type with the incoming edge type observation.

  This data structure is primarily used as a key into a dictionary of
  observation-dependent parameters. For instance, a policy should specify the
  probability of taking each special action for any given InRouteType.

  Instances are immutable so that they can be used as dictionary keys.

  Attributes:
    node_type: The type of node for this route.
    in_edge: The edge type used to enter this node.
  """
  node_type: graph_types.NodeType
  in_edge: graph_types.InEdgeType


@dataclasses.dataclass(frozen=True)
class InOutRouteType:
  """A type of node along with how that node is entered and exited.

  An InOutRouteType combines both an observation and a movement action. This
  is primarily used as a key into a dictionary of observation-and-action-
  dependent parameters. For instance, for each InOutRouteType, a policy should
  specify the probability of taking that movement action given the observation.

  Instances are immutable so that they can be used as dictionary keys.

  Attributes:
    node_type: The type of node for this route.
    in_edge: The edge type used to enter this node.
    out_edge: The edge type used to exit this node (i.e. the next action).
  """
  node_type: graph_types.NodeType
  in_edge: graph_types.InEdgeType
  out_edge: graph_types.OutEdgeType


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class RoutingParams:
  # pyformat: disable
  r"""Abstract routing parameters for an automaton, given a fixed schema.

  Routing parameters specify a distribution over possible actions conditioned
  on a current node type, the incoming edge type, the current automaton state,
  and a "variant" that encodes the relationship between the current node and
  the start node of the automaton. (For instance, there might be two variants,
  one for variables with the same identifier, and one for any other pair of
  nodes; the transition matrix would then allow the agent to observe whether
  a given variable matches the initial variable name. Or, there might be one for
  nodes that are "similar" under a dot-product key-value similiarity
  measure, and one for nodes that are not; the final transition matrix would
  then interpolate between the two variants depending on how similar the node
  embeddings are.)

  Variants correspond to the start-node-conditioned observations described in
  Appendix C.2, denoted there with `\gamma \in \Gamma`.

  Interpretation of these parameters depends on the route orderings of the
  corresponding AutomatonBuilder. Note that the distribution over actions for
  a single input route type is distributed among multiple indices of both
  attributes: output parameters for special actions appear in `special`, and
  output parameters for moving across an out edge and changing the state of the
  finite state machine state appear in `move`. The AutomatonBuilder's metadata
  is therefore necessary to properly normalize across all possible options.

  Attributes:
    move: <float32[variants, in_out_routes, fsm_states, fsm_states]> array,
      where:

        move[v_t, (node_type, in_edge, out_edge), s_t, s_{t+1}]
          = p( a_t="move across out_edge", s_{t+1}
               | node_type, in_edge, s_t, v_t)

    special: <float32[variants, in_routes, fsm_states, special_actions]>
    array,
      where:

        special[v_t, (node_type, in_edge), s_t, a_t]
          = p( a_t | node_type, in_edge, s_t, v_t )
  """
  # pyformat: enable
  move: jax_util.NDArray
  special: jax_util.NDArray


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class RoutingGateParams:
  # pyformat: disable
  """Alternative parameterization for an automaton, given a fixed schema.

  Note: this parameterization was NOT used for the experiments in the paper.

  RoutingGateParams parameterizes the automaton behavior based on a collection
  of sigmoid gates that determine what actions are allowed, interpreting FAIL
  and BACKTRACK special actions as fallbacks only for when no other action is
  possible.

  The advantage of this parameterization is that we can apply regularization to
  each of the gates to be either fully on or fully off, and then normalize
  after doing this. This means we can say that [0, 1], [1, 0], and [0.5, 0.5]
  are all acceptable with no regularization penalty, but something like
  [0.3, 0.7] is not allowed. Ideally, this will encourage the model not to rely
  on the exact continuous values but instead simply turn connections on and off.

  For now, this only supports automata with all three special actions.

  Attributes:
    move_gates: <float32[variants, in_out_routes, fsm_states, fsm_states]>
      of (binary) logits determining which movements are accepted.

    accept_gates: <float32[variants, in_routes, fsm_states]> array,
      of (binary) logits determining whether we can accept in a given state.

    backtrack_gates: <float32[variants, in_routes, fsm_states]> array,
      of (binary) logits determining whether we should backtrack, given that
      no accept or move gate is open.
  """
  # pyformat: enable
  move_gates: jax_util.NDArray
  accept_gates: jax_util.NDArray
  backtrack_gates: jax_util.NDArray


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class EncodedGraph:
  """Encoding of a graph POMDP as a structure of ndarrays, given a fixed schema.

  This object encodes the full environment dynamics of a specific POMDP, and
  maps from automaton actions to new observations and new states.

  Together with the automaton's policy routing table, this can be used to
  construct the full transition matrix for an absorbing Markov chain. To make
  this efficient, the graph representation we use is a direct encoding
  of the indices and weights required to do this conversion. (Note that the
  indices in this representation depend on the orderings defined by the
  corresponding AutomatonBuilder.)

  More specifically, we can think of every element of the transition matrix as
  being a linear combination of elements from the routing table (usually, each
  element of the transition matrix is either 0 or a weighted element of the
  abstract routing table, but there are a few cases where a single concrete
  transition corresponds to multiple abstract actions). We thus
  represent the graph as a sparse linear operator that transforms the routing
  table into the transition matrix.

  In order to efficiently solve our linear system, it is useful to separate
  out the normal transitions (moving from one node to another) from the special
  transitions (starting the automaton at a node, or having the automaton take
  a special action) which we need to handle separately. We thus store the
  transition matrix in four blocks:

  1. from an initial state (starting at some node), moving to another node
  2. from an initial state (starting at some node), taking a special action
  3. from a normal state (entering a node via an edge), moving to another node
  4. from a normal state (entering a node via an edge), taking a special action

  Parts 1 and 3 depend on the output edge types, so they index into a
  parameter vector of in-out-route types, and produce blocks based on the output
  edges in the graph. Blocks 2 and 4 do not depend on output edge types, so they
  index into a parameter vector of in-route types and produce blocks that depend
  only on the node types, not the output edges.

  Note that many aspects of the actual transition matrix are independent of
  the graph structure (i.e. there is a fixed number of special actions, a fixed
  number of automaton states, etc). This information is not encoded in the
  graph representation; for instance, the graph is encoded as if there is only
  one special action, and the full set of special actions is instantiated by
  mapping the same encoded-graph linear operator across the special action
  parameter vector.

  In addition to these blocks, we store a node aggregation array that associates
  each input-tagged node to its non-input-tagged index (i.e. discards the input
  information), which allows us to convert the state of the automaton to a node
  embedding once the automaton stops.

  Attributes:
    initial_to_in_tagged: SparseCoordOperator that maps from an array of
      InOutRouteTypes to block 1 of the transition matrix, which is of shape
      [nodes, in_route_types].
    initial_to_special: Array of indices of shape [nodes], which indexes into an
      array of InRouteTypes to produce block 2 of the matrix, (i.e. one special
      action parameter concrete node, representing the initial state)
    in_tagged_to_in_tagged: SparseCoordOperator that maps from an array of
      InOutRouteTypes to block 3 of the transition matrix, which is of shape
      [in_route_types, in_route_types].
    in_tagged_to_special: Array of indices of shape [in_route_types], which
      indexes into an array of InRouteTypes to produce block 4 of the matrix,
      (i.e. one special action parameter per concrete input-tagged node)
    in_tagged_node_indices: Array of indices of shape [in_route_types], which
      indexes into an array of untagged nodes (of shape [nodes]), and can be
      used to convert between an input-tagged representation and a
      non-input-tagged one.
  """
  initial_to_in_tagged: sparse_operator.SparseCoordOperator
  initial_to_special: jax_util.NDArray
  in_tagged_to_in_tagged: sparse_operator.SparseCoordOperator
  in_tagged_to_special: jax_util.NDArray
  in_tagged_node_indices: jax_util.NDArray


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class EncodedGraphMetadata:
  """Dimension-size metadata for an encoded graph.

  To facilitate batching, we store the sizes of the graph separately from the
  nonzero entries of the graph, since when batching graphs together, you want
  the MAXIMUM size (and it should be fine to pretend a graph is larger than it
  actually is).

  Attributes:
    num_nodes: Number of nodes in the graph.
    num_input_tagged_nodes: Number of concrete input-tagged nodes that are
      reachable in the graph.
  """
  num_nodes: int
  num_input_tagged_nodes: int


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class AutomatonGraphTransitionMatrix:
  # pyformat: disable
  """Transition matrix for a parameterized automaton on a specific graph.

  This class represents the combination of RoutingParams and an EncodedGraph,
  and specifies the transition matrix for the automaton moving on that specific
  graph. See the docstrings of EncodedGraph and RoutingParams for details.

  Attributes:

    initial_to_in_tagged:
      Array <float32[variants, nodes, fsm_states, in_tagged_nodes, fsm_states]>
      where

        initial_to_in_tagged[v, n_0, s_0, (n_1, in_edge_1), s_1] =
          p(n_1, in_edge_1, s_1 | v, n_0, s_0)

    initial_to_special:
      Array <float32[variants, nodes, fsm_states, special_actions]> where

        initial_to_special[v, n_0, s_0, a_1] = p(a_1 | v, n_0, s_0)

    in_tagged_to_in_tagged:
      Array <float32[variants, in_tagged_nodes, fsm_states, in_tagged_nodes,
                     fsm_states]> where

        in_tagged_to_in_tagged[v, (n_t, in_edge_t), s_t,
                               (n_{t+1}, in_edge_{t+1}), s_{t+1}] =
          p(n_{t+1}, in_edge_{t+1}, s_{t+1} | v, n_t, in_edge_t, s_t)

    in_tagged_to_special:
      Array <float32[variants, in_tagged_nodes, fsm_states, special_actions]>
      where

        in_tagged_to_special[v, (n_t, in_edge_t), s_t, a_t] =
          p(a_t | v, n_t, in_edge_t, s_t)

    in_tagged_node_indices: Array of indices of shape [in_route_types], which
      indexes into an array of untagged nodes (of shape [nodes]), and can be
      used to convert between an input-tagged representation and a
      non-input-tagged one. (Note that this is a copy of the equivalent
      attribute of EncodedGraph, stored here as well for simplicity.)
  """
  # pyformat: enable
  initial_to_in_tagged: jax_util.NDArray
  initial_to_special: jax_util.NDArray
  in_tagged_to_in_tagged: jax_util.NDArray
  in_tagged_to_special: jax_util.NDArray
  in_tagged_node_indices: jax_util.NDArray

  def concatenated_transitions(self):
    """Returns a concatenated version of all blocks of the transition matrix.

    This is not an efficient way to solve for final states, but can be useful
    for testing, visualization, and analysis.

    Returns:
      Array of shape [variants, nodes+in_tagged_nodes, fsm_states,
                      (in_tagged_nodes * fsm_states) + special_actions] where

       result[v, pos, state, action] = p(action | v, pos, state)
    """
    return jnp.concatenate(
        [
            jnp.concatenate([
                self.initial_to_in_tagged.reshape(
                    self.initial_to_in_tagged.shape[:3] + (-1,)),
                self.initial_to_special,
            ], -1),
            jnp.concatenate([
                self.in_tagged_to_in_tagged.reshape(
                    self.in_tagged_to_in_tagged.shape[:3] + (-1,)),
                self.in_tagged_to_special,
            ], -1),
        ],
        axis=1,
    )


class AutomatonBuilder:
  """Helps parameterize and apply automata to graphs with a fixed schema.

  An AutomatonBuilder instance takes a graph schema, extracts all possible
  node types and route types, and defines a canonical flattened ordering for
  each. It then provides methods that use these orderings to:

  - Build initial parameters for the automaton based on the route types
  - Normalize automaton parameters so that they represent valid probability
    distributions (while handling mismatched action space sizes)
  - Encode raw graphs (represented as Python objects) into EncodedGraphs
    (represented as indices based on those orderings)
  - Compute each block of the concrete transition matrix based on an encoded
    graph and automaton parameters

  Attributes:
    schema: The schema that this AutomatonBuilder corresponds to.
    special_actions: List of special actions the automaton can take. These
      should all be members of the enum `SpecialActions`.
    node_types: List of node types in the schema.
    in_route_types: List of node types with incoming edge types.
    in_out_route_types: List of node types with incoming and outgoing edge
      types.
    node_type_to_index: Dictionary from each node type to its index in
      `node_types`.
    in_route_type_to_index: Dictionary from each in-route type to its index in
      `in_route_types`.
    in_out_route_type_to_index: Dictionary from each in-out-route type to its
      index in `in_out_route_types`.
    in_route_to_node_type: Dictionary from an in-route index to the
      corresponding node type index (ignoring the incoming edge).
    in_out_route_to_in_route: Dictionary from an in-out-route index to the
      corresponding in-route type index (ignoring the outgoing edge).
  """

  def __init__(self,
               schema,
               with_backtrack=True,
               with_fail=True):
    """Creates an AutomatonBuilder with a specific schema.

    Args:
      schema: Graph schema to use.
      with_backtrack: Enable "backtrack" special action, which restarts from the
        initial state.
      with_fail: Enable "fail" special action, which stops the automaton without
        marking the current node as the final node.
    """
    self.schema = schema

    self.special_actions = [SpecialActions.FINISH]
    if with_backtrack:
      self.special_actions.append(SpecialActions.BACKTRACK)
    if with_fail:
      self.special_actions.append(SpecialActions.FAIL)

    self.node_types = []
    self.in_route_types = []
    self.in_out_route_types = []
    self.in_route_to_node_type = []
    self.in_out_route_to_in_route = []
    for node_type, node_schema in schema.items():
      self.node_types.append(node_type)
      for source in itertools.chain(node_schema.in_edges, [SOURCE_INITIAL]):
        self.in_route_types.append(
            InRouteType(node_type=node_type, in_edge=source))
        self.in_route_to_node_type.append(len(self.node_types) - 1)
        for edge in node_schema.out_edges:
          self.in_out_route_types.append(
              InOutRouteType(
                  node_type=node_type, in_edge=source, out_edge=edge))
          self.in_out_route_to_in_route.append(len(self.in_route_types) - 1)

    self.node_type_to_index = {
        node_type: i for i, node_type in enumerate(self.node_types)
    }
    self.in_route_type_to_index = {
        route: i for i, route in enumerate(self.in_route_types)
    }
    self.in_out_route_type_to_index = {
        route: i for i, route in enumerate(self.in_out_route_types)
    }

  def routing_reduce(self, rparams,
                     reduction):
    """Replaces each element with a reduction over entries in its distribution.

    The routing parameters contain a packed sequence of differently-sized
    categorical distributions: for each InRouteType, automaton state, and
    variant, there is a distribution over all actions (outgoing edge types plus
    special actions), which we store as a flat sequence of routing params.
    This function computes an aggregated value for each of these
    distributions, then scatters this constant back out over the routing
    parameters, such that each element is replaced with the aggregated value
    for its distribution.

    When reduction is "sum", this computes the normalization constant for the
    distribution. For a properly-normalized distribution, this should return an
    array of all ones. For an unnormalized distribution, dividing elementwise
    by the return value will normalize the distribution.

    When reduction is "max", computes the maximum probability in the
    distribution, useful for computing softmax.

    Args:
      rparams: RoutingParams corresponding to this AutomatonBuilder.
      reduction: What type of reduction to do. One of "sum", "max"

    Returns:
      RoutingParams containing the normalization constant. Each component will
      be broadcastible with the original input, but may not be exactly the
      same shape.
    """
    if reduction == "sum":
      arr_reduce = jnp.sum
      index_reduce = jax.ops.index_add
    elif reduction == "max":
      arr_reduce = jnp.max
      index_reduce = jax.ops.index_max
    else:
      raise ValueError(f"Invalid reduction {reduction}")

    # Convert our reindexing list into a JAX ndarray
    in_out_route_to_in_route_arr = jnp.array(self.in_out_route_to_in_route)

    # Reduce over special actions for each distribution
    distn_reduced_spec = arr_reduce(rparams.special, axis=-1)
    # Scatter in contribution of all outputs (summing over in-out-route values
    # with the same in-route type, and also over all possible next FSM states)
    distn_reduced = index_reduce(
        distn_reduced_spec,
        jax.ops.index[:, in_out_route_to_in_route_arr],
        arr_reduce(rparams.move, axis=-1),
    )

    # Now expand and scatter the reduced back out
    rparams_reduced = RoutingParams(
        special=distn_reduced[:, :, :, None],
        move=distn_reduced[:, in_out_route_to_in_route_arr, :, None])
    return rparams_reduced

  def routing_softmax(self, log_rparams):
    """Apply the softmax transformation to the routing params.

    This transforms parameters in log space into normalized probabilities,
    taking care not to cause overflows. It is mathematically equivalent to

      rparams = np.exp(log_rparams)
      return rparams / routing_reduce(rparams, "sum")

    (although it is more numerically stable and correctly handles the
    RoutingParams dataclass).

    Args:
      log_rparams: Log space routing params.

    Returns:
      Normalized routing probabilities.
    """
    # Compute the maximum value for each distribution so that we can subtract
    # it out before doing the softmax; this increases numerical stability
    # without changing the output.
    max_log_rparams = self.routing_reduce(
        jax.lax.stop_gradient(log_rparams), "max")

    # Subtract the stability constants and then exponentiate.
    unnorm_rparams = jax.tree_multimap(lambda v, m: jnp.exp(v - m), log_rparams,
                                       max_log_rparams)

    # Normalize the result.
    return jax.tree_multimap(jax.lax.div, unnorm_rparams,
                             self.routing_reduce(unnorm_rparams, "sum"))

  def initialize_routing_params(self,
                                key,
                                num_fsm_states,
                                num_variants,
                                state_change_prob=0.2,
                                move_prob=0.95,
                                noise_factor=0.01):
    """Builds initial routing parameters.

    We start with a base policy that:
    - determines whether to move or take a special action with probability
      `move_prob`
    - if it moves, either stays in the same state or changes states uniformly
      at random
    - chooses a node to move to uniformly at random, or a special action
      uniformly at random

    Then, if `noise_factor` is nonzero, we break symmetry by sampling a
    perturbed policy from a Dirichlet distribution whose expected value is this
    base policy. The concentration parameter for each component `i` of the
    sampled policy is set to `p_i / noise_factor` where `p_i` is the base policy
    probability for that component; as `noise_factor` approaches 0 the variance
    of the Dirichlet also approaches 0 and the sampled policy approaches the
    base policy, and as `noise_factor` grows to infinity the sampled policy
    becomes a random deterministic policy.

    Args:
      key: PRNGKey to use. May be None if noise_factor is 0
      num_fsm_states: Number of finite states for the automaton.
      num_variants: Number of variants of the routing table to build.
      state_change_prob: Probability of changing states.
      move_prob: Probability of moving (instead of taking a special action).
      noise_factor: How much noise to scale by; should be nonnegative.

    Returns:
      Properly-normalized RoutingParams describing initial policy.
    """
    if num_fsm_states < 1:
      raise ValueError("Automaton must have at least one state")
    elif num_fsm_states == 1:
      # With only one state, we ignore the state change probability
      state_transition_block = np.ones([1, 1])
    else:
      # Divide state-change probability evenly among the other states
      p_change = state_change_prob / (num_fsm_states - 1)
      p_stay = 1 - state_change_prob
      state_transition_block = (
          p_change * np.ones([num_fsm_states, num_fsm_states]) +
          (p_stay - p_change) * np.eye(num_fsm_states))

    if num_variants < 1:
      raise ValueError("Routing table must have at least one variant")

    # Build move parameters by dividing mass uniformly among edge types
    move_params = np.zeros([
        num_variants,
        len(self.in_out_route_types), num_fsm_states, num_fsm_states
    ])
    for node_type, node_schema in self.schema.items():
      for source in itertools.chain(node_schema.in_edges, [SOURCE_INITIAL]):
        per_edge_block = (
            move_prob / len(node_schema.out_edges) * state_transition_block)
        for out_edge in node_schema.out_edges:
          ioroute = InOutRouteType(
              node_type=node_type, in_edge=source, out_edge=out_edge)
          move_params[:, self.in_out_route_type_to_index[ioroute]] = (
              per_edge_block[None, :, :])

    move_params = jnp.array(move_params)

    # Build special action parameters by dividing mass evenly among actions
    special_each_prob = (1. - move_prob) / len(self.special_actions)
    special_params = jnp.full([
        num_variants,
        len(self.in_route_types),
        num_fsm_states,
        len(self.special_actions),
    ], special_each_prob)

    if noise_factor == 0:
      routing_params = RoutingParams(move=move_params, special=special_params)
    else:
      if key is None:
        raise ValueError("key must be a PRNGKey if noise_factor != 0")
      if noise_factor < 0:
        raise ValueError("noise_factor must be nonnegative")

      # Sample independent Gamma variates with the appropriate concentrations.
      k1, k2 = jax.random.split(key)
      move_params = jax.random.gamma(k1, move_params / noise_factor)
      special_params = jax.random.gamma(k2, special_params / noise_factor)

      # Re-normalize, producing a sample from the desired Dirichlet.
      unnorm_routing_params = RoutingParams(
          move=move_params, special=special_params)
      normalizer = self.routing_reduce(unnorm_routing_params, "sum")
      routing_params = jax.tree_multimap(jax.lax.div, unnorm_routing_params,
                                         normalizer)

    return routing_params

  def routing_gates_to_probs(self,
                             routing_gates):
    """Convert from routing gate parameters to routing probabilities.

    This is an alternative parameterization for the automaton: instead of
    parameterizing the policy based on a (logit) distribution over actions, we
    parameterize it as a collection of sigmoid gates. This is intended to be
    a more direct match for an FSA, where there is a set of allowed transitions.

    Note that this was not used for the experiments in the paper.

    At a high level, the process is:
      - Transform the accept and move gates into probabilities using sigmoid.
      - If at least one gate is open (sum >= 1), normalize to sum to 1.
      - Otherwise, assign remainder to backtrack or fail according to the
        backtrack gate.

    Args:
      routing_gates: Current routing gate parameters, as binary logits.

    Returns:
      Properly-normalized routing probability distribution.
    """
    assert self.special_actions == [
        SpecialActions.FINISH, SpecialActions.BACKTRACK, SpecialActions.FAIL
    ]

    move_gate_probs = jax.nn.sigmoid(routing_gates.move_gates)
    accept_gate_probs = jax.nn.sigmoid(routing_gates.accept_gates)

    backtrack_gate_probs = jax.nn.sigmoid(routing_gates.backtrack_gates)
    fail_gate_probs = jax.nn.sigmoid(-routing_gates.backtrack_gates)

    move_or_accept_totals = self.routing_reduce(
        RoutingParams(
            move=move_gate_probs,
            special=accept_gate_probs[Ellipsis, None],
        ), "sum")

    # If sum >= 1, normalize by dividing by the sum.
    norm_by_divide = RoutingParams(
        move=move_gate_probs,
        special=jnp.pad(
            accept_gate_probs[Ellipsis, None], [(0, 0), (0, 0), (0, 0), (0, 2)],
            mode="constant",
            constant_values=0))
    norm_by_divide = jax.tree_multimap(jax.lax.div, norm_by_divide,
                                       move_or_accept_totals)

    # If sum <= 1, assign remaining mass to backtracking or failing.
    special_left_over = 1 - jnp.squeeze(move_or_accept_totals.special, -1)
    norm_by_stop = RoutingParams(
        move=move_gate_probs,
        special=jnp.stack(
            arrays=[
                accept_gate_probs,
                special_left_over * backtrack_gate_probs,
                special_left_over * fail_gate_probs,
            ],
            axis=-1))

    result = jax.tree_multimap(
        lambda total, nbd, nbs: jnp.where(total >= 1, nbd, nbs),
        move_or_accept_totals, norm_by_divide, norm_by_stop)

    return result

  def initialize_routing_gates(
      self,
      key,
      num_fsm_states,
      num_variants,
      move_same_logit = 1.,
      move_different_logit = -2,
      accept_logit = -1,
      backtrack_logit = 1.,
      logistic_noise = 0.2):
    """Initialize routing gate parameterization.

    We sample each of the gate parameters as a logistic random variable with
    a type-specific mean and a shared variance.

    Note that this was not used for the experiments in the paper.

    Args:
      key: PRNGKey to use. May be None if logistic_noise is 0
      num_fsm_states: Number of finite states for the automaton.
      num_variants: Number of variants of the routing table to build.
      move_same_logit: Mean logit value for moving and staying in the same
        state.
      move_different_logit: Mean logit value for moving and changing states.
      accept_logit: Mean logit value for accepting.
      backtrack_logit: Mean logit value for backtrack (vs fail).
      logistic_noise: Scale factor for logistic noise.

    Returns:
      Routing gate parameters.
    """
    if num_fsm_states < 1:
      raise ValueError("Automaton must have at least one state")

    if num_variants < 1:
      raise ValueError("Routing table must have at least one variant")

    state_transition_block = (
        move_different_logit * np.ones([num_fsm_states, num_fsm_states]) +
        (move_same_logit - move_different_logit) * np.eye(num_fsm_states))

    # Copy our state transition block into each move gate.
    move_gates = np.zeros([
        num_variants,
        len(self.in_out_route_types), num_fsm_states, num_fsm_states
    ])
    for node_type, node_schema in self.schema.items():
      for source in itertools.chain(node_schema.in_edges, [SOURCE_INITIAL]):
        for out_edge in node_schema.out_edges:
          ioroute = InOutRouteType(
              node_type=node_type, in_edge=source, out_edge=out_edge)
          move_gates[:, self.in_out_route_type_to_index[ioroute]] = (
              state_transition_block[None, :, :])

    special_shape = (num_variants, len(self.in_route_types), num_fsm_states)

    if logistic_noise == 0:
      return RoutingGateParams(
          move_gates=jnp.array(move_gates),
          accept_gates=jnp.full(special_shape, accept_logit),
          backtrack_gates=jnp.full(special_shape, backtrack_logit))
    else:
      if key is None:
        raise ValueError("key must be a PRNGKey if logistic_noise != 0")
      if logistic_noise < 0:
        raise ValueError("logistic_noise must be nonnegative")

      k1, k2, k3 = jax.random.split(key, 3)
      return RoutingGateParams(
          move_gates=(
              move_gates +
              logistic_noise * jax.random.logistic(k1, move_gates.shape)),
          accept_gates=(
              accept_logit +
              logistic_noise * jax.random.logistic(k2, special_shape)),
          backtrack_gates=(
              backtrack_logit +
              logistic_noise * jax.random.logistic(k3, special_shape)))

  def encode_graph(self,
                   graph,
                   as_jax=True):
    """Converts a Python graph representation into ndarrays.

    In the encoded graph, order of untagged nodes will match the order of the
    keys in the graph, and the order of input-tagged nodes will match the order
    given by schema_util.all_input_tagged_nodes.

    Args:
      graph: Graph conforming to this builder's schema
      as_jax: Whether to convert arrays to JAX types (or leave it as numpy)

    Returns:
      EncodedGraph containing JAX or numpy arrays, along with metadata about
      the (implicit) size of the transition matrix.
    """
    # Fix an ordering for our untagged nodes and tagged nodes
    node_ids = list(graph.keys())
    in_tagged_nodes = schema_util.all_input_tagged_nodes(graph)
    n_nodes = len(node_ids)
    n_in_tagged_nodes = len(in_tagged_nodes)
    node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    in_tagged_node_to_index = {
        in_tagged: i for i, in_tagged in enumerate(in_tagged_nodes)
    }

    # Initialize our storage with empty values
    result = EncodedGraph(
        initial_to_in_tagged=sparse_operator.SparseCoordOperator(
            input_indices=[], output_indices=[], values=[]),
        initial_to_special=np.full([n_nodes], -1, dtype=np.int32),
        in_tagged_to_in_tagged=sparse_operator.SparseCoordOperator(
            input_indices=[], output_indices=[], values=[]),
        in_tagged_to_special=np.full([n_in_tagged_nodes], -1, dtype=np.int32),
        in_tagged_node_indices=np.full([n_in_tagged_nodes], -1, dtype=np.int32),
    )

    # Iterate over all concrete input-tagged nodes in the graph, including
    # "extended" input-tagged-nodes representing the initial state
    extended_in_tagged_nodes = in_tagged_nodes + [
        graph_types.InputTaggedNode(node_id, SOURCE_INITIAL)
        for node_id in node_ids
    ]
    for in_tagged in extended_in_tagged_nodes:
      node_index = node_id_to_index[in_tagged.node_id]
      node = graph[in_tagged.node_id]

      if in_tagged.in_edge != SOURCE_INITIAL:
        in_tagged_index = in_tagged_node_to_index[in_tagged]

        # Map from concrete ITN to node index
        result.in_tagged_node_indices[in_tagged_index] = node_index

      # Populate outgoing connections
      for out_ty, destinations in node.out_edges.items():
        # We evenly divide weight across each output edge
        weight = 1.0 / len(destinations)
        for dest in destinations:
          # Abstract input-output node, representing the transition type
          ioroute = InOutRouteType(
              node_type=node.node_type,
              in_edge=in_tagged.in_edge,
              out_edge=out_ty)
          # Concrete destination of this particular connection
          dest_index = in_tagged_node_to_index[dest]
          # Configure concrete graph encoding to point to abstract transition
          if in_tagged.in_edge == SOURCE_INITIAL:
            result.initial_to_in_tagged.input_indices.append(
                [self.in_out_route_type_to_index[ioroute]])
            result.initial_to_in_tagged.output_indices.append(
                [node_index, dest_index])
            result.initial_to_in_tagged.values.append(weight)
          else:
            result.in_tagged_to_in_tagged.input_indices.append(
                [self.in_out_route_type_to_index[ioroute]])
            result.in_tagged_to_in_tagged.output_indices.append(
                [in_tagged_index, dest_index])
            result.in_tagged_to_in_tagged.values.append(weight)

      # Configure special actions based on the abstract input-tagged node
      iroute = InRouteType(node_type=node.node_type, in_edge=in_tagged.in_edge)
      if in_tagged.in_edge == SOURCE_INITIAL:
        result.initial_to_special[node_index] = self.in_route_type_to_index[
            iroute]
      else:
        result.in_tagged_to_special[in_tagged_index] = (
            self.in_route_type_to_index[iroute])

    # Convert temporary lists to ndarrays
    result = dataclasses.replace(
        result,
        initial_to_in_tagged=sparse_operator.SparseCoordOperator(
            input_indices=np.array(
                result.initial_to_in_tagged.input_indices, dtype=np.int32),
            output_indices=np.array(
                result.initial_to_in_tagged.output_indices, dtype=np.int32),
            values=np.array(
                result.initial_to_in_tagged.values, dtype=np.float32)),
        in_tagged_to_in_tagged=sparse_operator.SparseCoordOperator(
            input_indices=np.array(
                result.in_tagged_to_in_tagged.input_indices, dtype=np.int32),
            output_indices=np.array(
                result.in_tagged_to_in_tagged.output_indices, dtype=np.int32),
            values=np.array(
                result.in_tagged_to_in_tagged.values, dtype=np.float32)))

    # Convert ndarrays to JAX if required
    if as_jax:
      result = jax.tree_map(jnp.array, result)

    metadata = EncodedGraphMetadata(
        num_nodes=n_nodes, num_input_tagged_nodes=n_in_tagged_nodes)
    return result, metadata

  def build_transition_matrix(
      self, rparams, graph,
      metadata):
    """Builds the transition matrix for the automaton.

    Args:
      rparams: Automaton parameters that specify the behavior of the automaton
      graph: Encoded representation of the graph to traverse
      metadata: Metadata specifying the size of the transition matrix

    Returns:
      Transition matrix for the automaton on this graph.
    """
    num_variants, _, num_states, _ = rparams.move.shape
    block_1 = graph.initial_to_in_tagged.apply_add(
        in_array=rparams.move,
        out_array=jnp.zeros([
            num_variants,
            metadata.num_nodes,
            num_states,
            metadata.num_input_tagged_nodes,
            num_states,
        ]),
        in_dims=(1,),
        out_dims=(1, 3),
    )
    block_2 = rparams.special[:, graph.initial_to_special, :, :]
    block_3 = graph.in_tagged_to_in_tagged.apply_add(
        in_array=rparams.move,
        out_array=jnp.zeros([
            num_variants,
            metadata.num_input_tagged_nodes,
            num_states,
            metadata.num_input_tagged_nodes,
            num_states,
        ]),
        in_dims=(1,),
        out_dims=(1, 3),
    )
    block_4 = rparams.special[:, graph.in_tagged_to_special, :, :]

    return AutomatonGraphTransitionMatrix(block_1, block_2, block_3, block_4,
                                          graph.in_tagged_node_indices)


def all_nodes_absorbing_solve(builder,
                              transition_matrix,
                              variant_weights,
                              start_machine_states,
                              steps,
                              backtrack_fails_prob = 0.001,
                              explicit_conv = False):
  r"""Compute absorbing probabilities from each node to each other node.

  This function computes the matrix \hat{A} described in Section 3.3.

  Args:
    builder: The automaton builder associated with this Markov chain.
    transition_matrix: The per-variant transition matrix for this Markov chain.
    variant_weights: Weights to assign to each routing variant for each node, as
      a <float[num_nodes, num_nodes, num_variants]> array (which should sum to 1
      across the last axis).
    start_machine_states: Initial machine state distribution for the starting
      nodes, as a <float[num_nodes, num_fsm_states]> array (which should sum to
      1 across the last axis).
    steps: How many steps of the automaton to simulate. Note that nonabsorbing
      paths longer than this will not contribute to the output, and will cause
      the gradients to be biased.
    backtrack_fails_prob: Probability that a BACKTRACK action is treated like a
      FAIL action. This makes sure that the automaton will have a higher success
      probability by directly going to the solution instead of immediately
      backtracking (otherwise, backtracking along the optimal path doesn't
      change the absorbing probabilities even though doing so is less
      numerically stable and more confusing).
    explicit_conv: Whether to explicitly write out the inner step as a conv.
      This used to improve performance but doesn't always seem to.

  Returns:
    Probabilities of absorbing at each node, as a <float[num_nodes, num_nodes]>
      array. These values will take into account backtracking, but may sum to
      less than 1 along the last axis because of FAIL states or truncated
      solving.
  """
  num_nodes, num_fsm_states = start_machine_states.shape
  num_variants = variant_weights.shape[-1]
  num_input_tagged_nodes = transition_matrix.in_tagged_to_in_tagged.shape[1]
  in_tagged_idxs = transition_matrix.in_tagged_node_indices

  # Move the variant axis to the beginning, for consistency with the transition
  # matrix and to reduce bad XLA padding.
  variant_weights = jnp.transpose(variant_weights, [2, 0, 1])
  per_itn_variant_weights = variant_weights[:, :, in_tagged_idxs]

  # Swap and combine input_tagged_nodes and fsm_states dimensions.
  initial_to_in_tagged = transition_matrix.initial_to_in_tagged.transpose(
      (0, 1, 2, 4, 3),).reshape([
          num_variants,
          num_nodes,
          num_fsm_states,
          num_fsm_states * num_input_tagged_nodes,
      ])
  in_tagged_to_in_tagged = transition_matrix.in_tagged_to_in_tagged.transpose(
      (0, 2, 1, 4, 3),).reshape([
          num_variants,
          num_fsm_states * num_input_tagged_nodes,
          num_fsm_states * num_input_tagged_nodes,
      ])

  # Run initial transition.
  # (This transition runs before the solve because, except for backtracking,
  # we will never return to the initial state.)
  initial_state = jnp.einsum("vnn,ns,vnsj->nj", variant_weights,
                             start_machine_states, initial_to_in_tagged)

  # Solve for visit counts.
  def solve_one(per_itn_variant_weights_slice, initial_state_slice):
    """Solve for a single chain's visit counts."""

    def i_minus_q(state_slice):
      """Compute (I-Q)x, the linear operator to invert."""
      # Extract per-input-tagged-node variants.
      per_itn_variants = jnp.tile(per_itn_variant_weights_slice,
                                  (1, num_fsm_states))
      # Do the einsum in two steps to make sure they happen in an efficient
      # order after batching.
      with_variants = jnp.einsum("vi,i->vi", per_itn_variants, state_slice)

      if explicit_conv:
        # On TPU, all einsums turn into generalized convolutions, but the
        # default heuristics for this sometimes cause a lot of padding to be
        # added. Instead, we can manually transform the einsum "vi,vij->j" into
        # a convolution, selecting the largest two dimensions (i and j) as the
        # "feature dimensions" to get an efficient implementation without
        # padding.
        # Note: This seems to help sometimes and hurt other times; in
        # particular, the backwards pass for this sometimes compiles to a worse
        # implementation.
        # vi ,vij-> j
        # 0CN,0IO->0CN
        new_state = jax.lax.conv_general_dilated(
            with_variants[:, :, None],
            in_tagged_to_in_tagged,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("0CN", "0IO", "0CN"))
        new_state = jnp.squeeze(new_state, (0, 2))
      else:
        new_state = jnp.einsum("vi,vij->j", with_variants,
                               in_tagged_to_in_tagged)

      return state_slice - new_state

    # Run the solver for `steps-1` iterations, since we have already simulated
    # the initial transition.
    return linear_solvers.richardson_solve(
        i_minus_q, initial_state_slice, iterations=steps - 1)

  visit_counts = jax.vmap(solve_one, (1, 0))(per_itn_variant_weights,
                                             initial_state)
  visit_counts = visit_counts.reshape([
      num_nodes,
      num_fsm_states,
      num_input_tagged_nodes,
  ])

  # Prepare to extract special actions.
  with_variants = jnp.einsum("vni,nsi->vnis", per_itn_variant_weights,
                             visit_counts)

  # Extract FINISH action.
  finish_index = builder.special_actions.index(SpecialActions.FINISH)
  finish_from_initial = jnp.einsum(
      "vnn,ns,vns->n", variant_weights, start_machine_states,
      transition_matrix.initial_to_special[:, :, :, finish_index])
  finish_from_in_tagged = jnp.einsum(
      "vnis,vis->ni", with_variants,
      transition_matrix.in_tagged_to_special[:, :, :, finish_index])
  finish_probs = jax.ops.index_add(
      jnp.diag(finish_from_initial),
      jax.ops.index[:, in_tagged_idxs],
      finish_from_in_tagged,
  )

  if SpecialActions.BACKTRACK in builder.special_actions:
    # Normalize FINISH by (FINISH + FAIL + backtrack_fails_prob * BACKTRACK)
    # so that we only consider paths that don't backtrack.
    weight_per_action = {
        SpecialActions.FINISH: 1.,
        SpecialActions.BACKTRACK: backtrack_fails_prob,
        SpecialActions.FAIL: 1.,
    }
    action_weights = jnp.array(
        [weight_per_action[action] for action in builder.special_actions])
    # Denominator of the normalization is the sum of this across all states.
    denominator_from_initial = jnp.einsum("vnn,ns,vnsa,a->n", variant_weights,
                                          start_machine_states,
                                          transition_matrix.initial_to_special,
                                          action_weights)
    denominator_from_in_tagged = jnp.einsum(
        "vnis,visa,a->n", with_variants, transition_matrix.in_tagged_to_special,
        action_weights)
    denominator_total = denominator_from_initial + denominator_from_in_tagged
    finish_probs = finish_probs / denominator_total[:, None]

  return finish_probs


def unroll_chain_steps(builder,
                       transition_matrix,
                       variant_weights,
                       start_machine_state,
                       node_index,
                       steps):
  """Compute automaton steps over time, for debugging and visualization.

  Note that index 0 of the returned "in_tagged_*" arrays represents the state
  of the automaton after taking 1 step.

  Args:
    builder: The automaton builder associated with this Markov chain.
    transition_matrix: The per-variant transition matrix for this Markov chain.
    variant_weights: Weights to assign to each routing variant for each node, as
      a <float[num_nodes, num_variants]> array (which should sum to 1 across the
      last axis).
    start_machine_state: Initial machine state distribution for the starting
      nodes, as a <float[num_fsm_states]> array (which should sum to 1).
    node_index: Initial node index where the solve should start, as a int32
      scalar.
    steps: How many steps to unroll.

  Returns:
    Dictionary of:
      - "initial_special": <float[num_special_actions]> array
        containing the probability of doing a special action at the initial
        state, instead of a normal transition.
      - "in_tagged_states": <float[steps, num_in_tagged_nodes,
                                   num_fsm_states]> array, containing the
        probability of being at each in-tagged state at each iteration.
      - "in_tagged_special": <float[steps, num_in_tagged_nodes,
                                    num_special_actions]> array,
        containing the cumulative probability of having taken each special
        action at each in-tagged state (effectively treating each special action
        at each node as a distinct absorbing state).
  """
  if steps < 1:
    raise ValueError("Must have at least 1 iteration.")
  # Initial step: compute initial transition and special actions.
  start_node_variant_weights = variant_weights[node_index]
  initial_to_in_tagged_slice = transition_matrix.initial_to_in_tagged[:,
                                                                      node_index]
  initial_to_special_slice = transition_matrix.initial_to_special[:, node_index]

  initial_step_state = jnp.einsum("v,s,vsjt->jt", start_node_variant_weights,
                                  start_machine_state,
                                  initial_to_in_tagged_slice)
  initial_special = jnp.einsum("v,s,vsa->a", start_node_variant_weights,
                               start_machine_state, initial_to_special_slice)

  # Loop over `steps` steps.
  def step(carry, _):
    state, special = carry
    per_itn_variants = variant_weights[transition_matrix.in_tagged_node_indices]
    with_variants = jnp.einsum("iv,is->vis", per_itn_variants, state)
    next_state = jnp.einsum("vis,visjt->jt", with_variants,
                            transition_matrix.in_tagged_to_in_tagged)
    delta_special = jnp.einsum("vis,visa->ia", with_variants,
                               transition_matrix.in_tagged_to_special)
    next_special = special + delta_special
    return (next_state, next_special), (next_state, next_special)

  num_input_tagged_nodes = initial_step_state.shape[0]
  initial_step_special = jnp.zeros(
      [num_input_tagged_nodes,
       len(builder.special_actions)])
  initial_carry = (initial_step_state, initial_step_special)
  _, (scan_states, scan_specials) = jax.lax.scan(
      step, initial_carry, None, length=steps - 1)

  in_tagged_states = jnp.concatenate([initial_step_state[None], scan_states],
                                     axis=0)
  in_tagged_special = jnp.concatenate(
      [initial_step_special[None], scan_specials], axis=0)
  return {
      "initial_special": initial_special,
      "in_tagged_states": in_tagged_states,
      "in_tagged_special": in_tagged_special,
  }


def aggregate_unrolled_per_node(
    unrolled, node_index,
    transition_matrix,
    graph_metadata):
  """Aggregate unrolled chain information to have per-node values.

  Transforms the output of unroll_chain_steps into a form where we have one
  value per node, instead of one value per (node, in edge, state) tuple.
  Useful for visualization. Also concatenates the special initial state into
  the result.

  Args:
    unrolled: Output of unroll_chain_steps.
    node_index: Initial node index where the unrolled steps started.
    transition_matrix: Transition matrix for the graph.
    graph_metadata: Metadata about the graph.

  Returns:
    Dictionary of:
      - "at_node": <float[steps+1, num_nodes]> array containing the
        probability of being at each node at each step (including the initial
        node as step 0).
      - "special": <float[steps+1, num_nodes, num_special_actions]> array
        containing the cumulative probability of having taken each special
        action at each node.
  """
  initial_special = unrolled["initial_special"]
  in_tagged_states = unrolled["in_tagged_states"]
  in_tagged_special = unrolled["in_tagged_special"]
  steps = in_tagged_states.shape[0]
  num_special_actions = in_tagged_special.shape[-1]
  # Combine states and special actions across nodes
  per_node_states = jax.ops.index_add(
      jnp.zeros([steps, graph_metadata.num_nodes]),
      jax.ops.index[:, transition_matrix.in_tagged_node_indices],
      jnp.sum(in_tagged_states, axis=2))
  per_node_special = jax.ops.index_add(
      jnp.zeros([steps, graph_metadata.num_nodes, num_special_actions]),
      jax.ops.index[:, transition_matrix.in_tagged_node_indices],
      in_tagged_special)
  # Add in initial special actions
  per_node_special = jax.ops.index_add(per_node_special,
                                       jax.ops.index[:, node_index],
                                       initial_special[None, :])
  # Concatenate the initial state before the first transition
  initial_node_state = jax.ops.index_add(
      jnp.zeros([1, graph_metadata.num_nodes]), jax.ops.index[0, node_index],
      1.)

  at_node = jnp.concatenate([initial_node_state, per_node_states], axis=0)
  special = jnp.pad(per_node_special, [(1, 0), (0, 0), (0, 0)], "constant")

  return {"at_node": at_node, "special": special}
