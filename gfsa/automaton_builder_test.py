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

# Lint as: python3
"""Tests for gfsa.automaton."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
from gfsa import automaton_builder
from gfsa import graph_types


class AutomatonTest(parameterized.TestCase):

  def build_simple_schema(self):
    return {
        graph_types.NodeType("a"):
            graph_types.NodeSchema(
                in_edges=[
                    graph_types.InEdgeType("ai_0"),
                    graph_types.InEdgeType("ai_1")
                ],
                out_edges=[graph_types.OutEdgeType("ao_0")]),
        graph_types.NodeType("b"):
            graph_types.NodeSchema(
                in_edges=[graph_types.InEdgeType("bi_0")],
                out_edges=[
                    graph_types.OutEdgeType("bo_0"),
                    graph_types.OutEdgeType("bo_1")
                ]),
    }

  def build_loop_graph(self):
    """Helper method to build this complex graph, to test graph encodings.

      ┌───────<──┐  ┌───────<─────────<─────┐
      │          │  │                       │
      │    [ao_0]│  ↓[ai_0]                 │
      │          (a0)                       │
      │    [ai_1]↑  │[ao_1]                 │
      │          │  │                       │
      │    [ao_0]│  ↓[ai_0]                 │
      ↓          (a1)                       ↑
      │    [ai_1]↑  │[ao_1]                 │
      │          │  │                       │
      │    [bo_0]│  ↓[bi_0]                 │
      │          ╭──╮───────>[bo_2]────┐    │
      │          │b0│───────>[bo_2]──┐ │    │
      │          │  │<──[bi_0]─────<─┘ │    │
      │          ╰──╯<──[bi_0]─────<─┐ │    │
      │    [bi_0]↑  │[bo_1]          │ │    ↑
      │          │  │                │ │    │
      │    [bo_0]│  ↓[bi_0]          │ │    │
      │          ╭──╮───────>[bo_2]──┘ │    │
      │          │b1│───────>[bo_2]──┐ │    │
      ↓          │  │<──[bi_0]─────<─┘ │    │
      │          ╰──╯<──[bi_0]─────<───┘    │
      │    [bi_0]↑  │[bo_1]                 │
      │          │  │                       │
      └───────>──┘  └───────>─────────>─────┘

    Returns:
      Tuple (schema, graph) for the above structure.
    """
    a = graph_types.NodeType("a")
    b = graph_types.NodeType("b")

    ai_0 = graph_types.InEdgeType("ai_0")
    ai_1 = graph_types.InEdgeType("ai_1")
    bi_0 = graph_types.InEdgeType("bi_0")
    bi_0 = graph_types.InEdgeType("bi_0")

    ao_0 = graph_types.OutEdgeType("ao_0")
    ao_1 = graph_types.OutEdgeType("ao_1")
    bo_0 = graph_types.OutEdgeType("bo_0")
    bo_1 = graph_types.OutEdgeType("bo_1")
    bo_2 = graph_types.OutEdgeType("bo_2")

    a0 = graph_types.NodeId("a0")
    a1 = graph_types.NodeId("a1")
    b0 = graph_types.NodeId("b0")
    b1 = graph_types.NodeId("b1")

    schema = {
        a:
            graph_types.NodeSchema(
                in_edges=[ai_0, ai_1], out_edges=[ao_0, ao_1]),
        b:
            graph_types.NodeSchema(
                in_edges=[bi_0], out_edges=[bo_0, bo_1, bo_2]),
    }
    test_graph = {
        a0:
            graph_types.GraphNode(
                a, {
                    ao_0: [graph_types.InputTaggedNode(b1, bi_0)],
                    ao_1: [graph_types.InputTaggedNode(a1, ai_0)]
                }),
        a1:
            graph_types.GraphNode(
                a, {
                    ao_0: [graph_types.InputTaggedNode(a0, ai_1)],
                    ao_1: [graph_types.InputTaggedNode(b0, bi_0)]
                }),
        b0:
            graph_types.GraphNode(
                b, {
                    bo_0: [graph_types.InputTaggedNode(a1, ai_1)],
                    bo_1: [graph_types.InputTaggedNode(b1, bi_0)],
                    bo_2: [
                        graph_types.InputTaggedNode(b0, bi_0),
                        graph_types.InputTaggedNode(b1, bi_0)
                    ]
                }),
        b1:
            graph_types.GraphNode(
                b, {
                    bo_0: [graph_types.InputTaggedNode(b0, bi_0)],
                    bo_1: [graph_types.InputTaggedNode(a0, ai_0)],
                    bo_2: [
                        graph_types.InputTaggedNode(b0, bi_0),
                        graph_types.InputTaggedNode(b1, bi_0)
                    ]
                }),
    }
    return schema, test_graph

  def test_constructor_actions_nodes_routes(self):
    builder = automaton_builder.AutomatonBuilder(
        self.build_simple_schema(), with_backtrack=False, with_fail=True)

    self.assertEqual(
        set(builder.special_actions), {
            automaton_builder.SpecialActions.FINISH,
            automaton_builder.SpecialActions.FAIL
        })

    self.assertEqual(
        set(builder.node_types),
        {graph_types.NodeType("a"),
         graph_types.NodeType("b")})

    self.assertEqual(
        set(builder.in_route_types), {
            automaton_builder.InRouteType(
                graph_types.NodeType("a"), automaton_builder.SOURCE_INITIAL),
            automaton_builder.InRouteType(
                graph_types.NodeType("a"), graph_types.InEdgeType("ai_0")),
            automaton_builder.InRouteType(
                graph_types.NodeType("a"), graph_types.InEdgeType("ai_1")),
            automaton_builder.InRouteType(
                graph_types.NodeType("b"), automaton_builder.SOURCE_INITIAL),
            automaton_builder.InRouteType(
                graph_types.NodeType("b"), graph_types.InEdgeType("bi_0")),
        })

    self.assertEqual(
        set(builder.in_out_route_types), {
            automaton_builder.InOutRouteType(
                graph_types.NodeType("a"), automaton_builder.SOURCE_INITIAL,
                graph_types.OutEdgeType("ao_0")),
            automaton_builder.InOutRouteType(
                graph_types.NodeType("a"), graph_types.InEdgeType("ai_0"),
                graph_types.OutEdgeType("ao_0")),
            automaton_builder.InOutRouteType(
                graph_types.NodeType("a"), graph_types.InEdgeType("ai_1"),
                graph_types.OutEdgeType("ao_0")),
            automaton_builder.InOutRouteType(
                graph_types.NodeType("b"), automaton_builder.SOURCE_INITIAL,
                graph_types.OutEdgeType("bo_0")),
            automaton_builder.InOutRouteType(
                graph_types.NodeType("b"), automaton_builder.SOURCE_INITIAL,
                graph_types.OutEdgeType("bo_1")),
            automaton_builder.InOutRouteType(
                graph_types.NodeType("b"), graph_types.InEdgeType("bi_0"),
                graph_types.OutEdgeType("bo_0")),
            automaton_builder.InOutRouteType(
                graph_types.NodeType("b"), graph_types.InEdgeType("bi_0"),
                graph_types.OutEdgeType("bo_1")),
        })

  def test_constructor_inverse_mappings(self):
    builder = automaton_builder.AutomatonBuilder(self.build_simple_schema())

    # Mappings should be inverses of the corresponding lists
    for i, node_type in enumerate(builder.node_types):
      self.assertEqual(builder.node_type_to_index[node_type], i)

    for i, in_route_type in enumerate(builder.in_route_types):
      self.assertEqual(builder.in_route_type_to_index[in_route_type], i)

    for i, in_out_route_type in enumerate(builder.in_out_route_types):
      self.assertEqual(builder.in_out_route_type_to_index[in_out_route_type], i)

  def test_constructor_information_removing_mappings(self):
    builder = automaton_builder.AutomatonBuilder(self.build_simple_schema())

    # Check consistency of information-removing mappings with the
    # corresponding pairs of lists.
    for in_out_route_type in builder.in_out_route_types:
      in_route_type = automaton_builder.InRouteType(in_out_route_type.node_type,
                                                    in_out_route_type.in_edge)
      self.assertEqual(
          builder.in_out_route_to_in_route[
              builder.in_out_route_type_to_index[in_out_route_type]],
          builder.in_route_type_to_index[in_route_type])

    for in_route_type in builder.in_route_types:
      node_type = graph_types.NodeType(in_route_type.node_type)
      self.assertEqual(
          builder.in_route_to_node_type[
              builder.in_route_type_to_index[in_route_type]],
          builder.node_type_to_index[node_type])

    for in_out_route_type in builder.in_out_route_types:
      in_route_type = automaton_builder.InRouteType(in_out_route_type.node_type,
                                                    in_out_route_type.in_edge)
      self.assertEqual(
          builder.in_out_route_to_in_route[
              builder.in_out_route_type_to_index[in_out_route_type]],
          builder.in_route_type_to_index[in_route_type])

  @parameterized.named_parameters([
      {
          "testcase_name": "sum",
          "reduction": "sum"
      },
      {
          "testcase_name": "max",
          "reduction": "max"
      },
      {
          "testcase_name": "softmax",
          "reduction": "softmax"
      },
  ])
  def test_routing_reduce_correct(self, reduction):
    """Compare JAX implementations to a (slow but correct) iterative one."""
    n_variants = 2
    n_states = 4

    def make_range_shaped(shape):
      return np.arange(np.prod(shape)).reshape(shape).astype("float32")

    schema = self.build_simple_schema()
    builder = automaton_builder.AutomatonBuilder(schema)
    routing_params = automaton_builder.RoutingParams(
        move=make_range_shaped([
            n_variants,
            len(builder.in_out_route_types),
            n_states,
            n_states,
        ]),
        special=make_range_shaped([
            n_variants,
            len(builder.in_route_types),
            n_states,
            len(builder.special_actions),
        ]),
    )

    # Compute aggregates with JAX
    if reduction == "softmax":
      routing_aggregates = builder.routing_softmax(routing_params)
    else:
      routing_aggregates = builder.routing_reduce(
          routing_params, reduction=reduction)
      routing_aggregates = jax.tree_multimap(
          lambda s, p: np.array(jnp.broadcast_to(s, p.shape)),
          routing_aggregates, routing_params)

    # Manual looping aggregates
    for variant in range(n_variants):
      for current_state in range(n_states):
        for in_route_type in builder.in_route_types:
          # Compute aggregates
          distn_vals = []
          iroute_idx = builder.in_route_type_to_index[in_route_type]
          for out_edge_type in schema[in_route_type.node_type].out_edges:
            ioroute_idx = builder.in_out_route_type_to_index[
                automaton_builder.InOutRouteType(in_route_type.node_type,
                                                 in_route_type.in_edge,
                                                 out_edge_type)]
            for next_state in range(n_states):
              distn_vals.append(routing_params.move[variant, ioroute_idx,
                                                    current_state, next_state])

          for action_idx in range(len(builder.special_actions)):
            distn_vals.append(routing_params.special[variant, iroute_idx,
                                                     current_state, action_idx])

          if reduction == "sum":
            distn_aggregate = [sum(distn_vals)] * len(distn_vals)
          elif reduction == "max":
            distn_aggregate = [max(distn_vals)] * len(distn_vals)
          elif reduction == "softmax":
            distn_aggregate = list(jax.nn.softmax(jnp.array(distn_vals)))
          else:
            raise ValueError(f"Invalid reduction {reduction}")

          i = 0
          # Check them with the JAX version
          for out_edge_type in schema[in_route_type.node_type].out_edges:
            ioroute_idx = builder.in_out_route_type_to_index[
                automaton_builder.InOutRouteType(in_route_type.node_type,
                                                 in_route_type.in_edge,
                                                 out_edge_type)]
            for next_state in range(n_states):
              np.testing.assert_allclose(
                  routing_aggregates.move[variant, ioroute_idx, current_state,
                                          next_state],
                  distn_aggregate[i],
                  rtol=1e-6)
              i += 1

          for action_idx in range(len(builder.special_actions)):
            np.testing.assert_allclose(
                routing_aggregates.special[variant, iroute_idx, current_state,
                                           action_idx],
                distn_aggregate[i],
                rtol=1e-6)
            i += 1

  def test_initial_routing_params_noiseless(self):
    schema = self.build_simple_schema()
    builder = automaton_builder.AutomatonBuilder(schema)
    routing_params = builder.initialize_routing_params(
        key=None,
        num_fsm_states=3,
        num_variants=2,
        state_change_prob=0.2,
        move_prob=0.9,
        noise_factor=0)

    outgoing_count = np.array([
        len(schema[in_out_route.node_type].out_edges)
        for in_out_route in builder.in_out_route_types
    ])[None, :, None]

    all_same_state_moves = routing_params.move[:, :, np.arange(3), np.arange(3)]
    expected = np.broadcast_to(0.9 * 0.8 / outgoing_count,
                               all_same_state_moves.shape)
    np.testing.assert_allclose(all_same_state_moves, expected)

    state_1 = []
    state_2 = []
    for i in range(3):
      for j in range(3):
        if i != j:
          state_1.append(i)
          state_2.append(j)

    all_different_state_moves = routing_params.move[:, :, state_1, state_2]
    expected = np.broadcast_to(0.9 * 0.2 / (2 * outgoing_count),
                               all_different_state_moves.shape)
    np.testing.assert_allclose(all_different_state_moves, expected)

    np.testing.assert_allclose(routing_params.special, 0.1 / 3)

  def test_initial_routing_params_with_noise(self):
    builder = automaton_builder.AutomatonBuilder(self.build_simple_schema())

    # Small amounts of noise shouldn't change parameters much
    initializer_kwargs = dict(
        num_fsm_states=3, num_variants=2, state_change_prob=0.2, move_prob=0.9)
    noiseless_params = builder.initialize_routing_params(
        key=None, noise_factor=0, **initializer_kwargs)
    eps_noise_params = builder.initialize_routing_params(
        key=jax.random.PRNGKey(1234), noise_factor=1e-6, **initializer_kwargs)

    np.testing.assert_allclose(
        noiseless_params.move, eps_noise_params.move, rtol=0.02)
    np.testing.assert_allclose(
        noiseless_params.special, eps_noise_params.special, rtol=0.02)

    # Even with more noise, should still be normalized
    noisy_params = builder.initialize_routing_params(
        key=jax.random.PRNGKey(1234), noise_factor=0.8, **initializer_kwargs)
    noisy_sums = builder.routing_reduce(noisy_params, "sum")

    np.testing.assert_allclose(noisy_sums.move, 1.0, rtol=1e-6)
    np.testing.assert_allclose(noisy_sums.special, 1.0, rtol=1e-6)

  def test_routing_gates_to_probs(self):
    builder = automaton_builder.AutomatonBuilder(self.build_simple_schema())

    # [variants, in_out_routes, fsm_states, fsm_states]
    # [variants, in_routes, fsm_states]
    move_gates = np.full([3, len(builder.in_out_route_types), 2, 2], 0.5)
    accept_gates = np.full([3, len(builder.in_route_types), 2], 0.5)
    backtrack_gates = np.full([3, len(builder.in_route_types), 2], 0.5)

    # Set one distribution to sum to more than 1.
    idx_d1_move1 = builder.in_out_route_type_to_index[
        automaton_builder.InOutRouteType(
            graph_types.NodeType("b"), graph_types.InEdgeType("bi_0"),
            graph_types.OutEdgeType("bo_0"))]
    move_gates[0, idx_d1_move1, 0, :] = [.2, .3]
    idx_d1_move2 = builder.in_out_route_type_to_index[
        automaton_builder.InOutRouteType(
            graph_types.NodeType("b"), graph_types.InEdgeType("bi_0"),
            graph_types.OutEdgeType("bo_1"))]
    move_gates[0, idx_d1_move2, 0, :] = [.4, .5]
    idx_d1_special = builder.in_route_type_to_index[
        automaton_builder.InRouteType(
            graph_types.NodeType("b"), graph_types.InEdgeType("bi_0"))]
    accept_gates[0, idx_d1_special, 0] = .6
    backtrack_gates[0, idx_d1_special, 0] = .3

    # Set another to sum to less than 1.
    idx_d2_move = builder.in_out_route_type_to_index[
        automaton_builder.InOutRouteType(
            graph_types.NodeType("a"), graph_types.InEdgeType("ai_0"),
            graph_types.OutEdgeType("ao_0"))]
    move_gates[2, idx_d2_move, 1, :] = [.1, .2]
    idx_d2_special = builder.in_route_type_to_index[
        automaton_builder.InRouteType(
            graph_types.NodeType("a"), graph_types.InEdgeType("ai_0"))]
    accept_gates[2, idx_d2_special, 1] = .3
    backtrack_gates[2, idx_d2_special, 1] = .75

    routing_gates = automaton_builder.RoutingGateParams(
        move_gates=jax.scipy.special.logit(move_gates),
        accept_gates=jax.scipy.special.logit(accept_gates),
        backtrack_gates=jax.scipy.special.logit(backtrack_gates))
    routing_probs = builder.routing_gates_to_probs(routing_gates)

    # Check probabilities for first distribution: should divide evenly.
    np.testing.assert_allclose(routing_probs.move[0, idx_d1_move1, 0, :],
                               np.array([.2, .3]) / 2.0)
    np.testing.assert_allclose(routing_probs.move[0, idx_d1_move2, 0, :],
                               np.array([.4, .5]) / 2.0)
    np.testing.assert_allclose(routing_probs.special[0, idx_d1_special, 0, :],
                               np.array([.6, 0, 0]) / 2.0)

    # Check probabilities for second distribution: should assign remainder to
    # backtrack and fail.
    np.testing.assert_allclose(routing_probs.move[2, idx_d2_move, 1, :],
                               np.array([.1, .2]))
    np.testing.assert_allclose(routing_probs.special[2, idx_d2_special, 1, :],
                               np.array([.3, .3, .1]))

  def initialize_routing_gates(self):
    """Just make sure that we can initialize routing gates."""
    builder = automaton_builder.AutomatonBuilder(self.build_simple_schema())

    # Noiseless
    noiseless_gates = builder.initialize_routing_gates(
        key=None, logistic_noise=0, num_fsm_states=3, num_variants=2)
    self.assertEqual(noiseless_gates.move_gates.shape,
                     (2, len(builder.in_out_route_types), 3, 3))
    self.assertEqual(noiseless_gates.accept_gates.shape,
                     (2, len(builder.in_route_types), 3))
    self.assertEqual(noiseless_gates.backtracKkgates.shape,
                     (2, len(builder.in_route_types), 3))

    # Perturbed
    noisy_gates = builder.initialize_routing_gates(
        key=jax.random.PRNGKey(0),
        logistic_noise=0.2,
        num_fsm_states=3,
        num_variants=2)
    self.assertEqual(noisy_gates.move_gates.shape,
                     (2, len(builder.in_out_route_types), 3, 3))
    self.assertEqual(noisy_gates.accept_gates.shape,
                     (2, len(builder.in_route_types), 3))
    self.assertEqual(noisy_gates.backtracKkgates.shape,
                     (2, len(builder.in_route_types), 3))

  def test_graph_encoding_size(self):
    """Test the size of the encoded graph."""
    schema, test_graph = self.build_loop_graph()
    builder = automaton_builder.AutomatonBuilder(schema)
    encoded_graph, graph_meta = builder.encode_graph(test_graph, as_jax=False)

    # Graph metadata should match our graph's actual size
    self.assertEqual(graph_meta.num_nodes, 4)
    self.assertEqual(graph_meta.num_input_tagged_nodes, 6)

    # Nonzero entries should match the number of possible transitions
    # Initial transition counts each NODE once, so each A node has 2 and each
    # B node has 4 outgoing transitions
    self.assertEqual(encoded_graph.initial_to_in_tagged.values.shape[0], 12)

    # Normal transitions count each input-tagged node once, so each A node has
    # 2*2=4 and each B node has 1*4=4 outgoing transitions
    self.assertEqual(encoded_graph.in_tagged_to_in_tagged.values.shape[0], 16)

  def test_transition_all_ones(self):
    """Test the transition matrix of an all-ones routing parameter vector."""
    schema, test_graph = self.build_loop_graph()
    builder = automaton_builder.AutomatonBuilder(schema)
    encoded_graph, graph_meta = builder.encode_graph(test_graph, as_jax=False)

    # The transition matrix for an all-ones routing params should be a
    # (weighted) directed adjacency matrix. We use 3 variants, 2 states.
    ones_routing_params = automaton_builder.RoutingParams(
        move=jnp.ones([3, 12, 2, 2]), special=jnp.ones([3, 5, 2, 3]))

    ones_transition_matrix = builder.build_transition_matrix(
        ones_routing_params,
        encoded_graph,
        graph_meta,
    ).concatenated_transitions()
    self.assertEqual(ones_transition_matrix.shape, (3, 4 + 6, 2, 6 * 2 + 3))

    # pyformat: disable
    # pylint: disable=bad-continuation,bad-whitespace,g-inline-comment-too-close
    expected = np.array([
        #  a0i0    a0i1    a1i0    a1i1    b0i0    b1i0    specials  < next
      [ #|------|-------|-------|-------|-------|-------| |--------| current V
        [[ 0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  1],  # ┬ a0
         [ 0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  1]], # |
        [[ 0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1],  # | a1
         [ 0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1]], # |
        [[ 0,  0,  0,  0,  0,  0,  1,  1,0.5,0.5,1.5,1.5,  1,  1,  1],  # | b0
         [ 0,  0,  0,  0,  0,  0,  1,  1,0.5,0.5,1.5,1.5,  1,  1,  1]], # |
        [[ 1,  1,  0,  0,  0,  0,  0,  0,1.5,1.5,0.5,0.5,  1,  1,  1],  # | b1
         [ 1,  1,  0,  0,  0,  0,  0,  0,1.5,1.5,0.5,0.5,  1,  1,  1]], # ┴
        [[ 0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  1],  # ┬ a0i0
         [ 0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  1]], # |
        [[ 0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  1],  # | a0i1
         [ 0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  1]], # |
        [[ 0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1],  # | a1i0
         [ 0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1]], # |
        [[ 0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1],  # | a1i1
         [ 0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1]], # |
        [[ 0,  0,  0,  0,  0,  0,  1,  1,0.5,0.5,1.5,1.5,  1,  1,  1],  # | b0i0
         [ 0,  0,  0,  0,  0,  0,  1,  1,0.5,0.5,1.5,1.5,  1,  1,  1]], # |
        [[ 1,  1,  0,  0,  0,  0,  0,  0,1.5,1.5,0.5,0.5,  1,  1,  1],  # | b0i1
         [ 1,  1,  0,  0,  0,  0,  0,  0,1.5,1.5,0.5,0.5,  1,  1,  1]], # ┴
      ]
    ] * 3)
    # pyformat: enable
    # pylint: enable=bad-continuation,bad-whitespace,g-inline-comment-too-close
    np.testing.assert_allclose(ones_transition_matrix, expected)

  def test_transition_sentinel_integers(self):
    """Test that the transition matrix puts each element in the right place."""
    schema, test_graph = self.build_loop_graph()
    builder = automaton_builder.AutomatonBuilder(schema)
    encoded_graph, graph_meta = builder.encode_graph(test_graph, as_jax=False)

    # Apply to some sentinel integers to check correct indexing (with only one
    # variant and state, since indexing doesn't use those)
    # Each integer is of the form XYZ where
    #   X = {a:1, b:2}[node_type]
    #   Y = {initial:9, i0:0, i1:1}[in_edge]
    #   Z = {o0:0, o1:1, o2:2, finish:3, backtrack:4, fail:5}[action]
    sentinel_routing_params = automaton_builder.RoutingParams(
        move=jnp.array([[100, 101, 110, 111, 190, 191],
                        [200, 201, 202, 290, 291, 292]]).reshape((1, 12, 1, 1)),
        special=jnp.array([
            [103, 104, 105],
            [113, 114, 115],
            [193, 194, 195],
            [203, 204, 205],
            [293, 294, 295],
        ]).reshape((1, 5, 1, 3)))

    range_transition_matrix = builder.build_transition_matrix(
        sentinel_routing_params,
        encoded_graph,
        graph_meta,
    ).concatenated_transitions()
    self.assertEqual(range_transition_matrix.shape, (1, 4 + 6, 1, 6 * 1 + 3))

    # pyformat: disable
    # pylint: disable=bad-continuation,bad-whitespace,g-inline-comment-too-close
    expected = np.array([
        #  a0i0 a0i1 a1i0 a1i1       b0i0       b1i0   specials    < next
      [ #   |    |    |    |          |          |    |---------|   current V
        [[  0,   0, 191,   0,         0,       190, 193, 194, 195]],   # ┬ a0
        [[  0, 190,   0,   0,       191,         0, 193, 194, 195]],   # | a1
        [[  0,   0,   0, 290,     292/2, 291+292/2, 293, 294, 295]],   # | b0
        [[291,   0,   0,   0, 290+292/2,     292/2, 293, 294, 295]],   # ┴ b1
        [[  0,   0, 101,   0,         0,       100, 103, 104, 105]],   # ┬ a0i0
        [[  0,   0, 111,   0,         0,       110, 113, 114, 115]],   # | a0i1
        [[  0, 100,   0,   0,       101,         0, 103, 104, 105]],   # | a1i0
        [[  0, 110,   0,   0,       111,         0, 113, 114, 115]],   # | a1i1
        [[  0,   0,   0, 200,     202/2, 201+202/2, 203, 204, 205]],   # | b0i0
        [[201,   0,   0,   0, 200+202/2,     202/2, 203, 204, 205]],   # ┴ b0i1
      ]
    ])
    # pyformat: enable
    # pylint: enable=bad-continuation,bad-whitespace,g-inline-comment-too-close
    np.testing.assert_allclose(range_transition_matrix, expected)

  def build_doubly_linked_list_graph(self, length):
    """Helper method to build a doubly-linked-list graph and schema."""
    schema = {
        graph_types.NodeType("node"):
            graph_types.NodeSchema(
                in_edges=[
                    graph_types.InEdgeType("next_in"),
                    graph_types.InEdgeType("prev_in"),
                ],
                out_edges=[
                    graph_types.OutEdgeType("next_out"),
                    graph_types.OutEdgeType("prev_out"),
                ])
    }
    graph = {}
    for i in range(length):
      graph[graph_types.NodeId(str(i))] = graph_types.GraphNode(
          graph_types.NodeType("node"), {
              graph_types.OutEdgeType("next_out"): [
                  graph_types.InputTaggedNode(
                      node_id=graph_types.NodeId(str((i + 1) % length)),
                      in_edge=graph_types.InEdgeType("prev_in"))
              ],
              graph_types.OutEdgeType("prev_out"): [
                  graph_types.InputTaggedNode(
                      node_id=graph_types.NodeId(str((i - 1) % length)),
                      in_edge=graph_types.InEdgeType("next_in"))
              ]
          })
    return schema, graph

  def test_all_nodes_absorbing_solve(self):
    schema, graph = self.build_doubly_linked_list_graph(4)
    builder = automaton_builder.AutomatonBuilder(schema)
    enc_graph, enc_meta = builder.encode_graph(graph)

    # We set up the automaton with 5 variants and 2 states, but only use the
    # first state, to make sure that variants and states are interleaved
    # correctly.

    # Variant 0: move forward
    # Variant 1: move backward
    # Variant 2: finish
    # Variant 3: restart
    # Variant 4: fail
    variant_weights = jnp.array([
        # From node 0, go forward.
        [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [.7, 0, .3, 0, 0], [0, 0, 1, 0, 0]],
        # From node 1, go backward with small failure probabilities.
        [[0, 0.9, 0, 0, 0.1], [0, 0.9, 0, 0, 0.1], [.7, 0, .3, 0, 0],
         [0, 0, 1, 0, 0]],
        # Node 2 bounces around and ultimately accepts on node 0.
        [[0.9, 0, 0.1, 0, 0], [0, 1, 0, 0, 0], [0.5, 0.5, 0, 0, 0],
         [0, 1, 0, 0, 0]],
        # Node 3 immediately accepts, or restarts after 0 or 1 steps.
        [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
         [0, 0.1, 0.8, 0.1, 0]],
    ])
    routing_params = automaton_builder.RoutingParams(
        move=jnp.pad(
            jnp.array([
                [1., 0., 1., 0., 1., 0.],
                [0., 1., 0., 1., 0., 1.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
            ]).reshape([5, 6, 1, 1]), [(0, 0), (0, 0), (0, 1), (0, 1)]),
        special=jnp.pad(
            jnp.array([
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]],
                [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]],
                [[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]],
            ]).reshape([5, 3, 1, 3]), [(0, 0), (0, 0), (0, 1), (0, 0)]))
    tmat = builder.build_transition_matrix(routing_params, enc_graph, enc_meta)

    # Absorbing probs follow the paths described above.
    # Note that when starting at node 3, with probability 0.2 the automaton
    # tries to backtrack, but with probability 0.2 * 0.01 backtracking fails
    # (as specified by backtrack_fails_prob) and thus the total absorbing
    # probability is 0.8 / (0.8 + 0.2 * 0.01) = 0.997506
    expected_absorbing_probs = jnp.array([
        [0, 0, 0.3, 0.7],
        [0, 0, 0, 0.81],
        [1, 0, 0, 0],
        [0, 0, 0, 0.997506],
    ])
    absorbing_probs = automaton_builder.all_nodes_absorbing_solve(
        builder,
        tmat,
        variant_weights,
        jnp.pad(jnp.ones([4, 1]), [(0, 0), (0, 1)]),
        steps=1000,
        backtrack_fails_prob=0.01)

    jax.test_util.check_close(absorbing_probs, expected_absorbing_probs)

  def test_all_nodes_absorbing_solve_explicit_conv(self):
    schema, graph = self.build_doubly_linked_list_graph(4)
    builder = automaton_builder.AutomatonBuilder(schema)
    enc_graph, enc_meta = builder.encode_graph(graph)

    variant_weights = jax.random.dirichlet(
        jax.random.PRNGKey(0), jnp.ones((4, 4, 5)))
    routing_params = builder.initialize_routing_params(
        jax.random.PRNGKey(1), num_fsm_states=3, num_variants=5)
    start_states = jax.random.dirichlet(jax.random.PRNGKey(0), jnp.ones((4, 3)))

    # Confirm that the explicit conv doesn't change results.
    def go(routing_params, variant_weights, start_states, explicit_conv=True):
      tmat = builder.build_transition_matrix(routing_params, enc_graph,
                                             enc_meta)
      return automaton_builder.all_nodes_absorbing_solve(
          builder,
          tmat,
          variant_weights,
          start_states,
          steps=1000,
          backtrack_fails_prob=0.01,
          explicit_conv=explicit_conv)

    vals, vjpfun = jax.vjp(go, routing_params, variant_weights, start_states)
    unopt_vals, unopt_vjpfun = jax.vjp(
        functools.partial(go, explicit_conv=False), routing_params,
        variant_weights, start_states)

    jax.test_util.check_close(vals, unopt_vals)
    some_cotangent = jax.random.normal(jax.random.PRNGKey(0), vals.shape)
    jax.test_util.check_close(
        vjpfun(some_cotangent), unopt_vjpfun(some_cotangent))

  def test_unroll_and_aggregate(self):
    schema, graph = self.build_doubly_linked_list_graph(4)
    builder = automaton_builder.AutomatonBuilder(schema)
    enc_graph, enc_meta = builder.encode_graph(graph)

    # Three variants, but all the same, just to check shape consistency
    variant_weights = jnp.broadcast_to(jnp.array([0.7, 0.3, 0.]), [4, 3])

    # In state 0: keep moving in the current direction with prob 0.1, swap
    # directions and states with prob 0.9 (except init, which is a special case)
    # In state 1: take a special action
    routing_params = automaton_builder.RoutingParams(
        move=jnp.broadcast_to(
            jnp.array([
                # from next, to next
                [[0.0, 0.9], [0.0, 0.0]],
                # from next, to prev
                [[0.1, 0.0], [0.0, 0.0]],
                # from prev, to next
                [[0.1, 0.0], [0.0, 0.0]],
                # from next, to prev
                [[0.0, 0.9], [0.0, 0.0]],
                # from init, to next
                [[0.1, 0.0], [0.0, 0.0]],
                # from init, to prev
                [[0.1, 0.0], [0.0, 0.0]],
            ]),
            [3, 6, 2, 2]),
        special=jnp.broadcast_to(
            jnp.array([
                # from next
                [[0, 0, 0], [0.2, 0.3, 0.5]],
                # from prev
                [[0, 0, 0], [0.5, 0.2, 0.3]],
                # from init
                [[0.1, 0.3, 0.4], [0.0, 0.0, 1.0]],
            ]),
            [3, 3, 2, 3]))
    tmat = builder.build_transition_matrix(routing_params, enc_graph, enc_meta)

    unrolled = automaton_builder.unroll_chain_steps(
        builder,
        tmat,
        variant_weights,
        jnp.array([1., 0.]),
        node_index=0,
        steps=6)

    expected_initial_special = np.array([0.1, 0.3, 0.4])
    np.testing.assert_allclose(unrolled["initial_special"],
                               expected_initial_special)

    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_in_tagged_states = np.array([
        # In-tagged node key:
        #  0 from 1,     0 from 3,     1 from 2,     1 from 0,
        #  2 from 3,     2 from 1,     3 from 0,     3 from 2
        # State key: [prob of being in state 0, prob of being in state 1]
        # -- First step from initial --
        [[0, 0], [0, 0], [0, 0], [1e-1, 0],
         [0, 0], [0, 0], [1e-1, 0], [0, 0]],
        # -- Second step --
        [[0, 9e-2], [0, 9e-2], [0, 0], [0, 0],
         [1e-2, 0], [1e-2, 0], [0, 0], [0, 0]],
        # -----------------
        [[0, 0], [0, 0], [1e-3, 9e-3], [0, 0],
         [0, 0], [0, 0], [0, 0], [1e-3, 9e-3]],
        # -----------------
        [[1e-4, 0], [1e-4, 0], [0, 0], [0, 0],
         [0, 9e-4], [0, 9e-4], [0, 0], [0, 0]],
        # -----------------
        [[0, 0], [0, 0], [0, 0], [1e-5, 9e-5],
         [0, 0], [0, 0], [1e-5, 9e-5], [0, 0]],
        # -----------------
        [[0, 9e-6], [0, 9e-6], [0, 0], [0, 0],
         [1e-6, 0], [1e-6, 0], [0, 0], [0, 0]],
    ])
    np.testing.assert_allclose(
        unrolled["in_tagged_states"], expected_in_tagged_states, atol=1e-8)

    expected_in_tagged_special = np.array([
        # In-tagged node key:
        #  0 from 1,                 0 from 3
        #  1 from 2,                 1 from 0,
        #  2 from 3,                 2 from 1
        #  3 from 0,                 3 from 2
        # Action key: [finish, backtrack, fail] (cumulative)
        # -- First step from initial --
        # (no special actions because we just left the initial node)
        [[0, 0, 0]]*8,
        # -- Second step --
        # (no special actions yet because everything was in state 0)
        [[0, 0, 0]]*8,
        # -----------------
        [[1.8e-2, 2.7e-2, 4.5e-2], [4.5e-2, 1.8e-2, 2.7e-2],
         [0, 0, 0],                [0, 0, 0],
         [0, 0, 0],                [0, 0, 0],
         [0, 0, 0],                [0, 0, 0]],
        # -----------------
        [[1.8e-2, 2.7e-2, 4.5e-2], [4.5e-2, 1.8e-2, 2.7e-2],
         [1.8e-3, 2.7e-3, 4.5e-3], [0, 0, 0],
         [0, 0, 0],                [0, 0, 0],
         [0, 0, 0],                [4.5e-3, 1.8e-3, 2.7e-3]],
        # -----------------
        [[1.8e-2, 2.7e-2, 4.5e-2], [4.5e-2, 1.8e-2, 2.7e-2],
         [1.8e-3, 2.7e-3, 4.5e-3], [0, 0, 0],
         [1.8e-4, 2.7e-4, 4.5e-4], [4.5e-4, 1.8e-4, 2.7e-4],
         [0, 0, 0],                [4.5e-3, 1.8e-3, 2.7e-3]],
        # -----------------
        [[1.8e-2, 2.7e-2, 4.5e-2], [4.5e-2, 1.8e-2, 2.7e-2],
         [1.8e-3, 2.7e-3, 4.5e-3], [4.5e-5, 1.8e-5, 2.7e-5],
         [1.8e-4, 2.7e-4, 4.5e-4], [4.5e-4, 1.8e-4, 2.7e-4],
         [1.8e-5, 2.7e-5, 4.5e-5], [4.5e-3, 1.8e-3, 2.7e-3]],
    ])
    np.testing.assert_allclose(
        unrolled["in_tagged_special"], expected_in_tagged_special, atol=1e-8)
    # pyformat: enable
    # pylint: enable=bad-whitespace

    unrolled_combined = automaton_builder.aggregate_unrolled_per_node(
        unrolled, 0, 0, tmat, enc_meta)

    expected_unrolled = np.array([
        # "Zeroth" step: at initial node, no specials have happened yet
        [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        # The other steps are either the states from above, or sums of entries
        # in initial_special and in_tagged_special
        [[0, 0, 0.1, 0.3, 0.4], [0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0.1, 0, 0, 0, 0]],
        # -----------------
        [[0, 0.18, 0.1, 0.3, 0.4], [0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # -----------------
        [[0, 0, 0.163, 0.34500003, 0.472], [0.001, 0.009, 0, 0, 0],
         [0, 0, 0, 0, 0], [0.001, 0.009, 0, 0, 0]],
        # -----------------
        [[0.0002, 0, 0.163, 0.34500003, 0.472], [0, 0, 0.0018, 0.0027, 0.0045],
         [0, 0.0018, 0, 0, 0], [0, 0, 0.0045, 0.0018, 0.0027]],
        # -----------------
        [[0, 0, 0.163, 0.34500003, 0.472],
         [1e-05, 9e-05, 0.0018, 0.0027, 0.0045],
         [0, 0, 0.00063, 0.00045, 0.00072],
         [1e-05, 9e-05, 0.0045, 0.0018, 0.0027]],
        # -----------------
        [[0, 1.8e-05, 0.163, 0.34500003, 0.472],
         [0, 0, 0.001845, 0.002718, 0.004527],
         [2e-06, 0, 0.00063, 0.00045, 0.00072],
         [0, 0, 0.004518, 0.001827, 0.002745]],
    ])

    np.testing.assert_allclose(unrolled_combined, expected_unrolled, atol=1e-8)


if __name__ == "__main__":
  absltest.main()
