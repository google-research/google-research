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

"""Tests for gfsa.automaton_sampling."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
from gfsa import automaton_builder
from gfsa import automaton_sampling
from gfsa import graph_types
from gfsa import jax_util


class AutomatonSamplingTest(parameterized.TestCase):

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

  def test_one_node_particle_estimate_padding(self):
    schema, graph = self.build_doubly_linked_list_graph(4)
    builder = automaton_builder.AutomatonBuilder(schema)
    enc_graph, enc_meta = builder.encode_graph(graph)
    enc_graph_padded = automaton_builder.EncodedGraph(
        initial_to_in_tagged=enc_graph.initial_to_in_tagged.pad_nonzeros(64),
        initial_to_special=jax_util.pad_to(enc_graph.initial_to_special, 64),
        in_tagged_to_in_tagged=(
            enc_graph.in_tagged_to_in_tagged.pad_nonzeros(64)),
        in_tagged_to_special=(jax_util.pad_to(enc_graph.in_tagged_to_special,
                                              64)),
        in_tagged_node_indices=(jax_util.pad_to(
            enc_graph.in_tagged_node_indices, 64)))
    enc_meta_padded = automaton_builder.EncodedGraphMetadata(
        num_nodes=64, num_input_tagged_nodes=64)
    variant_weights = jnp.full([64, 5], 0.2)
    routing_params = automaton_builder.RoutingParams(
        move=jnp.full([5, 6, 2, 2], 0.2), special=jnp.full([5, 3, 2, 3], 0.2))
    tmat = builder.build_transition_matrix(routing_params, enc_graph_padded,
                                           enc_meta_padded)
    outs = automaton_sampling.one_node_particle_estimate(
        builder,
        tmat,
        variant_weights,
        start_machine_state=jnp.array([1., 0.]),
        node_index=0,
        steps=100,
        num_rollouts=100,
        max_possible_transitions=2,
        num_valid_nodes=enc_meta.num_nodes,
        rng=jax.random.PRNGKey(0))
    self.assertEqual(outs.shape, (64,))
    self.assertTrue(jnp.all(outs[:enc_meta.num_nodes] > 0))
    self.assertTrue(jnp.all(outs[enc_meta.num_nodes:] == 0))

  def test_all_nodes_particle_estimate(self):
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
        move=jnp.broadcast_to(
            jnp.pad(
                jnp.array([
                    [1., 0., 1., 0., 1., 0.],
                    [0., 1., 0., 1., 0., 1.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                ]).reshape([5, 6, 1, 1]), [(0, 0), (0, 0), (0, 0), (0, 1)]),
            [5, 6, 2, 2]),
        special=jnp.broadcast_to(
            jnp.array([
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]],
                [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]],
                [[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]],
            ]).reshape([5, 3, 1, 3]), [5, 3, 2, 3]))

    @jax.jit
    def go(variant_weights, routing_params, eps):
      variant_weights = ((1 - eps) * variant_weights +
                         eps * jnp.ones_like(variant_weights) / 5)
      routing_params = automaton_builder.RoutingParams(
          move=((1 - eps) * routing_params.move +
                eps * jnp.ones_like(routing_params.move) / 5),
          special=((1 - eps) * routing_params.special +
                   eps * jnp.ones_like(routing_params.special) / 5),
      )
      variant_weights = variant_weights / jnp.sum(
          variant_weights, axis=-1, keepdims=True)
      routing_params_sum = builder.routing_reduce(routing_params, "sum")
      routing_params = jax.tree_multimap(jax.lax.div, routing_params,
                                         routing_params_sum)
      tmat = builder.build_transition_matrix(routing_params, enc_graph,
                                             enc_meta)
      return automaton_sampling.all_nodes_particle_estimate(
          builder,
          tmat,
          variant_weights,
          jnp.pad(jnp.ones([4, 1]), [(0, 0), (0, 1)]),
          steps=100,
          rng=jax.random.PRNGKey(1),
          num_rollouts=10000,
          max_possible_transitions=2,  # only two edges to leave each node
          num_valid_nodes=enc_meta.num_nodes,
      )

    # Absorbing probs follow the paths described above.
    expected_absorbing_probs = jnp.array([
        [0, 0, 0.3, 0.7],
        [0, 0, 0, 0.81],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])

    particle_probs = go(variant_weights, routing_params, eps=0)

    # With 10000 rollouts we expect a standard deviation of up to
    # sqrt(0.3*0.7/10000) ~= 5e-3. Check that we are within 2 sigma.
    np.testing.assert_allclose(
        particle_probs, expected_absorbing_probs, atol=2 * 5e-3)


if __name__ == "__main__":
  absltest.main()
