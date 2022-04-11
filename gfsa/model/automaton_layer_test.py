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

"""Tests for gfsa.model.automaton_layer."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
from gfsa import automaton_builder
from gfsa import graph_types
from gfsa import sparse_operator
from gfsa.model import automaton_layer
from gfsa.model import side_outputs

# pyformat: disable
# pylint: disable=g-complex-comprehension
TESTCASES = [
    {
        "testcase_name": "_".join([
            "shared" if shared else "unshared",
            variant_type,
            "gate" if use_gate else "softmax",
        ]),
        "shared": shared,
        "variant_weights": variant_weights,
        "use_gate": use_gate,
        "logit_scaling": "learned",
        "estimator_type": "solver",
    }
    for shared in (True, False)
    for use_gate in (True, False)
    for variant_type, variant_weights in (
        ("no_variants", lambda: None),
        ("shared_variants", lambda: jnp.zeros((32, 32, 2))),
        ("unshared_variants", lambda: jnp.zeros((32, 32, 3, 2))),
    )
] + [
    {
        "testcase_name": "shared_no_variants_gate_dynamic_scaling",
        "shared": True,
        "variant_weights": lambda: None,
        "use_gate": True,
        "logit_scaling": "dynamic",
        "estimator_type": "solver",
    },
    {
        "testcase_name": "shared_unshared_variants_one_sample",
        "shared": True,
        "variant_weights": lambda: jnp.zeros((32, 32, 3, 2)),
        "use_gate": False,
        "logit_scaling": "none",
        "estimator_type": "one_sample",
        "sampling_max_possible_transitions": 4,
    },
]
# pylint: enable=g-complex-comprehension
# pyformat: enable


class AutomatonLayerTest(parameterized.TestCase):

  @parameterized.named_parameters(*TESTCASES)
  def test_automaton_layer_abstract_init(self, shared, variant_weights,
                                         use_gate, estimator_type, **kwargs):
    # Create a simple schema and empty encoded graph.
    schema = {
        graph_types.NodeType("a"):
            graph_types.NodeSchema(
                in_edges=[graph_types.InEdgeType("ai_0")],
                out_edges=[graph_types.OutEdgeType("ao_0")]),
    }
    builder = automaton_builder.AutomatonBuilder(schema)
    encoded_graph = automaton_builder.EncodedGraph(
        initial_to_in_tagged=sparse_operator.SparseCoordOperator(
            input_indices=jnp.zeros((128, 1), dtype=jnp.int32),
            output_indices=jnp.zeros((128, 2), dtype=jnp.int32),
            values=jnp.zeros((128,), dtype=jnp.float32),
        ),
        initial_to_special=jnp.zeros((32,), dtype=jnp.int32),
        in_tagged_to_in_tagged=sparse_operator.SparseCoordOperator(
            input_indices=jnp.zeros((128, 1), dtype=jnp.int32),
            output_indices=jnp.zeros((128, 2), dtype=jnp.int32),
            values=jnp.zeros((128,), dtype=jnp.float32),
        ),
        in_tagged_to_special=jnp.zeros((64,), dtype=jnp.int32),
        in_tagged_node_indices=jnp.zeros((64,), dtype=jnp.int32),
    )

    # Make sure the layer can be initialized and applied within a model.
    # This model is fairly simple; it just pretends that the encoded graph and
    # variants depend on the input.
    class TestModel(flax.deprecated.nn.Module):

      def apply(self, dummy_ignored):
        abstract_encoded_graph = jax.tree_map(
            lambda y: jax.lax.tie_in(dummy_ignored, y), encoded_graph)
        abstract_variant_weights = jax.tree_map(
            lambda y: jax.lax.tie_in(dummy_ignored, y), variant_weights())
        return automaton_layer.FiniteStateGraphAutomaton(
            encoded_graph=abstract_encoded_graph,
            variant_weights=abstract_variant_weights,
            dynamic_metadata=automaton_builder.EncodedGraphMetadata(
                num_nodes=32, num_input_tagged_nodes=64),
            static_metadata=automaton_builder.EncodedGraphMetadata(
                num_nodes=32, num_input_tagged_nodes=64),
            builder=builder,
            num_out_edges=3,
            num_intermediate_states=4,
            share_states_across_edges=shared,
            use_gate_parameterization=use_gate,
            estimator_type=estimator_type,
            name="the_layer",
            **kwargs)

    with side_outputs.collect_side_outputs() as side:
      with flax.deprecated.nn.stochastic(jax.random.PRNGKey(0)):
        # For some reason init_by_shape breaks the custom_vjp?
        abstract_out, unused_params = TestModel.init(
            jax.random.PRNGKey(1234), jnp.zeros((), jnp.float32))

    del unused_params
    self.assertEqual(abstract_out.shape, (3, 32, 32))

    if estimator_type == "one_sample":
      log_prob_key = "/the_layer/one_sample_log_prob_per_edge_per_node"
      self.assertIn(log_prob_key, side)
      self.assertEqual(side[log_prob_key].shape, (3, 32))


if __name__ == "__main__":
  absltest.main()
