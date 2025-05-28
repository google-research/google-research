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

"""Tests for shape."""

from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.abstract import shape
from abstract_nas.model import Model
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphNode
from abstract_nas.zoo import cnn


class ShapeTest(test.TestCase):

  def setUp(self):
    super().setUp()

    self.graph, self.constants, _ = cnn.CifarNet()
    state = Model(self.graph, self.constants).init(
        random.PRNGKey(0), {"input": jnp.ones((5, 32, 32, 3))})
    self.subgraph_model = SubgraphModel(
        self.graph, self.constants, state, {"input": jnp.ones((5, 32, 32, 3))})
    self.sp = shape.ShapeProperty().infer(self.subgraph_model)

  def test_infer(self):
    self.assertLen(self.sp.input_shapes, 1)
    self.assertIn("input", self.sp.input_shapes)
    self.assertEqual(self.sp.input_shapes["input"], (5, 32, 32, 3))

    self.assertLen(self.sp.output_shapes, 1)
    self.assertIn("fc/logits", self.sp.output_shapes)
    self.assertEqual(self.sp.output_shapes["fc/logits"], (5, 10))

  def test_infer_intermediates(self):
    sp = shape.ShapeProperty().infer(self.subgraph_model, intermediates=True)
    self.assertLen(sp.input_shapes, 1)
    self.assertIn("input", sp.input_shapes)
    self.assertEqual(sp.input_shapes["input"], (5, 32, 32, 3))

    self.assertLen(sp.output_shapes,
                   len(self.subgraph_model.subg_model.graph.ops))
    self.assertIn("fc/logits", sp.output_shapes)
    self.assertEqual(sp.output_shapes["fc/logits"], (5, 10))

  def test_infer_max_size(self):
    # The largest intermediate tensor comes immediately after the second conv
    # and has size 163840 = 5 * 32 * 32 * 32 * 2
    self.assertRaisesRegex(
        RuntimeError, ".*max_size.*",
        shape.ShapeProperty().infer, self.subgraph_model, max_size=327680 - 1)
    shape.ShapeProperty().infer(self.subgraph_model, max_size=327680)

  def test_self_satisfy(self):
    # The graph which generated the property should always satisfy the property.
    self.assertTrue(self.sp.verify(self.subgraph_model))

  def test_unsatisfy(self):
    # This test removes the last dense layer, so the new graph should have a
    # different shape (and therefore not satisfy the inferred shape property).
    graph = new_graph(
        input_names=["input"], output_names=["fc/relu"], ops=self.graph.ops)
    state = Model(graph, self.constants).init(
        random.PRNGKey(0), {"input": jnp.ones((5, 32, 32, 3))})
    subgraph_model = SubgraphModel(graph, self.constants, state,
                                   {"input": jnp.ones((5, 32, 32, 3))})
    self.assertFalse(self.sp.verify(subgraph_model))

  def test_rewire(self):
    subgraph_spec = [
        SubgraphNode(
            op=new_op(
                op_name="conv_layer1/conv/1",
                op_type=OpType.CONV,
                op_kwargs={
                    "features": 64,
                    "kernel_size": [1, 1]
                },
                input_names=["conv_layer0/avg_pool"]),),
        SubgraphNode(
            op=new_op(
                op_name="conv_layer1/gelu/1",
                op_type=OpType.GELU,
                input_names=["conv_layer1/conv/1"]),
            output_names=["conv_layer1/relu"])
    ]

    graph = replace_subgraph(self.graph, subgraph_spec)
    state = Model(graph, self.constants).init(
        random.PRNGKey(0), {"input": jnp.ones((5, 32, 32, 3))})
    subgraph_model = SubgraphModel(
        graph, self.constants, state,
        {"input": jnp.ones((5, 32, 32, 3))}, subgraph_spec)
    sp = shape.ShapeProperty().infer(subgraph_model)

    self.assertLen(sp.input_shapes, 1)
    self.assertIn("conv_layer0/avg_pool:0", sp.input_shapes)
    self.assertLen(sp.output_shapes, 2)
    self.assertIn("conv_layer1/gelu/1:0", sp.output_shapes)
    self.assertIn("conv_layer1/relu:0", sp.output_shapes)

if __name__ == "__main__":
  test.main()
