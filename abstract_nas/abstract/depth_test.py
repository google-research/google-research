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

"""Tests for depth."""

from absl.testing import absltest as test
from abstract_nas.abstract import depth
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphNode
from abstract_nas.zoo import cnn


class DepthTest(test.TestCase):

  def setUp(self):
    super().setUp()

    self.graph, self.constants, _ = cnn.CifarNet()
    self.subgraph_model = SubgraphModel(self.graph, self.constants, {}, {})
    self.dp = depth.DepthProperty().infer(self.subgraph_model)

  def test_infer(self):
    depth_map = self.dp.depth_map
    self.assertLen(depth_map, 1)
    self.assertIn("input:0", depth_map)
    self.assertLen(depth_map["input:0"], 1)
    self.assertIn("fc/logits:0", depth_map["input:0"])
    self.assertEqual(depth_map["input:0"]["fc/logits:0"], 7)

  def test_mutate(self):
    delta_max = 5
    dp = depth.DepthProperty(p=1.0, delta_max=delta_max)
    dp = dp.infer(self.subgraph_model).mutate()
    depth_map = dp.depth_map
    self.assertLen(depth_map, 1)
    self.assertIn("input:0", depth_map)
    self.assertLen(depth_map["input:0"], 1)
    self.assertIn("fc/logits:0", depth_map["input:0"])
    self.assertEqual(abs(7 - depth_map["input:0"]["fc/logits:0"]), delta_max)

  def test_self_satisfy(self):
    # The graph which generated the property should always satisfy the property.
    self.assertTrue(self.dp.verify(self.subgraph_model))

  def test_unsatisfy(self):
    # This test removes the last dense layer, so the new graph should be less
    # deep.
    graph = new_graph(
        input_names=["input"], output_names=["fc/relu"], ops=self.graph.ops)
    subgraph_model = SubgraphModel(graph, self.constants, {}, {})
    self.assertFalse(self.dp.verify(subgraph_model))

  def test_satisfy(self):
    # This test removes the last dense layer, so the old graph should be more
    # deep.
    ops = self.graph.ops[:-1]
    ops[-1].name = "fc/logits"
    graph = new_graph(
        input_names=["input"], output_names=["fc/logits"], ops=ops)
    subgraph_model = SubgraphModel(graph, self.constants, {}, {})
    dp = depth.DepthProperty().infer(subgraph_model)
    self.assertTrue(dp.verify(self.subgraph_model))

  def test_rewire(self):
    # orig: conv, relu, pool, conv, relu, pool, flatten, dense, relu, dense
    # new:  conv, relu, pool, conv, gelu, pool, flatten, dense, relu, dense
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
    subgraph_model = SubgraphModel(graph, self.constants, {}, {}, subgraph_spec)
    dp = depth.DepthProperty().infer(subgraph_model)

    depth_map = dp.depth_map
    self.assertLen(depth_map, 1)
    self.assertIn("conv_layer0/avg_pool:0", depth_map)
    self.assertLen(depth_map["conv_layer0/avg_pool:0"], 2)
    self.assertIn("conv_layer1/relu:0", depth_map["conv_layer0/avg_pool:0"])
    self.assertEqual(
        depth_map["conv_layer0/avg_pool:0"]["conv_layer1/relu:0"], 1)
    self.assertIn("conv_layer1/gelu/1:0", depth_map["conv_layer0/avg_pool:0"])
    self.assertEqual(
        depth_map["conv_layer0/avg_pool:0"]["conv_layer1/gelu/1:0"], 1)

  def test_multi_input(self):
    ops = [
        new_op(
            op_name="dense0",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="relu0",
            op_type=OpType.RELU,
            input_names=["dense0"]),
        new_op(
            op_name="dense1",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="relu1",
            op_type=OpType.RELU,
            input_names=["dense1"]),
        new_op(
            op_name="dense2",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="relu2",
            op_type=OpType.RELU,
            input_names=["dense2"]),
        new_op(
            op_name="add0",
            op_type=OpType.ADD,
            input_names=["relu0", "relu1"]),
        new_op(
            op_name="add1",
            op_type=OpType.ADD,
            input_names=["relu1", "relu2"]),
    ]
    graph = new_graph(
        input_names=["input"], output_names=["add0", "add1"], ops=ops)
    subgraph_spec = [
        SubgraphNode(
            op=new_op(
                op_name="relu0",
                op_type=OpType.RELU,
                input_names=["dense0"])),
        SubgraphNode(
            op=new_op(
                op_name="relu1",
                op_type=OpType.RELU,
                input_names=["dense1"])),
        SubgraphNode(
            op=new_op(
                op_name="relu2",
                op_type=OpType.RELU,
                input_names=["dense2"])),
        SubgraphNode(
            op=new_op(
                op_name="add0",
                op_type=OpType.ADD,
                input_names=["relu0", "relu1"]),
            output_names=["add0"]),
        SubgraphNode(
            op=new_op(
                op_name="add1",
                op_type=OpType.ADD,
                input_names=["relu1", "relu2"]),
            output_names=["add1"]),
    ]
    replaced_graph = replace_subgraph(graph, subgraph_spec)
    subgraph_model = SubgraphModel(replaced_graph, {}, {}, {}, subgraph_spec)
    dp = depth.DepthProperty().infer(subgraph_model)
    depth_map = dp.depth_map

    self.assertLen(depth_map, 3)
    self.assertIn("dense0:0", depth_map)
    self.assertIn("dense1:0", depth_map)
    self.assertIn("dense2:0", depth_map)
    self.assertLen(depth_map["dense0:0"], 1)
    self.assertEqual(depth_map["dense0:0"]["add0:0"], 2)
    self.assertLen(depth_map["dense1:0"], 2)
    self.assertEqual(depth_map["dense1:0"]["add0:0"], 2)
    self.assertEqual(depth_map["dense1:0"]["add1:0"], 2)
    self.assertLen(depth_map["dense2:0"], 1)
    self.assertEqual(depth_map["dense2:0"]["add1:0"], 2)

if __name__ == "__main__":
  test.main()
