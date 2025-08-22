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

"""Tests for linear."""

import functools

import jax
from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.abstract import linear
from abstract_nas.model import Model
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphNode
from abstract_nas.zoo import cnn


class LinearTest(test.TestCase):

  def setUp(self):
    super().setUp()

    self.graph_dense = new_graph(["input"], ["output"], [
        new_op(
            op_name="output",
            op_type=OpType.SOFTMAX,
            # op_kwargs={"features": 10},
            input_names=["input"])
    ])
    state_dense = Model(self.graph_dense).init(
        random.PRNGKey(0), {"input": jnp.ones((5, 5, 5))})
    self.subgraph_dense = SubgraphModel(
        self.graph_dense, None, state_dense, {"input": jnp.ones((5, 5, 5))})
    self.lp_dense = linear.LinopProperty().infer(self.subgraph_dense)

    self.graph_conv = new_graph(["input"], ["output"], [
        new_op(
            op_name="output",
            op_type=OpType.CONV,
            op_kwargs={
                "features": 10,
                "kernel_size": [3, 3]
            },
            input_names=["input"])
    ])
    state_conv = Model(self.graph_conv).init(
        random.PRNGKey(0), {"input": jnp.ones((5, 5, 5))})
    self.subgraph_conv = SubgraphModel(
        self.graph_conv, None, state_conv, {"input": jnp.ones((5, 5, 5))})
    self.lp_conv = linear.LinopProperty().infer(self.subgraph_conv)

  def test_infer(self):
    pairing = self.lp_dense.pairings
    self.assertLen(pairing, 1)
    pairing = pairing["output"]
    self.assertLen(pairing, 1)
    pairing = pairing["input"]
    self.assertEqual(pairing.in_dims, 3)
    self.assertEqual(pairing.out_dims, 3)

    self.assertEqual(pairing[0][0], linear.Pairing.Mapping.ONE_TO_ONE)
    self.assertEqual(pairing[1][0], linear.Pairing.Mapping.NONE)
    self.assertEqual(pairing[2][0], linear.Pairing.Mapping.ALL_TO_ONE)
    self.assertEqual(pairing[0][1], linear.Pairing.Mapping.NONE)
    self.assertEqual(pairing[1][1], linear.Pairing.Mapping.ONE_TO_ONE)
    self.assertEqual(pairing[2][1], linear.Pairing.Mapping.ALL_TO_ONE)
    self.assertEqual(pairing[0][2], linear.Pairing.Mapping.NONE)
    self.assertEqual(pairing[1][2], linear.Pairing.Mapping.NONE)
    self.assertEqual(pairing[2][2], linear.Pairing.Mapping.ALL_TO_ONE)

  def test_abstract(self):
    graphs = []

    conv_op = functools.partial(
        new_op,
        op_type=OpType.CONV,
        op_kwargs={
            "features": 10,
            "kernel_size": [3, 3]
        })
    dense_op = functools.partial(
        new_op,
        op_type=OpType.DENSE,
        op_kwargs={
            "features": 10,
        })

    for op_type in [
        OpType.RELU, OpType.SOFTMAX, OpType.LAYER_NORM, OpType.BATCH_NORM
    ]:
      for op_ctr in [conv_op, dense_op]:
        graphs.append([
            op_ctr(input_names=["input"], op_name="other"),
            new_op(op_name="output", op_type=op_type, input_names=["other"])
        ])
        graphs.append([
            new_op(op_name="other", op_type=op_type, input_names=["input"]),
            op_ctr(input_names=["other"], op_name="output"),
        ])

    input_tensor = {"input": jnp.ones((5, 5, 5, 5))}
    for graph in graphs:
      graph = new_graph(["input"], ["output"], graph)
      state = Model(graph).init(random.PRNGKey(1), input_tensor)

      # Make all the kernels positive, otherwise, the ReLU might zero out the
      # entire tensor.
      state = jax.tree_util.tree_map(abs, state)

      subg_model = SubgraphModel(graph, None, state, input_tensor)
      lp_abstract = linear.LinopProperty().infer(subg_model, abstract=True)
      lp_concrete = linear.LinopProperty().infer(subg_model, abstract=False)
      pairings_concerete = lp_concrete.pairings["output"]["input"].mappings
      pairings_abstract = lp_abstract.pairings["output"]["input"].mappings

      print("concrete:", pairings_concerete)
      print("abstract:", pairings_abstract)
      self.assertTrue(((pairings_abstract - pairings_concerete) == 0).all())

  def test_self_satisfy(self):
    self.assertTrue(self.lp_dense.verify(self.subgraph_dense))
    self.assertTrue(self.lp_conv.verify(self.subgraph_conv))

  def test_satisfy(self):
    self.assertTrue(self.lp_dense.verify(self.subgraph_conv))

  def test_unsatisfy(self):
    self.assertFalse(self.lp_conv.verify(self.subgraph_dense))

  def test_full(self):
    graph, constants, _ = cnn.CifarNet()
    subgraph_model = SubgraphModel(
        graph, constants, None, {"input": jnp.ones((10, 32, 32, 3))})
    linear.LinopProperty().infer(subgraph_model)

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
    inp = {"input": jnp.ones((10, 32, 32, 3))}
    subgraph_model = SubgraphModel(replaced_graph, {}, {}, inp, subgraph_spec)
    lp = linear.LinopProperty().infer(subgraph_model)
    pairings = lp.pairings

    self.assertLen(pairings, 2)
    self.assertIn("add0:0", pairings)
    self.assertLen(pairings["add0:0"], 2)
    self.assertIn("dense0:0", pairings["add0:0"])
    self.assertIn("dense1:0", pairings["add0:0"])
    self.assertIn("add1:0", pairings)
    self.assertLen(pairings["add1:0"], 2)
    self.assertIn("dense1:0", pairings["add1:0"])
    self.assertIn("dense2:0", pairings["add1:0"])


if __name__ == "__main__":
  test.main()
