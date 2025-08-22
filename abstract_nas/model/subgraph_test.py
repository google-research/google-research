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

"""Tests for graph_edits."""

from clu import parameter_overview
import flax
from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.model import Model
from abstract_nas.model import subgraph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import OpType
from abstract_nas.zoo import cnn
from abstract_nas.zoo import resnetv1


class SubgraphTest(test.TestCase):

  def setUp(self):
    super().setUp()

    self.graph, self.constants, _ = cnn.CifarNet()
    self.model = Model(self.graph, self.constants)
    self.state = self.model.init(random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))

    self.subgraph = [
        subgraph.SubgraphNode(
            op=new_op(
                op_name="conv_layer1/conv/1",
                op_type=OpType.CONV,
                op_kwargs={
                    "features": 64,
                    "kernel_size": [1, 1]
                },
                input_names=["conv_layer0/avg_pool"]),),
        subgraph.SubgraphNode(
            op=new_op(
                op_name="conv_layer1/gelu/1",
                op_type=OpType.GELU,
                input_names=["conv_layer1/conv/1"]),
            output_names=["conv_layer1/relu"])
    ]
    self.new_graph = subgraph.replace_subgraph(self.graph, self.subgraph)
    self.new_model = Model(self.new_graph, self.constants)
    self.new_state = self.new_model.init(
        random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))

  def test_subgraph_inserted(self):
    """Tests whether subgraph nodes were inserted."""
    for node in self.subgraph:
      found = False
      for op in self.new_graph.ops:
        if op.name == node.op.name:
          found = True
          break
      self.assertTrue(found, f"Did not find {node.op.name} in new graph")

  def test_subgraph_execution(self):
    """Tests whether new graph can be executed."""
    y = self.new_model.apply(self.new_state, jnp.ones((10, 32, 32, 3)))
    self.assertEqual(y.shape, (10, 10))

  def test_subgraph_pruning(self):
    """Tests whether new graph was pruned of old nodes."""
    new_params = flax.core.unfreeze(self.new_state)["params"]
    new_param_count = parameter_overview.count_parameters(new_params)

    params = flax.core.unfreeze(self.state)["params"]
    param_count = parameter_overview.count_parameters(params)
    self.assertLess(new_param_count, param_count)

  def test_weight_inheritance(self):
    """Tests weight inheritance."""
    old_params = flax.core.unfreeze(self.state)["params"]
    new_params = flax.core.unfreeze(self.new_state)["params"]

    frozen_params, trainable_params = subgraph.inherit_params(
        new_params, old_params)

    self.assertLen(new_params, len(trainable_params) + len(frozen_params))

    for param in ["fc/dense", "fc/logits", "conv_layer0/conv"]:
      assert param in frozen_params, f"expected param {param} to be frozen"
    self.assertIn("conv_layer1/conv/1", trainable_params,
                  ("expected param layer1/conv/1 to be trainable"))

  def test_multi_input_output(self):
    """Tests a subgraph substitution on a graph with multiple inputs / output ops.

    We use a ResNet model, which has skip connections. This test checks that the
    substitution produces the expected number of ops, and also that the newly
    produced graph is still executable.
    """

    graph, constants, _ = resnetv1.ResNet18(
        num_classes=10, input_resolution="small")
    model = Model(graph, constants)
    state = model.init(random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
    y = model.apply(state, jnp.ones((10, 32, 32, 3)))
    self.assertEqual(y.shape, (10, 10))

    subg = [
        subgraph.SubgraphNode(
            op=new_op(
                op_name="subgraph/conv0",
                op_type=OpType.CONV,
                op_kwargs={
                    "features": 64,
                    "kernel_size": [1, 1]
                },
                input_names=["resnet11/skip/relu1"])),
        subgraph.SubgraphNode(
            op=new_op(
                op_name="subgraph/gelu1",
                op_type=OpType.GELU,
                input_names=["subgraph/conv0"]),
            output_names=["resnet_stride1_filtermul1_basic12/relu2"])
    ]
    new_graph = subgraph.replace_subgraph(graph, subg)

    # the subgraph is 2 ops (conv / gelu) replacing 3 ops (conv / bn / relu)
    self.assertLen(graph.ops, len(new_graph.ops) + 1)

    new_model = Model(new_graph, constants)
    new_state = new_model.init(
        random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))

    y = new_model.apply(new_state, jnp.ones((10, 32, 32, 3)))
    self.assertEqual(y.shape, (10, 10))

if __name__ == "__main__":
  test.main()
