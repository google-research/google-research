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

"""Tests for sequential."""

from absl.testing import absltest as test
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.abstract import shape
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphNode
from abstract_nas.synthesis import sequential
from abstract_nas.zoo import cnn


class TestSequentialSynthesizer(sequential.AbstractSequentialSynthesizer):

  def synthesize(self):
    pass


class SequentialTest(test.TestCase):

  def test_abstract_sequential_synthesizer(self):
    graph, constants, _ = cnn.CifarNet()
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
    subgraph = SubgraphModel(graph, constants, None,
                             {"input": jnp.zeros((5, 32, 32, 10))},
                             subgraph_spec)
    TestSequentialSynthesizer([(subgraph, [])], 0)

  def test_abstract_sequential_synthesizer_fail(self):
    graph, constants, _ = cnn.CifarNet()
    subgraph_spec = [
        SubgraphNode(
            op=new_op(
                op_name="conv_layer1/conv/1",
                op_type=OpType.CONV,
                op_kwargs={
                    "features": 64,
                    "kernel_size": [1, 1]
                },
                input_names=["conv_layer0/avg_pool"]),
            output_names=["conv_layer1/conv"]),
        SubgraphNode(
            op=new_op(
                op_name="conv_layer1/gelu/1",
                op_type=OpType.GELU,
                input_names=["conv_layer1/conv"]),
            output_names=["conv_layer1/relu"])
    ]
    subgraph = SubgraphModel(graph, constants, None,
                             {"input": jnp.zeros((5, 32, 32, 10))},
                             subgraph_spec)
    self.assertRaisesRegex(ValueError, ".*exactly one input.*",
                           TestSequentialSynthesizer, [(subgraph, [])], 0)

  def test_abstract_sequential_synthesizer_output_features(self):
    graph, constants, _ = cnn.CifarNet()
    subgraph_spec = [
        SubgraphNode(
            op=new_op(
                op_name="conv_layer1/conv",
                op_type=OpType.CONV,
                op_kwargs={
                    "features": "S:-1*2",
                    "kernel_size": [1, 1]
                },
                input_names=["conv_layer0/avg_pool"]),),
        SubgraphNode(
            op=new_op(
                op_name="conv_layer1/relu",
                op_type=OpType.RELU,
                input_names=["conv_layer1/conv"]),
            output_names=["conv_layer1/relu"])
    ]
    subgraph = replace_subgraph(graph, subgraph_spec)
    subgraph_model = SubgraphModel(subgraph, constants, None,
                                   {"input": jnp.zeros((5, 32, 32, 10))},
                                   subgraph_spec)
    sp = shape.ShapeProperty().infer(subgraph_model)
    syn = TestSequentialSynthesizer([(subgraph_model, [sp])], 0)
    self.assertEqual(syn.output_features_mul, 2)
    self.assertEqual(syn.output_features_div, 1)

if __name__ == "__main__":
  test.main()
