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

"""Tests for the graph synthesizer."""

import functools
import random as py_rand
import time

from absl import logging
from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.abstract import depth
from abstract_nas.abstract import linear
from abstract_nas.abstract import shape
from abstract_nas.abstract.fingerprint import fingerprint_graph
from abstract_nas.model import Model
from abstract_nas.model import subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.synthesis.graph import GraphSynthesizer
from abstract_nas.synthesis.prog_sequential import ProgressiveSequentialSynthesizer as PSS
from abstract_nas.zoo import cnn
from abstract_nas.zoo import resnetv1


class GraphSynthesizerTest(test.TestCase):

  def setUp(self):
    super().setUp()

    seed = int(time.time())
    logging.info("Seed: %d", seed)
    py_rand.seed(seed)

    self.graph, self.constants, _ = cnn.CifarNet()
    self.m = Model(self.graph, self.constants)
    self.input = {"input": jnp.ones((5, 32, 32, 3))}
    self.state = self.m.init(random.PRNGKey(0), self.input)
    self.out = self.m.apply(self.state, self.input)[self.graph.output_names[0]]
    self.max_size = int(10e8)

  def _synthesize(self, subg, props):
    ctr = functools.partial(PSS, generation=0, max_len=3, filter_progress=True)
    synthesizer = GraphSynthesizer(
        [(subg, props)], sequential_ctr=ctr, generation=0)

    subg = synthesizer.synthesize()[0]
    subg_spec = subg.subgraph
    logging.info("synthesized...")
    for node in subg_spec:
      logging.info("%s", node.op.name)
      logging.info("%s", node.output_names)
    logging.info("")

    fingerprint_orig = fingerprint_graph(self.graph, self.constants, self.input)
    fingerprint_new = fingerprint_graph(subg.graph, self.constants, self.input)
    self.assertNotEqual(fingerprint_orig, fingerprint_new)

  def test_synthesizer_easy_one(self):
    """Replacing [conv3x3(features = 64)]."""
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:5]]
    subg[-1].output_names = self.graph.ops[5].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    dp = depth.DepthProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, dp])

  def test_synthesizer_easy_two(self):
    """Replacing [conv3x3(features = 64)]."""
    py_rand.seed(10)
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:5]]
    subg[-1].output_names = self.graph.ops[5].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    dp = depth.DepthProperty().infer(subgraph_model)
    lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, dp, lp])

  def test_synthesizer_one(self):
    """Replacing [conv3x3(features = 64), ReLU]."""
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:6]]
    subg[-1].output_names = self.graph.ops[6].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    dp = depth.DepthProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, dp])

  def test_synthesizer_two(self):
    """Replacing [conv3x3(features = 64), ReLU, avgpool2x2(strides=2x2)]."""
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:7]]
    subg[-1].output_names = self.graph.ops[7].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, lp])

  def test_synthesizer_resnet_small(self):
    self.graph, self.constants, _ = resnetv1.ResNet18(
        num_classes=10, input_resolution="small")
    self.m = Model(self.graph, self.constants)
    self.input = {"input": jnp.ones((5, 32, 32, 3))}
    self.state = self.m.init(random.PRNGKey(0), self.input)
    self.out = self.m.apply(self.state, self.input)[self.graph.output_names[0]]
    self.max_size = int(10e8)

    subg_ops = [self.graph.ops[4]] + self.graph.ops[9:11]
    subg = [subgraph.SubgraphNode(op=o) for o in subg_ops]
    subg[-1].output_names = [f"{subg[-1].op.name}:0"]
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, lp])

  def test_synthesizer_resnet_big(self):
    self.graph, self.constants, _ = resnetv1.ResNet18(
        num_classes=10, input_resolution="small")
    self.m = Model(self.graph, self.constants)
    self.input = {"input": jnp.ones((5, 32, 32, 3))}
    self.state = self.m.init(random.PRNGKey(0), self.input)
    self.out = self.m.apply(self.state, self.input)[self.graph.output_names[0]]
    self.max_size = int(10e8)

    subg_ops = self.graph.ops[3:5] + self.graph.ops[8:12]
    subg = [subgraph.SubgraphNode(op=o) for o in subg_ops]
    subg[-1].output_names = [f"{subg[-1].op.name}:0"]
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, lp])

if __name__ == "__main__":
  test.main()

