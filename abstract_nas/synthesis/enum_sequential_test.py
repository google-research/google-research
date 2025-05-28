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

"""Tests for enum_sequential."""

from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.abstract import depth
from abstract_nas.abstract import linear
from abstract_nas.abstract import shape
from abstract_nas.model import Model
from abstract_nas.model import subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.synthesis.enum_sequential import EnumerativeSequentialSynthesizer
from abstract_nas.synthesis.enum_sequential import sequence_generator
from abstract_nas.zoo import cnn


class EnumSequentialTest(test.TestCase):

  def setUp(self):
    super().setUp()

    self.graph, self.constants, _ = cnn.CifarNet()
    self.m = Model(self.graph, self.constants)
    self.input = {"input": jnp.ones((5, 32, 32, 3))}
    self.state = self.m.init(random.PRNGKey(0), self.input)
    self.out = self.m.apply(self.state, self.input)["fc/logits"]
    self.max_size = int(10e8)

    self.hard = False

  def test_sequence_generator(self):
    """Test the sequence_generator function for correctness.

    seq_generator should generate
      [[0], [1], [2],
       [0,0], [0,1], [0,2], [1,0], [1,1], [1,2], ...,
       [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], ...,]
    """

    def el_generator():
      i = 0
      while True:
        if i > 2:
          return
        yield i
        i += 1

    seqs = list(sequence_generator(el_generator, 3))
    self.assertLen(seqs, 3 + 3**2 + 3**3)
    for i in range(len(seqs)):
      if i < 3:
        self.assertEqual(seqs[i], [i])
      elif i < 3 + 3**2:
        self.assertEqual(seqs[i], [(i - 3) // 3, i % 3])
      else:
        self.assertEqual(seqs[i], [(i - 12) // 3 // 3,
                                   (i - 12) // 3 % 3, i % 3])

  def test_kwargs_for_op_to_product(self):
    op_kwargs = {"a": [1, 2, 3], "b": [1, 2], "c": [5, 6]}
    input_kwargs = {"d": [1], "e": [1, 2], "f": [3, 4]}
    product = EnumerativeSequentialSynthesizer.kwargs_for_op_to_product(
        op_kwargs, input_kwargs)
    expected_length = 1
    for _, v in op_kwargs.items():
      expected_length *= len(v)
    for _, v in input_kwargs.items():
      expected_length *= len(v)
    self.assertLen(product, expected_length)

    op_setting = {"a": 2, "b": 2, "c": 5}
    input_setting = {"d": 1, "e": 2, "f": 3}
    self.assertIn((op_setting, input_setting), product)

  def _synthesize(self, subg, props):
    synthesizer = EnumerativeSequentialSynthesizer(
        [(subg, props)], 0, max_len=3)

    subg = synthesizer.synthesize()[0]

    m = Model(subg.graph, self.constants)
    state = m.init(random.PRNGKey(0), self.input)
    out = m.apply(state, self.input)["fc/logits"]
    self.assertTrue((out != self.out).any())

  def test_synthesizer_easy_one(self):
    """Replacing [conv3x3(features = 64)].

    Because we do not test linear, this is replaced by dense3x3(features = 64)
    due to the enumeration order.
    """
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:5]]
    subg[-1].output_names = self.graph.ops[5].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    dp = depth.DepthProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, dp])

  def test_synthesizer_easy_two(self):
    """Replacing [conv3x3(features = 64)].

    Because we test all three props, this is replaced by conv3x3(features = 64)
    (i.e., an identical op) due to the enumeration order.
    """
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:5]]
    subg[-1].output_names = self.graph.ops[5].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    dp = depth.DepthProperty().infer(subgraph_model)
    lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, dp, lp])

  def test_synthesizer_one(self):
    """Replacing [conv3x3(features = 64), ReLU].

    Because we do not check for the linear property, [dense(features = 64),
    ReLU] works as well (which is what is synthesized due to the enumeration
    order).
    """
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:6]]
    subg[-1].output_names = self.graph.ops[6].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    dp = depth.DepthProperty().infer(subgraph_model)
    # lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, dp])

  def test_synthesizer_two(self):
    """Replacing [conv3x3(features = 64), ReLU, avgpool2x2(strides=2x2)].

    Because we do not check for the depth property, [dense(features = 64),
    avgpool2x2(strides=2x2)] works as well (which is what is synthesized due to
    the enumeration order).
    """
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:7]]
    subg[-1].output_names = self.graph.ops[7].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, lp])

  def test_synthesizer_hard(self):
    if not self.hard:
      return
    subg = [subgraph.SubgraphNode(op=o) for o in self.graph.ops[4:7]]
    subg[-1].output_names = self.graph.ops[7].input_names
    subgraph_model = SubgraphModel(
        self.graph, self.constants, self.state, self.input, subg)
    sp = shape.ShapeProperty().infer(subgraph_model, max_size=self.max_size)
    dp = depth.DepthProperty().infer(subgraph_model)
    lp = linear.LinopProperty().infer(subgraph_model)
    self._synthesize(subgraph_model, [sp, dp, lp])

if __name__ == "__main__":
  test.main()

