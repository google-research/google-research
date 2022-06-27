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

"""Tests for symbolic.graphviz."""

from absl.testing import absltest

from symbolic_functionals.syfes.symbolic import graphviz


class GraphvizTest(absltest.TestCase):

  def test_create_graph(self):
    graph = graphviz.create_graph(
        feature_names=['f1'],
        shared_parameter_names=['c1'],
        variable_names=['v1', 'v2'],
        bound_parameter_names=['gamma_utransform'],
        instruction_list=[
            ['UTransformInstruction', 'v1', 'f1'],
            ['Power2Instruction', 'v2', 'v1'],
            ['MultiplicationAdditionInstruction', 'v1', 'f1', 'v2'],
            ['DivisionInstruction', 'v1', 'v1', 'c1'],
        ])
    self.assertEqual(
        graph.source,
        """digraph {
	f1 [label=f1 fillcolor="#0072b2" style=filled]
	c1 [label=c1 fillcolor="#de8f05" style=filled]
	gamma_utransform [label=gamma_utransform fillcolor="#cc79a7" style=filled]
	"UTransformInstruction--0" [label=UTransform fillcolor="#009e73" shape=square style=filled]
	f1 [label=f1 fillcolor="#0072b2" style=filled]
		f1 -> "UTransformInstruction--0"
	gamma_utransform [label=gamma_utransform fillcolor="#cc79a7" style=filled]
		gamma_utransform -> "UTransformInstruction--0"
	"v1--1" [label=v1]
		"UTransformInstruction--0" -> "v1--1"
	"Power2Instruction--0" [label="a^2" fillcolor="#009e73" shape=square style=filled]
		"v1--1" -> "Power2Instruction--0"
	"v2--1" [label=v2]
		"Power2Instruction--0" -> "v2--1"
	"MultiplicationAdditionInstruction--0" [label="+=a*b" fillcolor="#009e73" shape=square style=filled]
	f1 [label=f1 fillcolor="#0072b2" style=filled]
		f1 -> "MultiplicationAdditionInstruction--0"
		"v2--1" -> "MultiplicationAdditionInstruction--0"
		"v1--1" -> "MultiplicationAdditionInstruction--0" [color="black:black"]
	"v1--2" [label=v1]
		"MultiplicationAdditionInstruction--0" -> "v1--2"
	"DivisionInstruction--0" [label="a/b" fillcolor="#009e73" shape=square style=filled]
		"v1--2" -> "DivisionInstruction--0"
	c1 [label=c1 fillcolor="#de8f05" style=filled]
		c1 -> "DivisionInstruction--0"
	"v1--3" [label=v1]
		"DivisionInstruction--0" -> "v1--3"
}""")


if __name__ == '__main__':
  absltest.main()
