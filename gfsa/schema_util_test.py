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

"""Tests for gfsa.schema_util."""

from absl.testing import absltest
from absl.testing import parameterized
from gfsa import graph_types
from gfsa import schema_util


class SchemaUtilTest(parameterized.TestCase):

  def test_conforms_to_schema(self):
    test_schema = {
        graph_types.NodeType("a"):
            graph_types.NodeSchema(
                in_edges=[
                    graph_types.InEdgeType("ai_0"),
                    graph_types.InEdgeType("ai_1")
                ],
                out_edges=[graph_types.OutEdgeType("ao_0")]),
        graph_types.NodeType("b"):
            graph_types.NodeSchema(
                in_edges=[graph_types.InEdgeType("bi_0")],
                out_edges=[
                    graph_types.OutEdgeType("bo_0"),
                    graph_types.OutEdgeType("bo_1")
                ]),
    }

    # Valid graph
    test_graph = {
        graph_types.NodeId("A"):
            graph_types.GraphNode(
                graph_types.NodeType("a"), {
                    graph_types.OutEdgeType("ao_0"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("B"),
                            graph_types.InEdgeType("bi_0")),
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("A"),
                            graph_types.InEdgeType("ai_1"))
                    ]
                }),
        graph_types.NodeId("B"):
            graph_types.GraphNode(
                graph_types.NodeType("b"), {
                    graph_types.OutEdgeType("bo_0"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("A"),
                            graph_types.InEdgeType("ai_1"))
                    ],
                    graph_types.OutEdgeType("bo_1"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("B"),
                            graph_types.InEdgeType("bi_0"))
                    ]
                })
    }
    schema_util.assert_conforms_to_schema(test_graph, test_schema)

  @parameterized.named_parameters(
      {
          "testcase_name": "missing_node",
          "graph": {
              graph_types.NodeId("A"):
                  graph_types.GraphNode(
                      graph_types.NodeType("a"), {
                          graph_types.OutEdgeType("ao_0"): [
                              graph_types.InputTaggedNode(
                                  graph_types.NodeId("B"),
                                  graph_types.InEdgeType("bi_0"))
                          ]
                      })
          },
          "expected_error": "Node A has connection to missing node B"
      }, {
          "testcase_name": "bad_node_type",
          "graph": {
              graph_types.NodeId("A"):
                  graph_types.GraphNode(
                      graph_types.NodeType("z"), {
                          graph_types.OutEdgeType("ao_0"): [
                              graph_types.InputTaggedNode(
                                  graph_types.NodeId("A"),
                                  graph_types.InEdgeType("ai_0"))
                          ]
                      })
          },
          "expected_error": "Node A's type z not in schema"
      }, {
          "testcase_name": "missing_out_edge",
          "graph": {
              graph_types.NodeId("A"):
                  graph_types.GraphNode(
                      graph_types.NodeType("a"),
                      {graph_types.OutEdgeType("ao_0"): []})
          },
          "expected_error": "Node A missing out edge of type ao_0"
      }, {
          "testcase_name": "bad_out_edge_type",
          "graph": {
              graph_types.NodeId("A"):
                  graph_types.GraphNode(
                      graph_types.NodeType("a"), {
                          graph_types.OutEdgeType("ao_0"): [
                              graph_types.InputTaggedNode(
                                  graph_types.NodeId("A"),
                                  graph_types.InEdgeType("ai_0"))
                          ],
                          "foo": [
                              graph_types.InputTaggedNode(
                                  graph_types.NodeId("A"),
                                  graph_types.InEdgeType("ai_0"))
                          ]
                      })
          },
          "expected_error": "Node A has out-edges of invalid type foo"
      }, {
          "testcase_name": "bad_in_edge_type",
          "graph": {
              graph_types.NodeId("A"):
                  graph_types.GraphNode(
                      graph_types.NodeType("a"), {
                          graph_types.OutEdgeType("ao_0"): [
                              graph_types.InputTaggedNode(
                                  graph_types.NodeId("A"),
                                  graph_types.InEdgeType("bar"))
                          ],
                      })
          },
          "expected_error": "Node A has in-edges of invalid type bar"
      })
  def test_does_not_conform_to_schema(self, graph, expected_error):
    test_schema = {
        graph_types.NodeType("a"):
            graph_types.NodeSchema(
                in_edges=[
                    graph_types.InEdgeType("ai_0"),
                    graph_types.InEdgeType("ai_1")
                ],
                out_edges=[graph_types.OutEdgeType("ao_0")]),
        graph_types.NodeType("b"):
            graph_types.NodeSchema(
                in_edges=[graph_types.InEdgeType("bi_0")],
                out_edges=[
                    graph_types.OutEdgeType("bo_0"),
                    graph_types.OutEdgeType("bo_1")
                ]),
    }
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegex(ValueError, expected_error):
      schema_util.assert_conforms_to_schema(graph, test_schema)
    # pylint: enable=g-error-prone-assert-raises

  def test_all_input_tagged_nodes(self):
    # (note: python3 dicts maintain order, so B2 comes before B1)
    graph = {
        graph_types.NodeId("A"):
            graph_types.GraphNode(
                graph_types.NodeType("a"), {
                    graph_types.OutEdgeType("ao_0"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("B1"),
                            graph_types.InEdgeType("bi_1")),
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("A"),
                            graph_types.InEdgeType("ai_1"))
                    ]
                }),
        graph_types.NodeId("B2"):
            graph_types.GraphNode(
                graph_types.NodeType("b"), {
                    graph_types.OutEdgeType("bo_0"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("A"),
                            graph_types.InEdgeType("ai_1"))
                    ],
                    graph_types.OutEdgeType("bo_1"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("B1"),
                            graph_types.InEdgeType("bi_0"))
                    ]
                }),
        graph_types.NodeId("B1"):
            graph_types.GraphNode(
                graph_types.NodeType("b"), {
                    graph_types.OutEdgeType("bo_0"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("A"),
                            graph_types.InEdgeType("ai_1"))
                    ],
                    graph_types.OutEdgeType("bo_1"): [
                        graph_types.InputTaggedNode(
                            graph_types.NodeId("B2"),
                            graph_types.InEdgeType("bi_0"))
                    ]
                }),
    }
    expected_itns = [
        graph_types.InputTaggedNode(
            graph_types.NodeId("A"), graph_types.InEdgeType("ai_1")),
        graph_types.InputTaggedNode(
            graph_types.NodeId("B2"), graph_types.InEdgeType("bi_0")),
        graph_types.InputTaggedNode(
            graph_types.NodeId("B1"), graph_types.InEdgeType("bi_0")),
        graph_types.InputTaggedNode(
            graph_types.NodeId("B1"), graph_types.InEdgeType("bi_1")),
    ]

    actual_itns = schema_util.all_input_tagged_nodes(graph)
    self.assertEqual(actual_itns, expected_itns)


if __name__ == "__main__":
  absltest.main()
