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

"""Tests for gfsa.ast_spec_inference."""

from absl.testing import absltest
from gfsa import ast_spec_inference
from gfsa import generic_ast_graphs


class ASTSpecInferenceTest(absltest.TestCase):

  def test_merge_observations(self):
    observations_a = ast_spec_inference.ASTObservations(
        example_count=3,
        node_types={
            "appears_in_a":
                ast_spec_inference.NodeObservations(
                    count=12,
                    count_root=3,
                    fields={
                        "foo":
                            ast_spec_inference.FieldObservations(
                                count_one=4, count_many=5)
                    }),
            "appears_in_both":
                ast_spec_inference.NodeObservations(
                    count=67,
                    count_root=8,
                    fields={
                        "field_in_a":
                            ast_spec_inference.FieldObservations(
                                count_one=1, count_many=2),
                        "field_in_both":
                            ast_spec_inference.FieldObservations(
                                count_one=3, count_many=4),
                    }),
        })
    observations_b = ast_spec_inference.ASTObservations(
        example_count=5,
        node_types={
            "appears_in_b":
                ast_spec_inference.NodeObservations(
                    count=7,
                    count_root=5,
                    fields={
                        "bar":
                            ast_spec_inference.FieldObservations(
                                count_one=3, count_many=2)
                    }),
            "appears_in_both":
                ast_spec_inference.NodeObservations(
                    count=13,
                    count_root=10,
                    fields={
                        "field_in_b":
                            ast_spec_inference.FieldObservations(
                                count_one=2, count_many=1),
                        "field_in_both":
                            ast_spec_inference.FieldObservations(
                                count_one=10, count_many=20),
                    }),
        })

    observations_merged = observations_a + observations_b
    expected = ast_spec_inference.ASTObservations(
        example_count=8,
        node_types={
            "appears_in_a":
                ast_spec_inference.NodeObservations(
                    count=12,
                    count_root=3,
                    fields={
                        "foo":
                            ast_spec_inference.FieldObservations(
                                count_one=4, count_many=5)
                    }),
            "appears_in_b":
                ast_spec_inference.NodeObservations(
                    count=7,
                    count_root=5,
                    fields={
                        "bar":
                            ast_spec_inference.FieldObservations(
                                count_one=3, count_many=2)
                    }),
            "appears_in_both":
                ast_spec_inference.NodeObservations(
                    count=80,
                    count_root=18,
                    fields={
                        "field_in_a":
                            ast_spec_inference.FieldObservations(
                                count_one=1, count_many=2),
                        "field_in_b":
                            ast_spec_inference.FieldObservations(
                                count_one=2, count_many=1),
                        "field_in_both":
                            ast_spec_inference.FieldObservations(
                                count_one=13, count_many=24),
                    }),
        })

    self.assertEqual(observations_merged, expected)

  def test_observe_types_and_fields(self):
    tree = generic_ast_graphs.GenericASTNode(
        0, "root", {
            "children": [
                generic_ast_graphs.GenericASTNode(
                    1, "foo", {
                        "a": [generic_ast_graphs.GenericASTNode(12, "bar", {})],
                        "b": [generic_ast_graphs.GenericASTNode(13, "bar", {})],
                        "c": [],
                        "d": [
                            generic_ast_graphs.GenericASTNode(14, "bar", {}),
                            generic_ast_graphs.GenericASTNode(15, "bar", {})
                        ],
                    }),
                generic_ast_graphs.GenericASTNode(
                    2, "foo", {
                        "a": [generic_ast_graphs.GenericASTNode(22, "bar", {})],
                        "b": [],
                        "d": [generic_ast_graphs.GenericASTNode(24, "bar", {})],
                    })
            ]
        })

    observations = ast_spec_inference.observe_types_and_fields(tree)
    expected = ast_spec_inference.ASTObservations(
        example_count=1,
        node_types={
            "root":
                ast_spec_inference.NodeObservations(
                    count=1,
                    count_root=1,
                    fields={
                        "children":
                            ast_spec_inference.FieldObservations(count_many=1)
                    }),
            "foo":
                ast_spec_inference.NodeObservations(
                    count=2,
                    count_root=0,
                    fields={
                        "a":
                            ast_spec_inference.FieldObservations(count_one=2),
                        "b":
                            ast_spec_inference.FieldObservations(count_one=1),
                        "c":
                            ast_spec_inference.FieldObservations(),
                        "d":
                            ast_spec_inference.FieldObservations(
                                count_one=1, count_many=1),
                    }),
            "bar":
                ast_spec_inference.NodeObservations(
                    count=6, count_root=0, fields={}),
        })

    self.assertEqual(observations, expected)

  def test_infer_ast_spec(self):
    observations = ast_spec_inference.ASTObservations(
        example_count=1,
        node_types={
            "root":
                ast_spec_inference.NodeObservations(
                    count=10,
                    count_root=10,
                    fields={
                        "nonempty_sequence":
                            ast_spec_inference.FieldObservations(
                                count_one=2, count_many=8),
                        "one_child":
                            ast_spec_inference.FieldObservations(count_one=10),
                    }),
            "foo":
                ast_spec_inference.NodeObservations(
                    count=20,
                    count_root=0,
                    fields={
                        "optional_child":
                            ast_spec_inference.FieldObservations(count_one=15),
                        "sequence":
                            ast_spec_inference.FieldObservations(
                                count_many=4, count_one=4),
                        "no_children":
                            ast_spec_inference.FieldObservations(),
                    }),
        })

    spec = ast_spec_inference.infer_ast_spec(observations)
    expected = {
        "root":
            generic_ast_graphs.ASTNodeSpec(
                fields={
                    "nonempty_sequence":
                        generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE,
                    "one_child":
                        generic_ast_graphs.FieldType.ONE_CHILD
                },
                sequence_item_types={
                    "nonempty_sequence": "root_nonempty_sequence"
                },
                has_parent=False),
        "foo":
            generic_ast_graphs.ASTNodeSpec(
                fields={
                    "optional_child":
                        generic_ast_graphs.FieldType.OPTIONAL_CHILD,
                    "sequence":
                        generic_ast_graphs.FieldType.SEQUENCE,
                    "no_children":
                        generic_ast_graphs.FieldType.NO_CHILDREN
                },
                sequence_item_types={"sequence": "foo_sequence"},
                has_parent=True)
    }
    self.assertEqual(spec, expected)


if __name__ == "__main__":
  absltest.main()
