# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Infers an automaton schema from examples.

This makes it possible to adapt a schema automatically for a dataset, by
implementing only a small translation layer.

This was used to extract a schema from the ETH Py150 Open training set.

We assume there is some set of node types, each of which has a fixed set of
fields. We classify each field as having:

a) always exactly one child
b) either zero or one child
c) a nonempty list of children
d) a sometimes-empty list of children
e) never any children

Based on this classification, we then generate a schema encoding all possible
entry and exit points for an automaton.
"""

from typing import Any, Dict
import dataclasses

from gfsa import generic_ast_graphs
from gfsa import jax_util


def _dict_sum(d1, d2):
  d = d1.copy()
  for k, v in d2.items():
    if k in d:
      d[k] = d[k] + v
    else:
      d[k] = v
  return d


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class FieldObservations:
  """Stores information about a single field.

  Attributes:
    count_one: How many nodes have exactly one child for this field.
    count_many: How many nodes have more than one child for this field.
  """
  count_one: int = 0
  count_many: int = 0

  def __add__(self, other):
    """Combines observations."""
    return FieldObservations(
        count_one=self.count_one + other.count_one,
        count_many=self.count_many + other.count_many)


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class NodeObservations:
  """Stores information about relationships for a specific node type.

  Attributes:
    count: How many times we have seen a node of this type.
    count_root: How many times we have seen this node as the root node (with no
      parent).
    fields: Observations for each relationship we have seen.
  """
  count: int = 0
  count_root: int = 0
  fields: Dict[str, FieldObservations] = dataclasses.field(default_factory=dict)

  def __add__(self, other):
    """Combines observations."""
    return NodeObservations(
        count=self.count + other.count,
        count_root=self.count_root + other.count_root,
        fields=_dict_sum(self.fields, other.fields))


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class ASTObservations:
  """Stores information for building an AST spec.

  Attributes:
    example_count: How many examples we have seen.
    node_types: Mapping from node type to our observations for that node type.
  """
  example_count: int = 0
  node_types: Dict[str,
                   NodeObservations] = dataclasses.field(default_factory=dict)

  def __add__(self, other):
    """Combines observations."""
    return ASTObservations(
        example_count=self.example_count + other.example_count,
        node_types=_dict_sum(self.node_types, other.node_types))


def observe_types_and_fields(
    root):
  """Build observations for a tree.

  Args:
    root: The root node of a tree.

  Returns:
    Collection of observations for that tree.
  """
  observations = ASTObservations(example_count=1)
  observations.node_types[root.node_type] = NodeObservations(count_root=1)
  pending = [root]
  while pending:
    node = pending.pop()
    if node.node_type not in observations.node_types:
      observations.node_types[node.node_type] = NodeObservations()
    node_obs = observations.node_types[node.node_type]
    node_obs.count += 1
    for field, children in node.fields.items():
      if field not in node_obs.fields:
        node_obs.fields[field] = FieldObservations()
      if len(children) == 1:
        node_obs.fields[field].count_one += 1
      elif len(children) > 1:
        node_obs.fields[field].count_many += 1
      pending.extend(children)

  return observations


def summarize_observations(observations):
  """Prints a summary of the observations."""
  print(f"example_count: {observations.example_count}")
  for node_type, node_obs in observations.node_types.items():
    print(node_type)
    nct = node_obs.count
    print(f"  count: {nct} " f"({nct / observations.example_count:.2g})")
    print(f"  count_root: {node_obs.count_root} "
          f"({node_obs.count_root / observations.example_count:.2g})")
    for field, field_obs in node_obs.fields.items():
      count_zero = nct - field_obs.count_one - field_obs.count_many
      print(f"  {field}: zero {count_zero} ({count_zero / nct:.2g}),"
            f" one {field_obs.count_one} ({field_obs.count_one / nct:.2g}),"
            f" many {field_obs.count_many} ({field_obs.count_many / nct:.2g})")


def infer_ast_spec(observations):
  """Infers an AST spec compatible with all observations.

  Since nonterminal information is not available when looking at only concrete
  ASTs, we use a separate sequence helper node for each field of each node type.

  Args:
    observations: Observations collected from examples.

  Returns:
    An AST specification compatible with the observations.

  Raises:
    ValueError: If an AST spec could not be generated due to conflicting
      observations.
  """
  ast_spec = {}
  for node_type, node_obs in observations.node_types.items():
    node_spec = generic_ast_graphs.ASTNodeSpec()

    if node_obs.count_root == 0:
      node_spec.has_parent = True
    elif node_obs.count_root == node_obs.count:
      node_spec.has_parent = False
    else:
      raise ValueError(f"Node type {node_type} appears both in root and child "
                       "position, which is not supported.")

    for field, field_obs in node_obs.fields.items():
      count_zero = node_obs.count - field_obs.count_one - field_obs.count_many

      if field_obs.count_many:
        if count_zero:
          field_type = generic_ast_graphs.FieldType.SEQUENCE
        else:
          field_type = generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE

        node_spec.sequence_item_types[field] = f"{node_type}_{field}"
      elif field_obs.count_one:
        if count_zero:
          field_type = generic_ast_graphs.FieldType.OPTIONAL_CHILD
        else:
          field_type = generic_ast_graphs.FieldType.ONE_CHILD
      else:
        field_type = generic_ast_graphs.FieldType.NO_CHILDREN

      node_spec.fields[field] = field_type
    ast_spec[node_type] = node_spec
  return ast_spec
