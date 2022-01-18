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

# lint as: python3
"""A representation of a small subset of Python as a graph schema.

The goal is to allow representing some simple functions as graphs in order to
learn Python static analyses from the AST. As such, it supports basic control
flow, but intentionally simplifies away many other "optional" parts of the
language (and thus probably won't support real-world code).

This representation was used for the static analysis tasks in the paper.

Some things that are not supported:
- Keyword arguments
- Chained assignments (such as x = y = foo(...))
- Chained comparisons (such as x < y < z)
- Decorators
- Imports, exceptions, "with" blocks, and many other AST node types that
  haven't yet been added in (but could be added if needed)

The graph representation is mostly a straightforward conversion of the AST
produced by the gast module, with a few differences:

- Metadata (i.e. variable names, load/store contexts, etc) is removed (we assume
  relevant metadata will be captured in initial node embeddings, not in the
  graph structure).
- Anywhere that a list of expressions or list of statements would appear,
  special list helper nodes are inserted to track position in the list (this is
  to simplify the schema, since otherwise every statement or expression would
  have to have optional next/previous edges).
"""

from typing import Dict, Tuple

import gast

from gfsa import automaton_builder
from gfsa import generic_ast_graphs
from gfsa import graph_types

PY_AST_SPECS = {
    # Module is the root node, so it does not have a parent
    "Module":
        generic_ast_graphs.ASTNodeSpec(
            has_parent=False,
            fields={"body": generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE},
            sequence_item_types={"body": "Module_body"}),
    # Statement nodes
    "FunctionDef":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "args": generic_ast_graphs.FieldType.ONE_CHILD,
                "returns": generic_ast_graphs.FieldType.OPTIONAL_CHILD,
                "body": generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE,
                "decorator_list": generic_ast_graphs.FieldType.NO_CHILDREN
            },
            sequence_item_types={"body": "FunctionDef_body"}),
    "Return":
        generic_ast_graphs.ASTNodeSpec(
            fields={"value": generic_ast_graphs.FieldType.OPTIONAL_CHILD}),
    "Assign":
        generic_ast_graphs.ASTNodeSpec(
            # Python ASTs use "targets" as the field name because chained
            # assignments can assign to multiple values (e.g. x=y=2). We
            # preserve the name for consistency, but assume there is only one
            # target.
            fields={
                "targets": generic_ast_graphs.FieldType.ONE_CHILD,
                "value": generic_ast_graphs.FieldType.ONE_CHILD
            }),
    "For":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "target": generic_ast_graphs.FieldType.ONE_CHILD,
                "iter": generic_ast_graphs.FieldType.ONE_CHILD,
                "body": generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE,
                # Don't support the (rare, but technically valid) for/else block
                "orelse": generic_ast_graphs.FieldType.NO_CHILDREN,
            },
            sequence_item_types={"body": "For_body"}),
    "While":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "test": generic_ast_graphs.FieldType.ONE_CHILD,
                "body": generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE,
                # Don't support the (rare, but technically valid) while/else
                # block
                "orelse": generic_ast_graphs.FieldType.NO_CHILDREN,
            },
            sequence_item_types={"body": "While_body"}),
    "If":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "test": generic_ast_graphs.FieldType.ONE_CHILD,
                "body": generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE,
                "orelse": generic_ast_graphs.FieldType.SEQUENCE,
            },
            sequence_item_types={
                "body": "If_body",
                "orelse": "If_orelse"
            }),
    "Expr":
        generic_ast_graphs.ASTNodeSpec(
            fields={"value": generic_ast_graphs.FieldType.ONE_CHILD}),
    "Break":
        generic_ast_graphs.ASTNodeSpec(),
    "Continue":
        generic_ast_graphs.ASTNodeSpec(),
    "Pass":
        generic_ast_graphs.ASTNodeSpec(),
    # Expression nodes
    "BinOp":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "left": generic_ast_graphs.FieldType.ONE_CHILD,
                "op": generic_ast_graphs.FieldType.ONE_CHILD,
                "right": generic_ast_graphs.FieldType.ONE_CHILD
            }),
    "BoolOp":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "op": generic_ast_graphs.FieldType.ONE_CHILD,
                "values": generic_ast_graphs.FieldType.NONEMPTY_SEQUENCE,
            },
            sequence_item_types={"values": "BoolOp_values"}),
    "Compare":
        generic_ast_graphs.ASTNodeSpec(
            # Python ASTs use "ops" and "comparators" for Compare nodes because
            # comparisons can be chained (x > y < z <= 3). We preserve the names
            # for consistency, but assume there is only one op and two values
            # being compared.
            fields={
                "left": generic_ast_graphs.FieldType.ONE_CHILD,
                "ops": generic_ast_graphs.FieldType.ONE_CHILD,
                "comparators": generic_ast_graphs.FieldType.ONE_CHILD
            }),
    "Call":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "func": generic_ast_graphs.FieldType.ONE_CHILD,
                "args": generic_ast_graphs.FieldType.SEQUENCE,
                "keywords": generic_ast_graphs.FieldType.NO_CHILDREN,
            },
            sequence_item_types={"args": "Call_args"}),
    "Constant":
        generic_ast_graphs.ASTNodeSpec(),
    "Name":
        generic_ast_graphs.ASTNodeSpec(
            fields={"ctx": generic_ast_graphs.FieldType.IGNORE}),
    # Operator nodes (technically, Python has many subcategories of op, but
    # for simplicity we treat them all identically)
    "And":
        generic_ast_graphs.ASTNodeSpec(),
    "Or":
        generic_ast_graphs.ASTNodeSpec(),
    "Add":
        generic_ast_graphs.ASTNodeSpec(),
    "Sub":
        generic_ast_graphs.ASTNodeSpec(),
    "Mult":
        generic_ast_graphs.ASTNodeSpec(),
    "Div":
        generic_ast_graphs.ASTNodeSpec(),
    "Eq":
        generic_ast_graphs.ASTNodeSpec(),
    "NotEq":
        generic_ast_graphs.ASTNodeSpec(),
    "Lt":
        generic_ast_graphs.ASTNodeSpec(),
    "LtE":
        generic_ast_graphs.ASTNodeSpec(),
    "Gt":
        generic_ast_graphs.ASTNodeSpec(),
    "GtE":
        generic_ast_graphs.ASTNodeSpec(),
    # Arguments node (we assume we only have positional arguments, so the
    # argument sequence could have been folded into the function, but we keep it
    # as a separate node for consistency with the AST node representation).
    "arguments":
        generic_ast_graphs.ASTNodeSpec(
            fields={
                "args": generic_ast_graphs.FieldType.SEQUENCE,
                "posonlyargs": generic_ast_graphs.FieldType.NO_CHILDREN,
                "vararg": generic_ast_graphs.FieldType.NO_CHILDREN,
                "kwonlyargs": generic_ast_graphs.FieldType.NO_CHILDREN,
                "kw_defaults": generic_ast_graphs.FieldType.NO_CHILDREN,
                "kwarg": generic_ast_graphs.FieldType.NO_CHILDREN,
                "defaults": generic_ast_graphs.FieldType.NO_CHILDREN
            },
            sequence_item_types={"args": "arguments_args"}),
}


def py_ast_to_generic(tree):
  """Convert a gast AST node to a generic representation.

  IDs are set based on the python `id` of the AST node. Only children that are
  AST nodes or lists of AST nodes will be processed.

  Args:
    tree: Node of the AST to convert.

  Returns:
    Generic representation of the AST.
  """
  fields = {}
  for field_name in tree._fields:
    value = getattr(tree, field_name)
    if isinstance(value, gast.AST):
      fields[field_name] = [py_ast_to_generic(value)]
    elif isinstance(value, list):
      if value and isinstance(value[0], gast.AST):
        fields[field_name] = [py_ast_to_generic(child) for child in value]
    else:
      # Doesn't contain any AST nodes, so ignore it.
      pass

  return generic_ast_graphs.GenericASTNode(
      node_id=id(tree), node_type=type(tree).__name__, fields=fields)


# Default definitions used elsewhere
SCHEMA = generic_ast_graphs.build_ast_graph_schema(PY_AST_SPECS)
BUILDER = automaton_builder.AutomatonBuilder(SCHEMA)


def py_ast_to_graph(
    tree):
  """Convert an unsimplified AST into a graph, with a forward mapping.

  Args:
    tree: The (unsimplified) AST for the program.

  Returns:
    - Graph representing the tree.
    - Dictionary that maps from AST node ids to graph node ids.
  """
  return generic_ast_graphs.ast_to_graph(
      root=py_ast_to_generic(tree),
      ast_spec=PY_AST_SPECS,
      ignore_unknown_fields=True)
