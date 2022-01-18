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

# Lint as: python3
"""Example structure for the var-misuse task."""

from typing import List, Optional

import dataclasses
import numpy as np
from tensor2tensor.data_generators import text_encoder
from tensorflow.io import gfile

from gfsa import automaton_builder
from gfsa import generic_ast_graphs
from gfsa import graph_types
from gfsa import jax_util
from gfsa import sparse_operator
from gfsa.datasets import graph_bundle
from gfsa.datasets import graph_edge_util

EDGE_NTH_CHILD_MAX = 32


@dataclasses.dataclass
class ExampleEncodingInfo:
  """Keeps track of objects needed to encode and decode examples.

  Attributes:
    ast_spec: AST spec defining how to encode an AST.
    token_encoder: Subword encoder for encoding syntax tokens.
    schema: Automaton schema for the produced graphs. Generated automatically.
    edge_types: List of all edge types produced by the encoding. Generated
      automatically.
    builder: Automaton builder for the produced graphs. Generated automatically.
  """
  # Provided at initialization time.
  ast_spec: generic_ast_graphs.ASTSpec
  token_encoder: text_encoder.SubwordTextEncoder

  # Generated automatically in __post_init__ from `ast_spec`.
  schema: graph_types.GraphSchema = dataclasses.field(init=False)
  edge_types: List[str] = dataclasses.field(init=False)
  builder: automaton_builder.AutomatonBuilder = dataclasses.field(init=False)

  def __post_init__(self):
    """Populates non-init fields based on `ast_spec`."""
    self.schema = generic_ast_graphs.build_ast_graph_schema(self.ast_spec)
    self.edge_types = sorted({
        graph_edge_util.SAME_IDENTIFIER_EDGE_TYPE,
        *graph_edge_util.PROGRAM_GRAPH_EDGE_TYPES,
        *graph_edge_util.schema_edge_types(self.schema),
        *graph_edge_util.nth_child_edge_types(EDGE_NTH_CHILD_MAX),
    })
    self.builder = automaton_builder.AutomatonBuilder(self.schema)

  @classmethod
  def from_files(cls, ast_spec_path,
                 encoder_vocab_path):
    """Builds an ExampleEncodingInfo object from files.

    Args:
      ast_spec_path: Path to a text file containing an AST spec definition.
        Format is expected to be a Python expression for a
        generic_ast_graphs.ASTSpec (as produced by `repr`). (Note that we assume
        that the source is trusted and safe to `eval`.)
      encoder_vocab_path: Path to a text file containing the vocabulary for a
        SubwordTextEncoder.

    Returns:
      A ExampleEncodingInfo populated with the contents of the given files.
    """
    with gfile.GFile(ast_spec_path, "r") as fp:
      ast_spec = eval(fp.read(), generic_ast_graphs.__dict__)  # pylint: disable=eval-used

    token_encoder = text_encoder.SubwordTextEncoder(encoder_vocab_path)

    return ExampleEncodingInfo(ast_spec=ast_spec, token_encoder=token_encoder)


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class GraphBundleWithTokens:
  """Graph bundle that also has a collection of tokens for each node.

  Attributes:
    bundle: Graph bundle representing the graph.
    tokens: Sparse operator mapping from an array of token embeddings to a list
      of nodes. Each node may have an arbitrary number of tokens (including
      zero). The tokens are considered to be unordered, and repeated tokens can
      be represented as entries in `tokens` with values greater than 1.
  """
  bundle: graph_bundle.GraphBundle
  tokens: sparse_operator.SparseCoordOperator


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class GraphBundleWithTokensPaddingConfig:
  """Configuration specifying how examples get padded to a constant shape.

  Attributes:
    bundle_padding: PaddingConfig for the `bundle` attribute.
    max_tokens: Maximum number of entries in the `tokens` operator; in other
      words, the maximum number of unique (node, token) pairs allowed.
  """
  bundle_padding: graph_bundle.PaddingConfig
  max_tokens: int


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class VarMisuseExample:
  """An example for a var misuse problem.

  Attributes:
    input_graph: Graph bundle with token information, containing the buggy
      version of the example.
    bug_node_index: Index of the node that corresponds to the variable misuse,
      or -1 if there is no bug.
    repair_node_mask: <bool[num_nodes]> array with True at locations that
      contain the correct replacement for the misused identifier.
    candidate_node_mask: <bool[num_nodes]> array with True at locations that
      contain an identifier that could be used as a repair target.
    unique_candidate_operator: Sparse operator mapping from nodes to unique
      candidate identifiers. We always pad this to the same length as the number
      of nodes in the graph; this should be fine for any real-world program. The
      identifier 0 is always the "no-bug" sentinel identifier.
    repair_id: ID for the correct repair.
  """
  input_graph: GraphBundleWithTokens
  bug_node_index: jax_util.NDArray
  repair_node_mask: jax_util.NDArray
  candidate_node_mask: jax_util.NDArray
  unique_candidate_operator: sparse_operator.SparseCoordOperator
  repair_id: jax_util.NDArray


def pad_example(example,
                config,
                allow_failure = False):
  """Pads an example so that it has a static shape determined by the config.

  Args:
    example: The example to pad.
    config: Configuration specifying the desired padding size.
    allow_failure: If True, returns None instead of failing if example is too
      large.

  Returns:
    A padded example with static shape.

  Raises:
    ValueError: If the graph is too big to pad to this size.
  """
  if example.input_graph.tokens.values.shape[0] > config.max_tokens:
    if allow_failure:
      return None
    raise ValueError("Example has too many tokens.")

  bundle = graph_bundle.pad_example(example.input_graph.bundle,
                                    config.bundle_padding, allow_failure)
  if bundle is None:
    return None
  return VarMisuseExample(
      input_graph=GraphBundleWithTokens(
          bundle=bundle,
          tokens=example.input_graph.tokens.pad_nonzeros(config.max_tokens),
      ),
      bug_node_index=example.bug_node_index,
      repair_node_mask=jax_util.pad_to(
          example.repair_node_mask,
          config.bundle_padding.static_max_metadata.num_nodes),
      candidate_node_mask=jax_util.pad_to(
          example.candidate_node_mask,
          config.bundle_padding.static_max_metadata.num_nodes),
      unique_candidate_operator=example.unique_candidate_operator.pad_nonzeros(
          config.bundle_padding.static_max_metadata.num_nodes),
      repair_id=example.repair_id)


def zeros_like_padded_example(
    config):
  """Builds a VarMisuseExample containing only zeros.

  This can be useful to initialize model parameters, or do tests.

  Args:
    config: Configuration specifying the desired padding size.

  Returns:
    An "example" filled with zeros of the given size.
  """
  return VarMisuseExample(
      input_graph=GraphBundleWithTokens(
          bundle=graph_bundle.zeros_like_padded_example(config.bundle_padding),
          tokens=sparse_operator.SparseCoordOperator(
              input_indices=np.zeros(
                  shape=(config.max_tokens, 1), dtype=np.int32),
              output_indices=np.zeros(
                  shape=(config.max_tokens, 1), dtype=np.int32),
              values=np.zeros(shape=(config.max_tokens,), dtype=np.int32))),
      bug_node_index=-1,
      repair_node_mask=np.zeros(
          shape=(config.bundle_padding.static_max_metadata.num_nodes,),
          dtype=np.float32),
      candidate_node_mask=np.zeros(
          shape=(config.bundle_padding.static_max_metadata.num_nodes,),
          dtype=np.float32),
      unique_candidate_operator=sparse_operator.SparseCoordOperator(
          input_indices=np.zeros(shape=(config.max_tokens, 1), dtype=np.int32),
          output_indices=np.zeros(shape=(config.max_tokens, 1), dtype=np.int32),
          values=np.zeros(shape=(config.max_tokens,), dtype=np.float32)),
      repair_id=0)
