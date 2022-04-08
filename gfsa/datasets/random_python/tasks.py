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

"""AST traversal tasks on random python code."""

import functools
from typing import Callable, Optional, Union

import astunparse
import dataclasses
import gast
import numpy as np

from gfsa import automaton_builder
from gfsa import py_ast_graphs
from gfsa.datasets import graph_bundle
from gfsa.datasets import graph_edge_util
from gfsa.datasets.random_python import python_numbers_control_flow
from gfsa.datasets.random_python import top_down_refinement

EDGE_TYPES = sorted({
    graph_edge_util.JUMPS_OUT_OF_EDGE_TYPE,
    graph_edge_util.SAME_IDENTIFIER_EDGE_TYPE,
    *graph_edge_util.PROGRAM_GRAPH_EDGE_TYPES,
    *graph_edge_util.schema_edge_types(py_ast_graphs.SCHEMA, False),
    *graph_edge_util.schema_edge_types(py_ast_graphs.SCHEMA, True),
})


def _make_arguments(*args):
  """Returns a gast arguments node with these argument nodes."""
  return gast.arguments(
      args=list(args),
      posonlyargs=[],
      vararg=None,
      kwonlyargs=[],
      kw_defaults=[],
      kwarg=None,
      defaults=[])


# Allow either a single deterministic distribution, or a randomly generated
# per-example subtree distribution.
RefinementDistnOrMetaDistn = Union[
    top_down_refinement.RefinementDistribution,
    Callable[[np.random.RandomState],
             top_down_refinement.RefinementDistribution]]


def make_ast(
    target_ast_node_count,
    rng = None,
    distribution = (
        python_numbers_control_flow.DATAFLOW_FNS_DISTRIBUTION)
):
  """Generates an AST for this task.

  Args:
    target_ast_node_count: How many nodes to put in the AST.
    rng: Random state to use.
    distribution: Sampling distribution to use when building the AST. May also
      be a callable that produces a distribution given a random state.

  Returns:
    AST of a generated program.
  """

  def root_build(body):
    """Given a list of statements, puts them into a function in a module."""
    return gast.Module(
        body=[
            gast.FunctionDef(
                name="random_function",
                args=_make_arguments(
                    python_numbers_control_flow.make_name("a"),
                    python_numbers_control_flow.make_name("b")),
                body=body,
                decorator_list=[],
                returns=None,
                type_comment=None)
        ],
        type_ignores=[])

  root_template = python_numbers_control_flow.ASTWithHoles(
      cost=5,
      holes=[
          top_down_refinement.Hole(
              python_numbers_control_flow.ASTHoleType.STMTS_NONEMPTY,
              python_numbers_control_flow.ASTHoleMetadata(("a", "b"), True,
                                                          False, 0))
      ],
      build=root_build)

  if rng is None:
    rng = np.random.RandomState()

  if callable(distribution):
    distribution = distribution(rng)

  tree = top_down_refinement.top_down_construct(
      root_object=root_template,
      target_cost=target_ast_node_count,
      refinement_distribution=distribution,
      rng=rng)

  # Re-parse the tree so that it is valid. This is required for program graph
  # analysis to work.
  return gast.parse(astunparse.unparse(gast.gast_to_ast(tree)))


@dataclasses.dataclass
class TaskExampleDistribution:
  """Distribution for generating full examples with task information.

  Attributes:
    target_ast_size: Target number of AST nodes.
    refinement_distribution: Distribution to sample from.
    padding_config: How to pad the generated examples.
  """
  target_ast_size: int
  refinement_distribution: RefinementDistnOrMetaDistn
  padding_config: graph_bundle.PaddingConfig


# Constants calibrated using `padding_calibration.calibrate_padding`.
# The distribution used for training the models in the paper was
# "data_flow_fns", and generalization experiments were conducted on
# "data_flow_fns_doublesize" and "data_flow_fns_halfsize".
# The "control_flow" and "data_flow" distributions are somewhat simpler, and
# were used for initial prototyping.
DISTRIBUTIONS = {
    "control_flow":
        TaskExampleDistribution(
            target_ast_size=159,
            refinement_distribution=(
                python_numbers_control_flow.CFG_DISTRIBUTION),
            padding_config=graph_bundle.PaddingConfig(
                static_max_metadata=automaton_builder.EncodedGraphMetadata(
                    num_nodes=256, num_input_tagged_nodes=512),
                max_initial_transitions=1024,
                max_in_tagged_transitions=2048,
                max_edges=2048)),
    "data_flow":
        TaskExampleDistribution(
            target_ast_size=172,
            refinement_distribution=(
                python_numbers_control_flow.DATAFLOW_DISTRIBUTION),
            padding_config=graph_bundle.PaddingConfig(
                static_max_metadata=automaton_builder.EncodedGraphMetadata(
                    num_nodes=256, num_input_tagged_nodes=512),
                max_initial_transitions=1024,
                max_in_tagged_transitions=2048,
                max_edges=4096)),
    "data_flow_fns":
        TaskExampleDistribution(
            target_ast_size=150,
            refinement_distribution=(
                python_numbers_control_flow.DATAFLOW_FNS_DISTRIBUTION),
            padding_config=graph_bundle.PaddingConfig(
                static_max_metadata=automaton_builder.EncodedGraphMetadata(
                    num_nodes=256, num_input_tagged_nodes=512),
                max_initial_transitions=1024,
                max_in_tagged_transitions=2048,
                max_edges=4096)),
    "data_flow_fns_doublesize":
        TaskExampleDistribution(
            target_ast_size=300,
            refinement_distribution=(
                python_numbers_control_flow.DATAFLOW_FNS_DISTRIBUTION),
            # Padding is double that for data_flow_fns.
            padding_config=graph_bundle.PaddingConfig(
                static_max_metadata=automaton_builder.EncodedGraphMetadata(
                    num_nodes=512, num_input_tagged_nodes=1024),
                max_initial_transitions=2048,
                max_in_tagged_transitions=4096,
                max_edges=8192)),
    "data_flow_fns_halfsize":
        TaskExampleDistribution(
            target_ast_size=75,
            refinement_distribution=(
                python_numbers_control_flow.DATAFLOW_FNS_DISTRIBUTION),
            # Padding is half that for data_flow_fns, except for
            # num_input_tagged_nodes which is larger to avoid dropping too many
            # examples.
            padding_config=graph_bundle.PaddingConfig(
                static_max_metadata=automaton_builder.EncodedGraphMetadata(
                    num_nodes=128, num_input_tagged_nodes=512),
                max_initial_transitions=512,
                max_in_tagged_transitions=1024,
                max_edges=2048)),
    "data_flow_fns_perturbed_weights_0.4_depth_6":
        TaskExampleDistribution(
            # Note: intentionally the same target size as data_flow_fns, to
            # make the comparison easier to describe. This distribution has
            # higher variance, so we may end up dropping more examples due to
            # padding.
            target_ast_size=150,
            refinement_distribution=functools.partial(
                python_numbers_control_flow.make_dataflow_fns_distribution,
                weights_temperature=0.4,
                max_depth_expected=3,
                max_depth_maximum=6),
            padding_config=graph_bundle.PaddingConfig(
                static_max_metadata=automaton_builder.EncodedGraphMetadata(
                    num_nodes=256, num_input_tagged_nodes=512),
                max_initial_transitions=1024,
                max_in_tagged_transitions=2048,
                max_edges=4096)),
}
