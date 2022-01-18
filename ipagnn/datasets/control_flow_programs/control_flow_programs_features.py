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

"""Feature connectors for the control flow programs dataset."""

from absl import logging  # pylint: disable=unused-import
import astunparse
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from ipagnn.datasets.control_flow_programs import control_flow_graph_feature
from ipagnn.datasets.control_flow_programs import control_flow_programs_version
from ipagnn.datasets.control_flow_programs import python_interpreter
from ipagnn.datasets.control_flow_programs import python_interpreter_trace
from ipagnn.datasets.control_flow_programs import python_programs
from ipagnn.datasets.control_flow_programs.encoders import encoders
from ipagnn.toolkit import shepherds as shepherds_lib

NO_OUTPUT = 0


def get_features_dict(feature_set_names,
                      program_encoder, state_encoder, branch_encoder,
                      text_encoder):
  """Returns the features dict for the requested feature sets."""
  statement_features = {
      "statements": tfds.features.Text(encoder=program_encoder),
      "length": tfds.features.Tensor(shape=tuple(), dtype=tf.int32),
      "num_statements": tfds.features.Tensor(shape=tuple(), dtype=tf.int32),
      "intermediate_outputs": tfds.features.Text(encoder=state_encoder),
      "intermediate_outputs_mask": tfds.features.Sequence(
          tfds.features.Tensor(shape=tuple(), dtype=tf.bool),
      ),
      "intermediate_output_lengths": tfds.features.Sequence(
          tfds.features.Tensor(shape=tuple(), dtype=tf.int32),
      ),
      "intermediate_outputs_count": tfds.features.Tensor(shape=tuple(),
                                                         dtype=tf.int32),
      "branch_decisions": tfds.features.Text(encoder=branch_encoder),
      "branch_decisions_mask": tfds.features.Sequence(
          tfds.features.Tensor(shape=tuple(), dtype=tf.bool),
      ),
      "branch_decisions_count": tfds.features.Tensor(shape=tuple(),
                                                     dtype=tf.int32),
  }
  feature_sets = dict(
      human_readable={
          "human_readable_code": tfds.features.Text(),
          "human_readable_target_output": tfds.features.Text(),
          # TODO(dbieber): Enable for the next version of the dataset.
          # # Used for partial programs.
          # "original_human_readable_code": tfds.features.Text(),
      },
      code={
          "code_" + key: value
          for key, value in statement_features.items()
      },
      trace={
          "trace_" + key: value
          for key, value in statement_features.items()
      },
      output={
          "target_output":
              tfds.features.Text(encoder=state_encoder),
          "target_output_length":
              tfds.features.Tensor(shape=tuple(), dtype=tf.int32),
          "lm_text":
              tfds.features.Text(encoder=text_encoder),
      },
      cfg={
          "cfg": control_flow_graph_feature.ControlFlowGraphFeature(
              include_back_edges=True, encoder=program_encoder),
          "cfg_forward": control_flow_graph_feature.ControlFlowGraphFeature(
              include_back_edges=False, encoder=program_encoder),
      },
  )
  if control_flow_programs_version.at_least("0.0.52"):
    names = (
        [
            "NoError",
            "RuntimeError",  # 1 second timeout
            "ZeroDivisionError",
            "AssertionError",
            "ValueError",
            "TypeError",
            "IndexError",
            "NameError",
        ]
        + (
            ["AttributeError"]
            if control_flow_programs_version.at_least("0.0.57")
            else []
        )
        + [
            "RecursionError",
            "MemoryError",
        ]
    )
    feature_sets["output"]["error_type"] = tfds.features.ClassLabel(names=names)

  features = {}
  for feature_set_name in feature_set_names:
    features.update(feature_sets[feature_set_name])
  return tfds.features.FeaturesDict(features)


def get_shepherds(program_generator_config, feature_set_names):
  """Gets the shepherds to use for supergraph-batching the features."""
  del program_generator_config  # Unused.
  shepherd_sets = dict(
      human_readable=[
          shepherds_lib.DenseTensorShepherd("human_readable_code",
                                            dtype=tf.string, element_shape=[]),
          shepherds_lib.DenseTensorShepherd("human_readable_target_output",
                                            dtype=tf.string, element_shape=[]),
          # TODO(dbieber): Enable for the next version of the dataset.
          # shepherds_lib.DenseTensorShepherd("original_human_readable_code",
          # dtype=tf.string, element_shape=[]),
      ],
      code=[],
      trace=[],
      output=[
          shepherds_lib.DenseTensorShepherd("target_output", dtype=tf.int64,
                                            expand_dims=-1),
          shepherds_lib.DenseTensorShepherd("target_output_length",
                                            dtype=tf.int32, element_shape=[]),
      ],
      cfg=[],
  )
  shepherds = []
  for feature_set_name in feature_set_names:
    shepherds.extend(shepherd_sets[feature_set_name])
  return shepherds


def build_representation(python_source, values_lists, branch_decisions_lists,
                         tokens_per_statement, base, target_output_length,
                         output_mod):
  """Builds a partial example_dict representation of the already run source."""
  intermediate_outputs = []
  intermediate_outputs_mask = []
  branch_decisions = []
  branch_decisions_mask = []
  for values_list, branch_decisions_list in zip(
      values_lists, branch_decisions_lists):
    # We select the most recent value of each statement for the code model's
    # intermediate values, to be used by the model for auxiliary losses.
    # For the trace representation, there only is one value per statement since
    # each statement has been run exactly once.
    if values_list:
      assert branch_decisions_list
      statement_output = values_list[-1]["v0"]
      if output_mod is not None:
        try:
          statement_output %= output_mod
        except TypeError:
          statement_output = 1
      statement_output_list = encoders.as_nary_list(
          statement_output, base, target_output_length)
      statement_output_mask = [True] * target_output_length
      branch_decision = branch_decisions_list[-1]
    else:
      assert not branch_decisions_list
      statement_output = NO_OUTPUT
      statement_output_list = [NO_OUTPUT] * target_output_length
      statement_output_mask = [False] * target_output_length
      branch_decision = python_interpreter_trace.NO_BRANCH_DECISION

    padding = tokens_per_statement - target_output_length
    intermediate_outputs.extend([NO_OUTPUT] * padding + statement_output_list)
    intermediate_outputs_mask.extend([False] * padding + statement_output_mask)
    branch_decisions.extend(
        [0] * (tokens_per_statement - 1) + [branch_decision])
    branch_decisions_mask.extend(
        [False] * (tokens_per_statement - 1)
        + [branch_decision is not python_interpreter_trace.NO_BRANCH_DECISION])
  intermediate_outputs_count = len(intermediate_outputs)
  intermediate_output_lengths = [1] * intermediate_outputs_count
  return {
      "statements": python_source,
      "length": tokens_per_statement * len(python_source.split("\n")),
      "num_statements": len(python_source.split("\n")),
      "intermediate_outputs": intermediate_outputs,
      "intermediate_outputs_mask": intermediate_outputs_mask,
      "intermediate_output_lengths": intermediate_output_lengths,
      "intermediate_outputs_count": intermediate_outputs_count,
      "branch_decisions": branch_decisions,
      "branch_decisions_count": len(branch_decisions),
      "branch_decisions_mask": branch_decisions_mask,
  }


# TODO(dbieber): Refactor signature to be:
# def generate_example_from_python_object(python_object, info):
# def generate_example_from_python_object(python_object, executor, info):
# Original:
# Args:
#   python_object: Either a string representing Python source, or a tuple of the
#     form (python_source, partial_python_source) where partial_python_source
#     has a single line replaced with a placeholder.
#   executor: A python_interpreter Executor object.
#   info: The Dataset Info object.
def generate_example_from_python_object(executor, base, python_object,
                                        tokens_per_statement,
                                        target_output_length,
                                        mod,
                                        output_mod):
  """Generates an example dict from the program given by `python_object`.

  Args:
    executor: A python_interpreter Executor object.
    base: The base in which numbers are represented.
    python_object: Either a string representing Python source, or a tuple of the
      form (python_source, partial_python_source) where partial_python_source
      has a single line replaced with a placeholder.
    tokens_per_statement: The number of tokens to use to represent a statement
      in the encoded program.
    target_output_length: The length of the program output, measured in tokens.
    mod: The value (if any) to mod the intermediate values of the program by
      after each step of execution.
    output_mod: The value (if any) to mod the final values of the program by.
  Returns:
    An example dictionary.
  """
  # base = info.program_generator_config.base
  # tokens_per_statement = info.program_encoder.tokens_per_statement
  # target_output_length = info.program_generator_config.num_digits
  # output_mod = None
  # try:
  #   output_mod = info.program_generator_config.output_mod
  # except:
  #   pass
  if isinstance(python_object, tuple):
    # Generate example and partial-example.
    python_source, partial_python_source = python_object
    example = _generate_example_from_python_source(
        executor, base, python_source, tokens_per_statement,
        target_output_length, mod, output_mod)
    partial_example = _generate_example_from_python_source(
        executor, base, partial_python_source, tokens_per_statement,
        target_output_length, mod, output_mod)
    # TODO(dbieber): Use a more general method for listing output fields.
    if control_flow_programs_version.at_least("0.0.52"):
      partial_example["error_type"] = example["error_type"]
    partial_example["target_output"] = example["target_output"]
    partial_example["target_output_length"] = example["target_output_length"]
    # partial_example["original_human_readable_code"] = (
    #     example["human_readable_code"])
    partial_example["human_readable_target_output"] = (
        example["human_readable_target_output"])
    return partial_example
  else:
    # Just generate a full example.
    python_source = python_object
    example = _generate_example_from_python_source(
        executor, base, python_source, tokens_per_statement,
        target_output_length, mod, output_mod)
    # example["original_human_readable_code"] = "N/A"
    return example


def _generate_example_from_python_source(executor, base, python_source,
                                         tokens_per_statement,
                                         target_output_length,
                                         mod,
                                         output_mod):
  """Generates an example dict from the given statements."""
  human_readable_code = python_source
  cfg = python_programs.to_cfg(python_source)
  python_source_lines = python_source.strip().split("\n")

  # TODO(dbieber): This should occur in exactly one location.
  # (also in environment.py)
  values = {"v0": 1}
  trace_fn = python_interpreter_trace.make_trace_fn(python_source, cfg)
  # TODO(dbieber): Evaluating may have already occurred in environment.
  try:
    values = python_interpreter.evaluate_cfg(
        executor, cfg, mod=mod,
        initial_values=values, trace_fn=trace_fn,
        timeout=200)
    error_type = "NoError"
  except Exception as e:  # pylint: disable=broad-except
    error_type = type(e).__name__
  target_output = values["v0"]

  if output_mod is not None:
    try:
      target_output %= output_mod
    except TypeError:
      target_output = 1

  code_features = build_representation(
      python_source, trace_fn.trace.cfg_node_index_values,
      trace_fn.trace.cfg_node_index_branch_decisions,
      tokens_per_statement, base, target_output_length, output_mod)

  use_full_lines_in_trace = False
  if use_full_lines_in_trace:
    trace_lines = [
        python_source_lines[line_index]
        for line_index in trace_fn.trace.trace_line_indexes
    ]
    trace_python_source = "\n".join(trace_lines)
  else:
    trace_control_flow_nodes = [
        cfg.nodes[cfg_node_index]
        for cfg_node_index in trace_fn.trace.trace_cfg_node_indexes
    ]
    # TODO(dbieber): This also occurs in environment `state_as_example`.
    # Refactor.
    python_source_lines = []
    for control_flow_node in trace_control_flow_nodes:
      ast_node = control_flow_node.instruction.node
      python_source_line = astunparse.unparse(ast_node, version_info=(3, 5))
      python_source_line = python_source_line.strip()
      python_source_lines.append(python_source_line)
    trace_python_source = "\n".join(python_source_lines)
  trace_features = build_representation(
      trace_python_source, trace_fn.trace.trace_values,
      trace_fn.trace.trace_branch_decisions,
      tokens_per_statement, base, target_output_length, output_mod)

  target_output_list = encoders.as_nary_list(
      target_output, base, target_output_length)

  lm_text = f"{human_readable_code} SEP {target_output}"

  example_dict = {
      # human_readable_features
      "human_readable_code": human_readable_code,
      # "original_human_readable_code": human_readable_code,
      "human_readable_target_output": str(target_output),

      # target_output
      "target_output": target_output_list,
      "target_output_length": target_output_length,
      "lm_text": lm_text,
      "error_type": error_type,

      # control flow graph
      "cfg": (cfg, python_source),
      "cfg_forward": (cfg, python_source),
  }
  example_dict.update({
      "code_" + key: value
      for key, value in code_features.items()
  })
  example_dict.update({
      "trace_" + key: value
      for key, value in trace_features.items()
  })
  return example_dict
