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

"""tensorflow/datasets ControlFlowPrograms dataset."""

import itertools
import sys

from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from ipagnn.common import dataset_builder
from ipagnn.datasets import info
from ipagnn.datasets.control_flow_programs import control_flow_programs_configs
from ipagnn.datasets.control_flow_programs import control_flow_programs_features
from ipagnn.datasets.control_flow_programs import python_interpreter
from ipagnn.datasets.control_flow_programs.encoders import encoders
from ipagnn.datasets.control_flow_programs.program_generators import program_generators

CITATION = r"""Internal dataset."""


def _filter_features(example, keys):
  """Filters out features that aren't included in the dataset."""
  return {
      key: example[key] for key in keys
  }


class ExampleGenerator(object):
  """Generates ControlFlowProgram examples."""

  def __init__(self, program_generator_config, partial, program_encoder):
    self.program_generator_config = program_generator_config
    self.partial = partial
    self.executor = python_interpreter.ExecExecutor()
    self.program_encoder = program_encoder

  def generate_k_examples(self, split, k):
    logging.info("Generating k examples.")
    for example in itertools.islice(self.generate_all_examples(split), k):
      yield example

  def generate_all_examples(self, split):
    del split  # Unused.
    while True:
      yield self.generate_example(self.program_generator_config.length)

  def generate_example(self, length):
    python_object = self.generate_python_object(length)
    return self.generate_example_from_python_object(python_object)

  def generate_python_object(self, length):
    if self.partial:
      return program_generators.generate_python_source_and_partial_python_source(
          length, self.program_generator_config)
    else:
      return program_generators.generate_python_source(
          length, self.program_generator_config)

  def generate_example_from_python_object(self, python_object):
    tokens_per_statement = self.program_encoder.tokens_per_statement
    return control_flow_programs_features.generate_example_from_python_object(
        self.executor,
        self.program_generator_config.base,
        python_object,
        tokens_per_statement,
        target_output_length=self.program_generator_config.num_digits,
        mod=self.program_generator_config.mod,
        output_mod=self.program_generator_config.output_mod)


class ControlFlowPrograms(tfds.core.BeamBasedBuilder,
                          dataset_builder.DatasetBuilder):
  """ControlFlowPrograms dataset."""

  BUILDER_CONFIGS = control_flow_programs_configs.get_builder_configs()

  def __init__(self, *args, **kwargs):
    self.representation = None
    super(ControlFlowPrograms, self).__init__(*args, **kwargs)

  def _features(self):
    """Returns the features dict for creating the dataset Info."""
    return control_flow_programs_features.get_features_dict(
        self.builder_config.feature_sets,
        self.program_encoder, self.state_encoder, self.branch_encoder,
        self.text_encoder)

  def _shepherds(self):
    """Returns the shepherds for each of the supergraph batched features."""
    # Some shepherds are defined on the feature connectors.
    shepherds, roots = super(ControlFlowPrograms, self)._shepherds()

    # Others are defined in control_flow_programs_features.
    program_generator_config = self.builder_config.program_generator_config
    additional_shepherds = control_flow_programs_features.get_shepherds(
        program_generator_config, self.builder_config.feature_sets)
    shepherds.extend(additional_shepherds)
    roots.extend([None] * len(additional_shepherds))
    return shepherds, roots

  def _info(self):
    """`tfds.core.DatasetInfo` for this builder."""
    program_generator_config = self.builder_config.program_generator_config
    self.program_encoder = (
        encoders.get_program_encoder(program_generator_config))
    self.state_encoder = (
        encoders.PassThruEncoder(vocab_size=program_generator_config.base))
    self.branch_encoder = encoders.PassThruEncoder(vocab_size=3)

    self.example_generator = ExampleGenerator(
        self.builder_config.program_generator_config,
        self.builder_config.partial_program,
        self.program_encoder)

    def corpus_generator_fn():
      for example in self.example_generator.generate_k_examples(
          split="train", k=20):
        yield example["lm_text"]

    self.text_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus_generator_fn(), target_vocab_size=2**10,
        max_subword_length=6,
    )

    # Decide on the input and target keys.
    inputs_key = "code_statements"
    target_key = "target_output"
    if "multivar-templates" in self.builder_config.name:
      target_key = "error_type"

    max_diameter = program_generator_config.length * 2
    return info.LearnedInterpretersDatasetInfo(
        builder=self,
        description="Data for ControlFlowPrograms dataset.",
        features=self._features(),
        supervised_keys=(inputs_key, target_key),
        citation=CITATION,

        # ControlFlowPrograms specific data:
        builder_config=self.builder_config,
        max_diameter=max_diameter,
        program_generator_config=program_generator_config,
        program_encoder=self.program_encoder,
        state_encoder=self.state_encoder,
        branch_encoder=self.branch_encoder,
    )

  def _split_generators(self, dl_manager):
    """Specify dataset splits, setting up calls to _generate_examples.

    This is the first entrypoint for tfds's download_and_prepare function.

    Args:
      dl_manager: (DownloadManager) Download manager to download the data.

    Returns:
      `list<tfds.core.SplitGenerator>`.
    """
    sys.setrecursionlimit(10000)
    del dl_manager  # Unused.
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split("train"),
            gen_kwargs={"split": "train"}),
    ]

  def generate_example_from_python_object(self, python_object):
    """Entrypoint used by reinforce for generating examples."""
    return self.example_generator.generate_example_from_python_object(
        python_object)

  def generate_python_object_from_string(self, string):
    """Entrypoint for dataset_utils for generating examples interactively.

    Args:
      string: A string, with no newlines; the input from the user when
        generating an environment in interactive mode.
    Returns:
      The python_object for a single example.
    """
    # We cannot use new lines in interactive mode currently, so the user must
    # separate Python lines with semicolons. The user may also just enter an
    # integer and we will generate a random program of that length.

    # Syntax for partial programs: HOLE=<hole statement index>;<full program>
    if string.startswith("HOLE="):
      hole_statement_index_string, remainder = string.split(";", 1)
      full_program = remainder.split(";")
      hole_statement_index = int(hole_statement_index_string.split("=")[-1])
      original_statement = full_program[hole_statement_index]
      original_indent = (len(original_statement)
                         - len(original_statement.lstrip()))
      partial_program = full_program.copy()
      partial_program[hole_statement_index] = " " * original_indent + "_ = 0"
      return ("\n".join(full_program), "\n".join(partial_program))
    try:
      length = int(string)
      return self.example_generator.generate_python_object(length)
    except ValueError:
      return "\n".join(string.split(";"))

  def generate_example_from_string(self, string):
    """Entrypoint for dataset_utils for generating examples interactively."""
    python_object = self.generate_python_object_from_string(string)
    return self.example_generator.generate_example_from_python_object(
        python_object)

  def _build_pcollection(self, pipeline, split):
    """The final entrypoint for generation of a single split."""
    beam = tfds.core.lazy_imports.apache_beam

    keys = list(self.info.features.keys())
    programs_per_shard = 1000
    shards = max(1, int(self.builder_config.max_examples / programs_per_shard))
    split_sizes = [int(self.builder_config.max_examples / shards)] * shards
    example_generator = self.example_generator

    def _generate_k_examples(k):
      for example in example_generator.generate_k_examples(split, k):
        hash_key = hash(example["human_readable_code"])
        yield hash_key, _filter_features(example, keys)

    return (
        pipeline
        | "SplitSizes" >> beam.Create(split_sizes)
        | "GenerateExamples" >> beam.FlatMap(_generate_k_examples)
        | "RemoveDuplicates" >> beam.CombinePerKey(lambda vs: next(iter(vs)))
    )

  def as_in_memory_dataset(self, split=None):
    """Constructs a tf.data.Dataset object using an in-memory generator.

    This method can be used for curriculum learning. The difficulty of the task
    can be changed by the training algorithm during training. The learning
    algorithm may call set_task in order to change the parameters used in data
    generation.

    The data is not written to disk.

    The following example shows how to use this to advance the difficulty of
    the examples every time the loss drops below a threshold.

    Example:
      dataset, set_task = arithmetic.as_in_memory_dataset()
      length = 1
      set_task(length=length)
      for example in dataset:
        loss = model(example)
        if loss < 0.05:
          length += 1
          set_task(length=length)

    Args:
      split: The split of the dataset to generate. Unused.
    Returns:
      dataset: A tf.data.Dataset object that generates an unbounded number of
        examples.
      set_task: A function that allows the learning algorithm to update the
        parameters used for data generation. See the set_task inner function
        docstring for details.
    """
    keys = self.info.features.keys()

    def generator_fn():
      """Generates examples in memory for the cfp dataset."""
      while True:
        if generator_fn.task_fn is not None:
          length = generator_fn.task_fn()
        else:
          length = generator_fn.length
        example = self.example_generator.generate_example(length)
        example = _filter_features(example, keys)
        encoded_example = self.info.features.encode_example(example)
        yield self.info.features.decode_example(encoded_example)

    def set_task(task_fn=None, length=None):
      """Updates the parameters the dataset is using for example generation.

      If task_fn is set, it will be used for each example to determine the
      num_digits and nesting params. Otherwise, the constant `length` will be
      used.

      Args:
        task_fn: Function of no arguments that returns (num_digits, nesting).
        length: If task_fn is not set, the length param for example generation.
      """
      generator_fn.task_fn = task_fn
      if length is not None:
        generator_fn.length = length

    set_task(length=self.builder_config.program_generator_config.length)

    dtype = self.info.features.dtype
    shape = self.info.features.shape
    dataset = tf.data.Dataset.from_generator(generator_fn, dtype, shape)
    dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, set_task

  def as_python_dataset(self, split=None):

    def generator_fn():
      """Generates object representations of examples, not encoded as Tensors.

      If generate_example_from_python_object is applied, then the info.features
      feature connector can be used to encode/decode a single output as an
      example.

      Yields:
        Each yielded output is Python object representing a single example.
      """
      while True:
        if generator_fn.task_fn is not None:
          length = generator_fn.task_fn()
        else:
          length = generator_fn.length
        python_object = self.example_generator.generate_python_object(length)
        yield python_object

    def set_task(task_fn=None, length=None):
      """Updates the parameters the dataset is using for example generation.

      If task_fn is set, it will be used for each example to determine the
      num_digits and nesting params. Otherwise, the constant `length` will be
      used.

      Args:
        task_fn: Function of no arguments that returns (num_digits, nesting).
        length: If task_fn is not set, the length param for example generation.
      """
      generator_fn.task_fn = task_fn
      if length is not None:
        generator_fn.length = length

    set_task(length=self.builder_config.program_generator_config.length)
    generator = generator_fn()
    return generator, set_task

  def key(self, key):
    if not self.representation:
      logging.warn("No representation set. Using default (code).")
      self.representation = "code"
    assert self.representation in ("code", "trace")

    if key in [
        "statements",
        "length",
        "num_statements",
        "intermediate_outputs",
        "intermediate_outputs_mask",
        "intermediate_output_lengths",
        "intermediate_outputs_count",
        "branch_decisions",
        "branch_decisions_count",
        "branch_decisions_mask",
    ]:
      return self.representation + "_" + key
    else:
      return key
