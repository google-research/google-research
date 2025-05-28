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

"""FeatureNeighborhood input reader."""

import os

import feature_neighborhood_tensor_opts_pb2 as pb_opts
from google.protobuf import text_format

from lingvo import compat as tf
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.core import py_utils

import pynini
import util


def _trim_and_pad_1d(t, length, pad_value):
  """Trim/pad to the first axis of `t` to be of size `length`."""
  t = t[:length]
  pad_amt = length - tf.shape(t)[0]
  padded_t = tf.pad(t, [(0, pad_amt)], constant_values=pad_value)
  padded_t.set_shape([length])
  return padded_t


def _trim_and_pad_2d(t, length_1, length_2, pad_value):
  """Trim/pad to the first and second axess of `t` to the given size."""
  t = t[:length_1, :length_2]
  pad_amt_1 = length_1 - tf.shape(t)[0]
  pad_amt_2 = length_2 - tf.shape(t)[1]
  padded_t = tf.pad(t, [(0, pad_amt_1), (0, pad_amt_2)],
                    constant_values=pad_value)
  return padded_t


# Not needed now, but maybe later
class Error(Exception):
  """Run-time error in FeatureNeighborhood input."""


class FeatureNeighborhoodInput(base_input_generator.BaseSequenceInputGenerator):
  """Batched categorical or frame features and phoneme/grapheme targets."""

  def _DecodeFeatureNeighborhood(self, source_id, record):
    """Decodes single neighborhood.

    Args:
      source_id: Record ID.
      record: A TFRecord item containing serialized sequence example.

    Returns:
      Nested map of the following tensors:
        cognate_id: string tensor
        spelling: int32 tensor of shape [max_spelling_len]
        pronunciation: int32 tensor of shape [max_pronunciation_len]
        neighbor_spellings: int32 tensor of shape
          [max_neighbors, max_spelling_len]
        neighbor_pronunciations: int32 tensor of shape
          [max_neighbors, max_pronunciation_len]
    """
    context_features_spec = {
        "cognate_id": tf.io.FixedLenFeature([], tf.string),
        "main_name": tf.io.VarLenFeature(dtype=tf.int64),
        "main_pron": tf.io.VarLenFeature(dtype=tf.int64),
    }
    sequence_features_spec = {
        "neighbor_names": tf.io.VarLenFeature(dtype=tf.int64),
        "neighbor_prons": tf.io.VarLenFeature(dtype=tf.int64),
    }
    ctx_features, seq_features = tf.io.parse_single_sequence_example(
        record,
        context_features=context_features_spec,
        sequence_features=sequence_features_spec)

    # Densify and convert to small ints.
    main_name = tf.cast(tf.sparse.to_dense(ctx_features["main_name"]), tf.int32)
    main_pron = tf.cast(tf.sparse.to_dense(ctx_features["main_pron"]), tf.int32)
    neighbor_names = tf.cast(tf.sparse.to_dense(
        seq_features["neighbor_names"]), tf.int32)
    neighbor_prons = tf.cast(tf.sparse.to_dense(
        seq_features["neighbor_prons"]), tf.int32)

    # Perform padding.
    p = self.params
    pad_value = p.feature_neighborhood_input.batch_opts.pad_value
    max_spelling_len = p.feature_neighborhood_input.max_spelling_len
    max_pronunciation_len = p.feature_neighborhood_input.max_pronunciation_len
    max_neighbors = p.feature_neighborhood_input.max_neighbors

    main_name = _trim_and_pad_1d(main_name, max_spelling_len, pad_value)
    main_pron = _trim_and_pad_1d(main_pron, max_pronunciation_len, pad_value)
    neighbor_names = _trim_and_pad_2d(
        neighbor_names, max_neighbors, max_spelling_len, pad_value)
    neighbor_prons = _trim_and_pad_2d(
        neighbor_prons, max_neighbors, max_pronunciation_len, pad_value)

    batch_key = 1  # Batch key is always 1.
    example = py_utils.NestedMap(
        cognate_id=ctx_features["cognate_id"],
        main_name=main_name,
        main_pron=main_pron,
        neighbor_names=neighbor_names,
        neighbor_prons=neighbor_prons)
    return example, batch_key

  def _DataSourceFromFilePattern(self, file_pattern, **extra_input_kwargs):
    """Parses FeatureNeighborhoods into bundles of tensors.

    Args:
      file_pattern: Input regex specifying path to the input records.
      **extra_input_kwargs: Extra arguments.

    Returns:
      Nested map of the following tensors:
        cognate_id: string tensor of shape [batch]
        spelling: int32 tensor of shape [batch, max_spelling_len]
        pronunciation: int32 tensor of shape [batch, max_pronunciation_len]
        neighbor_spellings: int32 tensor of shape
          [batch, max_neighbors, max_spelling_len]
        neighbor_pronunciations: int32 tensor of shape
          [batch, max_neighbors, max_pronunciation_len]
    """
    example, _ = generic_input.GenericInput(
        file_pattern=self.params.file_pattern,
        processor=self._DecodeFeatureNeighborhood,
        **self.CommonInputOpArgs())

    p = self.params
    batch = py_utils.NestedMap()
    batch.cognate_id = example.cognate_id
    batch.spelling = example.main_name
    batch.pronunciation = example.main_pron
    if p.use_neighbors:
      batch.neighbor_spellings = example.neighbor_names
      batch.neighbor_pronunciations = example.neighbor_prons
    return batch

  @classmethod
  def Params(cls):
    """Parameter definitions and defaults."""
    p = super().Params()
    p.Define(
        "feature_neighborhood_input",
        util.Util.CreateParamsForMessage(
            pb_opts.FeatureNeighborhoodTensorOpts.DESCRIPTOR),
        "FeatureNeighborhood input config matching "
        "the FeatureNeighborhoodTensorOpts proto message")
    p.Define("batch_size", 2, "Batch size")
    p.Define("use_neighbors", True,
             "Whether or not to pass through neighbor data")
    p.Define("eval_mode", False, "Set up input pipeline for evaluation.")
    return p

  def __init__(self, params):
    super().__init__(params)
    iargs = self.CommonInputOpArgs()
    if self.params.eval_mode:
      iargs["repeat_count"] = 1

  def _PreprocessInputBatch(self, batch):
    """Preprocesses input batch from _InputBatch.

    Args:
      batch: A NestedMap (or list of NestedMaps when using TPU sharded infeed)
        containing input tensors in the format returned by _InputBatch.

    Returns:
      A NestedMap containing preprocessed inputs to feed to the model.
    """
    p = self.params
    batch_size = self.InfeedBatchSize()

    max_spelling_len = p.feature_neighborhood_input.max_spelling_len
    max_pronunciation_len = p.feature_neighborhood_input.max_pronunciation_len

    batch.pronunciation.set_shape([batch_size, max_pronunciation_len])
    batch.spelling.set_shape([batch_size, max_spelling_len])

    if p.use_neighbors:
      max_neighbors = p.feature_neighborhood_input.max_neighbors
      batch.neighbor_pronunciations.set_shape(
          [batch_size, max_neighbors, max_pronunciation_len])
      batch.neighbor_spellings.set_shape(
          [batch_size, max_neighbors, max_spelling_len])

    return batch

  @staticmethod
  def BasicConfig(base, combined_symbol_table):
    cfg = """
      input {{
        symbols: "{input_syms}"
        start_of_sentence: "<s>"
        end_of_sentence: "</s>"
      }}
      output {{
        symbols: "{output_syms}"
        start_of_sentence: "<s>"
        end_of_sentence: "</s>"
      }}
      append_eos: true
      max_spelling_len: 20
      max_pronunciation_len: 40
      max_neighbors: 50
      batch_opts {{
        pad_type: PAD_MAX_LEN
        pad_dim: 0
        pad_value: 0
        fixed_len: 128
      }}""".format(
          input_syms=os.path.join(base, combined_symbol_table),
          output_syms=os.path.join(base, combined_symbol_table))
    opts = pb_opts.FeatureNeighborhoodTensorOpts()
    text_format.Parse(cfg, opts)
    return (opts, pynini.SymbolTable.read_text(opts.input.symbols),
            pynini.SymbolTable.read_text(opts.output.symbols))

  @staticmethod
  def ParameterizedConfigs(input_symbol_path,
                           output_symbol_path,
                           append_eos=True,
                           max_spelling_len=20,
                           max_pronunciation_len=40,
                           max_neighbors=50,
                           split_output_on_space=False):
    cfg = """
      input {{
        symbols: "{input_syms}"
        start_of_sentence: "<s>"
        end_of_sentence: "</s>"
      }}
      output {{
        symbols: "{output_syms}"
        start_of_sentence: "<s>"
        end_of_sentence: "</s>"
      }}
      append_eos: {append_eos}
      max_spelling_len: {max_spelling_len}
      max_pronunciation_len: {max_pronunciation_len}
      max_neighbors: {max_neighbors}
      split_output_on_space: {split_output_on_space}
      batch_opts {{
        pad_type: PAD_MAX_LEN
        pad_dim: 0
        pad_value: 0
        fixed_len: 128
      }}""".format(
          input_syms=input_symbol_path,
          output_syms=output_symbol_path,
          append_eos=append_eos,
          max_spelling_len=max_spelling_len,
          max_pronunciation_len=max_pronunciation_len,
          max_neighbors=max_neighbors,
          split_output_on_space=split_output_on_space)
    opts = pb_opts.FeatureNeighborhoodTensorOpts()
    text_format.Parse(cfg, opts)
    return (opts, pynini.SymbolTable.read_text(opts.input.symbols),
            pynini.SymbolTable.read_text(opts.output.symbols))
