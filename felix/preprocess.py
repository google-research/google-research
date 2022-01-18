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

"""Utility functions for preprocessing Felix* examples."""
from typing import Optional, Union
from felix import bert_example
from felix import example_builder_for_felix_insert
from felix import insertion_converter
from felix import pointing_converter
from felix import utils


def initialize_builder(
    use_pointing, use_open_vocab, label_map_file,
    max_seq_length, max_predictions_per_seq, vocab_file,
    do_lower_case,
    special_glue_string_for_sources,
    max_mask,
    insert_after_token,
):
  """Returns a builder for tagging and insertion BERT examples."""

  is_felix_insert = (not use_pointing and use_open_vocab)
  label_map = utils.read_label_map(
      label_map_file, use_str_keys=(not is_felix_insert))

  if use_pointing:
    if use_open_vocab:
      converter_insertion = insertion_converter.InsertionConverter(
          max_seq_length=max_seq_length,
          max_predictions_per_seq=max_predictions_per_seq,
          label_map=label_map,
          vocab_file=vocab_file)
      converter_tagging = pointing_converter.PointingConverter({},
                                                               do_lower_case)

    builder = bert_example.BertExampleBuilder(
        label_map=label_map,
        vocab_file=vocab_file,
        max_seq_length=max_seq_length,
        converter=converter_tagging,
        do_lower_case=do_lower_case,
        use_open_vocab=use_open_vocab,
        converter_insertion=converter_insertion,
        special_glue_string_for_sources=special_glue_string_for_sources)
  else:  # Pointer disabled.
    if use_open_vocab:
      builder = example_builder_for_felix_insert.FelixInsertExampleBuilder(
          label_map,
          vocab_file,
          do_lower_case,
          max_seq_length,
          max_predictions_per_seq,
          max_mask,
          insert_after_token,
          special_glue_string_for_sources)
    else:
      raise ValueError('LaserTagger model cannot be trained with the Felix '
                       'codebase yet, set `FLAGS.use_open_vocab=True`')
  return builder
