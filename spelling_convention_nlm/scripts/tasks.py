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

"""Custom tasks for US/UK spelling variation experiments."""

import functools
import os
import re


import seqio
import tensorflow as tf


DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

DEFAULT_MODEL = 'mt5'
TASK_CAPTION = 'prompt_scoring'


@seqio.map_over_dataset
def preprocess_tsv(
    line,
    field_delim='\t',
    num_fields=2,
    inputs_format='{0}',
    targets_format='{1}',
    field_names=None,
    use_quote_delim=False,
):
  r"""Parse tab-delimited strings into inputs and targets.

  This function takes a tf.data.Dataset of strings, each of which contains
  tab-delimited fields.  The function returns a tf.data.Dataset of feature
  dictionaries of the form {"inputs": string, "targets": string}.

  inputs_format contains a template string and field numbers or names used to
  produce the "inputs" string.
  targets_format contains a template string and field numbers or names used to
  produce the "targets" string.

  Example (field numbers):
    The input dataset contains the lines:
    "6,7,42"
    "2,9,18"
    preprocess_tsv(dataset,
                   field_delim=',',
                   inputs_format='numerator: {2} denominator: {1}',
                   targets_format='quotient: {0}'
    would produce a dataset containing the dictionaries:
    {"inputs": "numerator: 42 denominator: 7", "targets": "quotient: 6"}
    {"inputs": "numerator: 18 denominator: 9", "targets": "quotient: 2"}

  Example (field names):
    The input dataset contains the lines:
    "6,7,42"
    "2,9,18"
    preprocess_tsv(dataset,
                   field_delim=',',
                   field_names=['quot', 'denom', 'numer'],
                   inputs_format='numerator: {numer} denominator: {denom}',
                   targets_format='quotient: {quot}'
    would produce a dataset containing the dictionaries:
    {"inputs": "numerator: 42 denominator: 7", "targets": "quotient: 6"}
    {"inputs": "numerator: 18 denominator: 9", "targets": "quotient: 2"}

  Args:
    line: an example containing comma/tab-delimited string.
    field_delim: a string, the delimiter to split on e.g. ',' for csv.
    num_fields: an integer
    inputs_format: a string, the desired output format with placeholders for
      field values.
    targets_format: a string, the desired output format with placeholders for
      field values.
    field_names: a list of strings, the ordered names of the TSV fields.
      defaults to None (i.e. use field number in *_format)
    use_quote_delim: If false, treats double quotation marks as regular
      characters inside of the string fields (ignoring RFC 4180, Section 2,
      Bullet 5).

  Returns:
    A feature dict with 'inputs' and 'targets' features.
  """

  def _format_part_with_field_numbers(part, field_values):
    found = re.findall(r'{(\d+)}', part)
    if found:
      return field_values[int(found[0])]
    else:
      return part

  def _format_part_with_field_names(part, field_names, field_values):
    field_names_re = '|'.join(['{{({})}}'.format(x) for x in field_names])
    found = re.findall(field_names_re, part)
    if found:
      pos = field_names.index(''.join(found[0]))
      return field_values[int(pos)]
    else:
      return part

  def _format(format_string, field_names, field_values):
    if field_names is None:
      parts = [
          _format_part_with_field_numbers(p, field_values)
          for p in re.split(r'({\d+})', format_string)
      ]
    else:
      field_names_re = (
          '(' + '|'.join(['{{{}}}'.format(x) for x in field_names]) + ')'
      )
      parts = [
          _format_part_with_field_names(p, field_names, field_values)
          for p in re.split(field_names_re, format_string)
      ]
    return tf.strings.join(parts)

  field_values = tf.io.decode_csv(
      line,
      record_defaults=['']
      * (num_fields if field_names is None else len(field_names)),
      field_delim=field_delim,
      use_quote_delim=use_quote_delim,
  )
  return {
      'inputs': _format(inputs_format, field_names, field_values),
      'targets': _format(targets_format, field_names, field_values),
  }


# ------------- Inference Tasks:


def _register_scoring_task(
    data_path, source_column, target_column, num_fields, vocab_path
):
  """Registers single scoring task."""
  task_name = ('{model}_{caption}').format(
      model=DEFAULT_MODEL,
      caption=TASK_CAPTION,
  )
  seqio.TaskRegistry.add(
      task_name,
      source=seqio.TextLineDataSource(
          split_to_filepattern={
              'test': data_path,
          }
      ),
      preprocessors=[
          functools.partial(
              preprocess_tsv,
              field_delim='\t',
              num_fields=num_fields,
              inputs_format='{%d}' % source_column,
              targets_format='{%d}' % target_column,
          ),
          *DEFAULT_PREPROCESSORS,
      ],
      output_features={
          'inputs': seqio.Feature(
              vocabulary=seqio.SentencePieceVocabulary(
                  os.path.join(vocab_path, 'sentencepiece.model')
              )
          ),
          'targets': seqio.Feature(
              vocabulary=seqio.SentencePieceVocabulary(
                  os.path.join(vocab_path, 'sentencepiece.model')
              )
          ),
      },
      metric_fns=[],
  )
  return task_name
