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

"""A collection of utilities that are used for preparing data."""

from collections.abc import Callable, Sequence
import hashlib
import logging
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
MASK = '<MASK>'
NL = '<NL>'
SEED = 11731
random.seed(SEED)
np.random.seed(SEED)


def is_comment(line):
  """Checks if `line` is a C/C++ comment or begins a comment."""
  comment_strings = {
      '//',
      '/*',
      '*/',
  }
  return any(cstring in line for cstring in comment_strings)


assert is_comment('//this is a test')
assert is_comment('/*this is a test')
assert is_comment('/*this is a test/*')
assert not is_comment('this is a test')


def is_pragma(line):
  """Checks if the line contains a pragma."""
  return 'pragma hls' in line.lower()


def identity(x):
  return x


def filter_non_pragma(x):
  """Removes lines that are not pragma."""
  return [xi for xi in x if 'pragma' not in xi.lower()]


def create_labeled_data(file_paths,
                        job_type,
                        span_filter_f,
                        delta,
                        add_hard_neg_samples,
                        balance_classes = True,
                        add_neg_samples = True,
                        skip_arguments = None,
                        neg_samples_fraction = None,
                        return_spans = False):

  """Create labeled data."""

  raw_data = create_raw_labeled_data(
      file_paths=file_paths,
      span_filter_f=span_filter_f,
      delta=delta,
      random_seed=SEED,
      add_hard_neg_samples=add_hard_neg_samples,
      balance_classes=balance_classes,
      add_neg_samples=add_neg_samples)

  if job_type == 'generation':
    return post_process_generation(
        raw_data,
        skip_arguments=skip_arguments,
        neg_samples_fraction=neg_samples_fraction,
        return_spans=return_spans,
        random_state=SEED)

  elif job_type == 'github_binary_classification':
    return post_process_github_classification(
        raw_data, return_spans=return_spans)


def create_raw_labeled_data(
    file_paths,
    span_filter_f,
    delta,
    random_seed,
    balance_classes,
    add_neg_samples,
    add_hard_neg_samples,
    tag_as_pragma = lambda x: not is_comment(x) and is_pragma(x),  # pylint: disable=line-too-long
    tag_as_non_pragma = lambda x: not is_comment(x) and not is_pragma(x)
):
  """Prepares dataset from given file_paths.

  This function prepares a generic dataset that can be used for classification
  and generation.

  Args:
    file_paths: The list of paths from which we read.
    span_filter_f: Removes certain spans
    delta: The window size
    random_seed: Random seed for data generation.
    balance_classes: Whether to balance classes
    add_neg_samples: If True, it enables adding negative samples to the data.
    add_hard_neg_samples: If true, hard negative sample are added
    tag_as_pragma: Whether to tag this as a pragma entry.
    tag_as_non_pragma: Whether to tag this as a non-pragam entry.

  Returns:
    A Panda dataframe that contains the entire dataset.
  """

  def _generate_pragma_data():
    pragma_inputs, pragma_outputs, pragma_spans = get_io_pairs(
        file_paths,
        delta=delta,
        check_line_function=tag_as_pragma,
        compress_spans=True,
        span_filter_f=span_filter_f,
        return_spans=True)
    logging.info(len(pragma_inputs))

    pragma_data = pd.DataFrame({
        'input': pragma_inputs,
        'label': 1,
        'target': pragma_outputs,
        'span': pragma_spans
    })
    pragma_data.drop_duplicates(subset=['input'], inplace=True)
    return pragma_data

  def _generate_nonpragma_data():
    non_pragma_inputs, non_pragma_outputs, non_pragma_spans = get_io_pairs(
        file_paths,
        delta=delta,
        check_line_function=tag_as_non_pragma,
        compress_spans=False,
        span_filter_f=span_filter_f,
        return_spans=True)
    logging.info(len(non_pragma_inputs))
    non_pragma_data = pd.DataFrame({
        'input': non_pragma_inputs,
        'label': 0,
        'target': non_pragma_outputs,
        'span': non_pragma_spans
    })
    non_pragma_data.drop_duplicates(subset=['input'], inplace=True)
    return non_pragma_data

  def _balance_and_dedup():
    if balance_classes:
      if add_hard_neg_samples:
        non_pragma_data_sampled = non_pragma_data.sample(
            n=len(pragma_data), random_state=random_seed)
      else:
        non_pragma_data_sampled = get_hard_neg_sample(
            data=non_pragma_data, n=len(pragma_data))
      data = pd.concat([pragma_data, non_pragma_data_sampled], axis=0)
    else:
      data = pd.concat([pragma_data, non_pragma_data], axis=0)
    data.drop_duplicates(subset=['input'], inplace=True)
    logging.info('Finally got {%s} records', len(data))
    logging.info(data.label.value_counts())
    return data

  def _cleanup():
    # Replace \t with four spaces since we are dealing with TSVs downstream.
    data['input'] = data['input'].apply(lambda x: x.replace('\t', '    '))
    data['target'] = data['target'].apply(lambda x: x.replace('\t', '    '))

  # Step 1
  pragma_data = _generate_pragma_data()
  if not add_neg_samples:
    logging.info('Finally got {%s} records', len(pragma_data))
    logging.info(pragma_data.label.value_counts())
    return pragma_data

  # Step 2
  non_pragma_data = _generate_nonpragma_data()

  # Step 3
  data = _balance_and_dedup()

  # Step 4
  _cleanup()

  return data


def get_io_pairs(all_file_paths,
                 check_line_function,
                 compress_spans,
                 delta,
                 span_filter_f,
                 return_spans = False):
  """Generates input/output pairs for AutoPragma.

  Given a list of file paths, this method first reads a file in a list, then
  extracts the spans that satisfy the check_line_function. If compress_spans is
  true, multiple spans are compressed to a single MASK token.  This is useful
  when multiple pragmas are present next to each other.

  Args:
    all_file_paths: A sequence of filenames.
    check_line_function: A callable for checking the lines.
    compress_spans: Whether to compress the spanning window.
    delta: An integer defining the delta for data selection.
    span_filter_f: A callable function to check the span of filter.
    return_spans: Whether to return span.

  Returns:
    A set of data.
  """

  files_already_read = set()
  inputs, outputs = [], []
  spans = [] if return_spans else None
  n_exceptions = 0
  n_skipped = 0
  n_processed = 0
  for file_path in tqdm(
      all_file_paths, total=len(all_file_paths), desc='Reading files'):
    try:
      lines = read_file_into_lines(file_path)
      file_signature = hashlib.sha256(' '.join(lines).encode()).hexdigest()
      if file_signature in files_already_read:
        n_skipped += 1
        continue
      files_already_read.add(file_signature)
      n_processed += 1
      spans = get_spans(
          lines=lines,
          check_line_function=check_line_function,
          compress_spans=compress_spans)
    # pylint: disable=broad-except
    except Exception as _:
      n_exceptions += 1
      continue

    processed_spans = []
    for span in spans:
      try:
        context, output = get_context_given_span(
            lines, span, span_filter_f=span_filter_f, delta=delta)
      # pylint: disable=broad-except
      except Exception as _:
        continue
      inputs.append(context)
      outputs.append(output)
      if return_spans:
        processed_spans.append(span)

  logging.info('Processed = %d, Exceptions = %d', n_processed, n_exceptions)
  if return_spans:
    return inputs, outputs, processed_spans
  else:
    return inputs, outputs


def read_file_into_lines(path):
  with tf.io.gfile.Open(path, 'r') as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    return lines


def get_spans(lines, check_line_function,
              compress_spans):
  """Receive spans around lines contingent on the `check_line_function`.

  Each span is represented by a window [start_span, end_span]. In the context of
  autopragma, these are locations of pragmas.

  Args:
    lines: A sequence of lines.
    check_line_function: A callable to apply on each line.
    compress_spans: Whether to compress the spans.

  Returns:
    A sequence of integer tuples.
  """

  def _merge_spans():
    """Merges contiguous spans that satisfy `check_line_function`."""
    okay_spans = []
    span_begin_line = okay_lines[0]
    prev_line = span_begin_line

    for curr_line in okay_lines[1:]:
      is_contiguous = curr_line - prev_line == 1
      if is_contiguous:
        prev_line = curr_line
      else:
        okay_spans.append((span_begin_line, prev_line + 1))
        span_begin_line = curr_line
        prev_line = curr_line

    okay_spans.append((span_begin_line, prev_line + 1))
    return okay_spans

  okay_lines = []
  for i, line in enumerate(lines):
    if check_line_function(line):
      okay_lines.append(i)

  if compress_spans:
    return _merge_spans()

  return [(s, s + 1) for s in okay_lines]


def get_context_given_span(lines,
                           span,
                           delta,
                           span_filter_f=lambda x: x):
  """Gets context of length `delta` around `span` from `lines`.

  The lines returned do not include span[0], but include span[1]
  Lines that are omitted by span_filter_f are not included.

  Args:
    lines: Sequence of strings.
    span: A tuple of integers defining the surrounding window.
    delta: An integer definding the delta.
    span_filter_f: A linear function.

  Returns:
    A tuple of strings for the surrounding lines.
  """
  prefix_context = span_filter_f(lines[max(span[0] - delta, 0):span[0]])
  suffix_context = span_filter_f(lines[span[1]:span[1] + delta])
  span = NL.join(lines[span[0]:span[1]])
  input_context = NL.join(prefix_context + [MASK] + suffix_context)
  output_pragma = f'{MASK} {span}'
  return (input_context, output_pragma)


def post_process_github_classification(data,
                                       return_spans):
  """Post-process the resulting github classifications.

  Args:
    data: The input dataframe.
    return_spans: Whether to return span.

  Returns:
    A Pandas dataframe of the resulting post-processing.
  """
  columns_to_return = ['input', 'label']
  if return_spans:
    columns_to_return.append('span')
  return data[columns_to_return]


def post_process_generation(data,
                            skip_arguments,
                            return_spans,
                            random_state,
                            neg_samples_fraction = 0.0):
  """Post processes generation output.

  Args:
    data: The input data.
    skip_arguments: Whether to skip arguments for the pragma.
    return_spans: Whether to return the spans.
    random_state: An integer defining a random state.
    neg_samples_fraction: The fraction of negative samples to add.

  Returns:
    Processed dataframe.
  """

  data_with_pragma = data[data.label == 1]

  logging.info('Skip arguments set to %s', str(skip_arguments))
  data_with_pragma['target'] = data_with_pragma['target'].apply(
      lambda t: process_pragma_target(t, skip_arguments))

  if neg_samples_fraction and neg_samples_fraction > 0:
    data_without_pragma = data[data.label == 0].sample(
        frac=neg_samples_fraction, random_state=random_state)
    data_without_pragma['input'] = data_without_pragma['input'].apply(
        lambda x: x.replace('\t', '    '))
    data_without_pragma['target'] = 'N/A'

    data = pd.concat([data_with_pragma, data_without_pragma], axis=0)
  else:
    data = data_with_pragma
  data['input'] = data['input'].apply(lambda x: x.replace('\t', '    '))
  data.drop_duplicates(subset=['input'], inplace=True)

  columns_to_return = ['input', 'target']
  if return_spans:
    columns_to_return.append('span')
  return data[columns_to_return]


def process_pragma_target(target_str, skip_arguments):
  """Processes the pragma. Pragma arguments are skipped if `skip_arguments`.

  Args:
    target_str: The target string for processing.
    skip_arguments: Whether to skip the arguments.

  Returns:
    A string that is after processing pragmas.
  """

  def _process_pragma(pragma_str):
    # often, programmers use a macro for the pragmas themselves (HLS_).
    # Here, we try to standardize them.
    if 'HLS_' in pragma_str:
      pragma_str = pragma_str.replace('HLS_', 'HLS ')
    if 'hls_' in pragma_str:
      pragma_str = pragma_str.replace('hls_', 'hls ')
    pragma_fields = pragma_str.split()

    if len(pragma_fields) < 3:
      return pragma_str  # non-standard case, usually #pragma HLS something

    hash_pragma, hls_tok, pragma_type = pragma_fields[0], pragma_fields[
        1], pragma_fields[2]
    # #pragma hls pipeline

    processed_pragma = f'{hash_pragma} {hls_tok.upper()} {pragma_type.lower()}'
    if skip_arguments:
      return processed_pragma

    if len(pragma_fields) > 3:
      arguments = pragma_fields[3:]
      arguments = ' '.join(arguments).strip()
      arguments = arguments.replace(' =', '=')  # standardize assignment
      arguments = arguments.replace('= ', '=')  # standardize assignment
      arguments = arguments.replace('OFF', 'off')
      return f'{processed_pragma} {arguments}'
    else:
      return processed_pragma

  target_str = target_str.split(MASK)[1].strip()
  pragmas = target_str.split(NL)
  pragmas = [_process_pragma(p) for p in pragmas]
  if skip_arguments:  # duplicate pragmas possible
    pragmas = list(dict.fromkeys(pragmas))  # dedup but retain order
  return MASK + ' ' + NL.join(pragmas)


def get_hard_neg_sample(data,
                        n,
                        prop_hard = 0.66,
                        hard_col = 'has_loop_or_pragma'):
  """Returns `n` random samples from `data`.

  The hard examples are those negative examples where the context contains
  both a for loop and a pragma, *yet* there should not be a predicted pragma
  at the chosen location.

  Args:
    data: The data collected.
    n: The number of samples.
    prop_hard: A float defining the propbability of selection.
    hard_col: A string defining the hard column.

  Returns:
    Returns n randomly sampled data.
  """
  assert prop_hard <= 1
  n_hard_examples = int(n * prop_hard)

  def _has_for(txt):
    return 'for (' in txt or 'for(' in txt

  def _has_while(txt):
    return 'while' in txt

  def _has_pragma(txt):
    return 'pragma' in txt

  data['has_for'] = data['text'].apply(_has_for)
  data['has_while'] = data['text'].apply(_has_while)
  data['has_pragma'] = data['text'].apply(_has_pragma)
  data['has_pragma_and_for'] = data.apply(
      lambda row: row['has_for'] and row['has_pragma'], axis=1)
  data['has_loop'] = data.apply(
      lambda row: row['has_for'] or row['has_while'], axis=1)
  data['has_loop_or_pragma'] = data.apply(
      lambda row: row['has_loop'] or row['has_pragma'], axis=1)

  # pylint:disable=g-bool-id-comparison
  hard_examples = data[data[hard_col] is True]
  n_remaining = max(n - n_hard_examples, n - len(hard_examples))

  non_hard_examples = data[
      data[hard_col] is
      False]  # these examples may also be hard in ways we don't know
  # pylint:enable=g-bool-id-comparison
  return pd.concat([
      hard_examples.sample(n=n_hard_examples, random_state=SEED),
      non_hard_examples.sample(n=n_remaining, random_state=SEED)
  ],
                   axis=0)
