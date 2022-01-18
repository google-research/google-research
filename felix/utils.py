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

"""Utility functions for Felix."""

import json
from typing import Callable, Iterator, Mapping, MutableSequence, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
import tensorflow as tf

from felix import felix_constants as constants
from felix import tokenization

FeedDict = Mapping[str, Sequence[Sequence[float]]]
SourceTargetPair = Tuple[MutableSequence[str], str]


def get_token_list(text):
  """Returns a list of tokens.

  This function expects that the tokens in the text are separated by space
  character(s). Example: "ca n't , touch". This is the case at least for the
  public DiscoFuse and WikiSplit datasets.

  Args:
    text: String to be split into tokens.
  """
  return text.split()


def build_feed_dict(tokens,
                    tokenizer,
                    target_tokens = None,
                    max_seq_length = 128,
                    max_predictions_per_seq = 20):
  """Returns a dictionary used for predicting/training the insertion model.

  Converts a list of source tokens, containing masks, to a dictionary of
  features used by a TF model. If a target sequence is provided, then the
  targets for the MASKs are set.

  Args:
    tokens: Input tokens, with mask tokens.
    tokenizer: Tokenizer used to convert tokens to IDs.
    target_tokens: (Optional) The targets of the mask tokens.
    max_seq_length: Maximum sequence length.
    max_predictions_per_seq: Maximum number of mask tokens.

  Returns:
    Dictionary with model features or None if `len(tokens) > max_seq_length` or
    if the number of MASKs is larger than `max_predictions_per_seq`.
  """
  mask_position = []
  mask_target_id = []
  mask_target_weight = []

  for idx, token in enumerate(tokens):
    if token != constants.MASK:
      continue

    mask_position.append(idx)
    if target_tokens:
      mask_target_id += tokenizer.convert_tokens_to_ids([target_tokens[idx]])
    else:
      mask_target_id.append(0)
    mask_target_weight.append(1.0)

  # Deleted tokens (bracketed by unused) should have a segment_id of 2.
  unused = False
  segment_ids = []
  for token in tokens:
    if token == constants.DELETE_SPAN_START or unused:
      unused = True
      segment_ids.append(2)
    else:
      segment_ids.append(0)
    if token == constants.DELETE_SPAN_END:
      unused = False
  input_mask = [1] * len(tokens)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  assert len(segment_ids) == len(input_ids)
  # Padding.
  while len(input_ids) < max_seq_length:
    segment_ids.append(0)
    input_ids.append(0)
    input_mask.append(0)

  if len(input_ids) > max_seq_length:
    return None

  assert len(input_ids) == max_seq_length, "len(input_ids) = {}".format(
      len(input_ids))
  assert len(input_mask) == max_seq_length, "len(input_mask) = {}".format(
      len(input_mask))
  assert len(segment_ids) == max_seq_length, "len(segment_ids) = {}".format(
      len(segment_ids))

  if len(mask_position) > max_predictions_per_seq:
    return None
  while len(mask_position) < max_predictions_per_seq:
    mask_target_weight.append(0)
    mask_position.append(0)
    mask_target_id.append(0)

  feed_dict = {
      "input_ids": [input_ids],
      "input_mask": [input_mask],
      "segment_ids": [segment_ids],
      "masked_lm_positions": [mask_position],
      "masked_lm_ids": [mask_target_id],
      "masked_lm_weights": [mask_target_weight],
  }

  return feed_dict


def _int_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _text_feature(values):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(
          value=[element.encode("utf8") for element in values]))


def feed_dict_to_tf_example(feed_dict,
                            source = None,
                            target = None):
  """Returns a TF example for MLM insertion model."""
  features = {
      "input_ids": _int_feature(feed_dict["input_ids"][0]),
      "input_mask": _int_feature(feed_dict["input_mask"][0]),
      "segment_ids": _int_feature(feed_dict["segment_ids"][0]),
      "masked_lm_positions": _int_feature(feed_dict["masked_lm_positions"][0]),
      "masked_lm_ids": _int_feature(feed_dict["masked_lm_ids"][0]),
      "masked_lm_weights": _float_feature(feed_dict["masked_lm_weights"][0]),
  }
  if source:
    features["text_source"] = _text_feature([source])
  if target:
    features["text_target"] = _text_feature([target])
  return tf.train.Example(features=tf.train.Features(feature=features))


class Features(NamedTuple):
  """A data holder for various features that can be read from files."""
  source: MutableSequence[str]
  target: str
  output_variant_id: Optional[int] = None

  @staticmethod
  def from_source_target_pair(pair):
    return Features(source=pair[0], target=pair[1])

SourcesAndFeaturesPair = Tuple[MutableSequence[str], Features]


def text_file_iterator(fname_pattern):
  """Returns an iterator over lines of the files covered by fname_pattern."""
  for fname in get_filenames(fname_pattern):
    with tf.io.gfile.GFile(fname, "r") as f:
      for line in f:
        yield line


def skip_header_text_file_iterator(fname_pattern):
  """Similar to text_file_iterator, but skipping the first line of each file."""
  for fname in get_filenames(fname_pattern):
    tf.io.gfile.GFile(fname)
    it = tf.io.gfile.GFile(fname, "r")
    it.next()  # skip the header line
    for line in it:
      yield line


def get_parse_tsv_line_fn(
    return_none_on_error = False,
    reverse = False):
  """A higher-order function producing TSV line-parsing functions.

  Args:
    return_none_on_error: Whether to return None on encountering an error (such
      as too few TSV columns) rather than raising an Error.
    reverse: When True, returns ([`target`], `source`) instead of ([`source`],
      `target`). Useful for working with "reverse" (a.k.a. "noise" models that
      go from `target` to `source`.

  Returns:
    A parsing function that goes from a text line to a ([source], target) pair
    (or a ([`target`], `source`) pair when `reverse`=True).
  """

  def parse_tsv_line(line):
    """Parses the first two columns, `source` and `target`, from a TSV line.

    Any further columns are ignored.

    Args:
      line: A text line.

    Returns:
      a tuple ([source], target), with `source` being wrapped in a list.

    Raises:
      ValueError: when the line has less than two TSV columns and
        `return_none_on_error`=False.
    """
    split = line.rstrip("\n").split("\t")
    if len(split) < 2:
      message = 'TSV line has less than two tab-delimited fields:\n"{}"'.format(
          line)
      if return_none_on_error:
        logging.warning(message)
        return None
      else:
        raise ValueError(message)
    source, target = split[:2]
    if reverse:
      return [target], source
    else:
      return [source], target

  return parse_tsv_line


def parse_discofuse_line(line):
  """Parses a DiscoFuse example from a line from a TSV file.

  The documentation for this format:
  https://github.com/google-research-datasets/discofuse#data-format

  Args:
    line: A line from a TSV file.

  Returns:
    A pair (<source texts list>, <target text>).
  """
  coherent_1, coherent_2, incoherent_1, incoherent_2, _, _, _, _ = (
      line.rstrip("\n").split("\t"))
  # Strip because the second coherent sentence might be empty.
  fusion = (coherent_1 + " " + coherent_2).strip()
  return [incoherent_1, incoherent_2], fusion


def yield_sources_and_targets(
    input_file_pattern,
    input_format,
    source_key = None,
    target_key = None):
  """Produces an iterator over pairs (source list, targets) parsed from a file.

  Args:
    input_file_pattern: Path/pattern to the input file(s).
    input_format: Format of the input file.
    source_key: Source text feature name. Only considered when
      `input_format=sstable`.
    target_key: Target text feature name. Only considered when
      `input_format=sstable`.

  Yields:
    Pairs of (list of source texts, target text).
  """
  data_spec = {
      "wikisplit": (text_file_iterator, get_parse_tsv_line_fn()),
      "discofuse": (skip_header_text_file_iterator, parse_discofuse_line),
  }

  if input_format not in data_spec:
    raise ValueError("Unsupported input_format: {}".format(input_format))

  file_iterator_fn, parse_fn = data_spec[input_format]
  for item in file_iterator_fn(input_file_pattern):
    # Pytype correctly infers possible types for `item`, but does not handle
    # well the various possible signatures of `parse_fn`.
    parsed_item = parse_fn(item)  # pytype: disable=wrong-arg-types
    if parsed_item is not None:
      yield parsed_item


def get_filenames(patterns):
  """Obtains a list of filenames corresponding to the pattern.

  Supports patterns, as well as plain
  file names, as well as comma-separated lists of patterns.

  Caveat: Will not work if the patterns have commas (',') in them.

  Args:
    patterns: File pattern or comma-separated patterns.

  Raises:
      RuntimeError: If `patterns` is valid but cannot be expanded/does not match
          any files.

  Returns:
    list of individual paths to each file.
  """
  all_files = []
  for pattern in patterns.split(","):
    # points to a specific file.
    files = tf.io.gfile.glob(pattern)
    if not files:
      raise RuntimeError("Could not find files matching: %s" % pattern)
    all_files.extend(files)

  return all_files


def read_label_map(
    path,
    use_str_keys = False):
  """Returns label map read from the given path.

  Args:
    path: Path to the label map file.
    use_str_keys: Whether to use label strings as keys instead of
      (base tag, num insertions) tuple keys. The latter is only used by
      FelixInsert.
  """
  label_map = {}
  with tf.io.gfile.GFile(path) as f:
    if path.endswith(".json"):
      label_map = json.load(f)
    else:
      for tag in f:
        tag = tag.strip()
        # Empty lines are skipped.
        if tag:
          if tag in label_map:
            raise ValueError("Duplicate label in label_map: {}".format(tag))
          label_map[tag] = len(label_map)
  if not use_str_keys:
    new_label_map = {}
    for key, val in label_map.items():
      if "|" in key:
        pos_pipe = key.index("|")
        new_key = (key[:pos_pipe], int(key[pos_pipe + 1:]))
      else:
        new_key = (key, 0)
      new_label_map[new_key] = val
    label_map = new_label_map
  return label_map
