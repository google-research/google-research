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

"""Build BERT Examples from text (source, target) pairs."""

import collections
import itertools
from typing import Mapping, MutableSequence, Optional, Sequence, Tuple

import frozendict
import tensorflow as tf

from felix import felix_constants as constants
from felix import insertion_converter
from felix import pointing_converter
from felix import tokenization
from felix import utils


class BertExample:
  """Class for training and inference examples for BERT.

    Attributes:
      features: A dictionary of features with numeral lists as values.
      features_float: A dictionary of features with float lists as values.
      scalar_features: A dictionary of features with scalar values.
  """

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               labels,
               point_indexes,
               labels_mask,
               input_tokens,
               label_tokens = None,
               source_text = None,
               target_text = None):
    """Constructor for BERTExample.

    Args:
      input_ids: list of ids of source tokens.
      input_mask: list of 1s and 0s. 0 indicates a PAD token.
      segment_ids: List of segment ids for BERT.
      labels: list of added_phrases. If list is empty we assume we are at test
        time.
      point_indexes: list of target points.
      labels_mask: list of 1s and 0s. 0 indicates a PAD token.
      input_tokens: List of tokens (as text), For debug purposes.
      label_tokens: List of labels (as text). Optional. For debug purposes.
      source_text: Raw string of source input. Optional. For debug purposes.
      target_text: Raw string of target output. Optional. For debug purposes.
    """
    if not labels:
      self.features = collections.OrderedDict([('input_ids', input_ids),
                                               ('input_mask', input_mask),
                                               ('segment_ids', segment_ids)])
      self.features_float = {}
    else:
      self.features = collections.OrderedDict([
          ('input_ids', input_ids),
          ('point_indexes', point_indexes),
          ('input_mask', input_mask),
          ('segment_ids', segment_ids),
          ('labels', labels),
      ])

      self.features_float = collections.OrderedDict([
          ('labels_mask', labels_mask),
      ])
    self.scalar_features = collections.OrderedDict()
    self.debug_features = collections.OrderedDict()
    self.debug_features['input_tokens'] = input_tokens
    if label_tokens is not None:
      self.debug_features['label_tokens'] = label_tokens
    if source_text is not None:
      self.debug_features['text_source'] = [source_text]
    if target_text is not None:
      self.debug_features['text_target'] = [target_text]

  def pad_to_max_length(self, max_seq_length, pad_token_id):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """

    for key, feature in itertools.chain(self.features.items(),
                                        self.features_float.items()):
      pad_len = max_seq_length - len(feature)
      pad_id = pad_token_id if key == 'input_ids' else 0

      feature.extend([pad_id] * pad_len)
      if len(feature) != max_seq_length:
        raise ValueError('{} has length {} (should be {}).'.format(
            key, len(feature), max_seq_length))

  def _int_feature(self, values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  def _float_feature(self, values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

  def to_tf_example(self):
    """Returns this object as a tf.Example."""

    tf_features = collections.OrderedDict([
        (key, self._int_feature(val)) for key, val in self.features.items()
    ])
    # Add scalar integer features.
    for key, value in self.scalar_features.items():
      tf_features[key] = self._int_feature([value])

    # Add label mask feature.
    for key, value in self.features_float.items():
      tf_features[key] = self._float_feature(value)

    # Add debug fields.
    for key, value in self.debug_features.items():
      tf_features[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[element.encode('utf8') for element in value]))

    return tf.train.Example(features=tf.train.Features(feature=tf_features))


class BertExampleBuilder:
  """Builder class for BertExample objects.

  Attributes:
    label_map: Mapping from tags to tag IDs.
    tokenizer: A tokenization.FullTokenizer, which converts between strings and
      lists of tokens.
  """

  def __init__(self,
               label_map,
               max_seq_length,
               do_lower_case,
               converter,
               use_open_vocab,
               vocab_file = None,
               converter_insertion = None,
               special_glue_string_for_sources = None):
    """Initializes an instance of BertExampleBuilder.

    Args:
      label_map: Mapping from tags to tag IDs.
      max_seq_length: Maximum sequence length.
      do_lower_case: Whether to lower case the input text. Should be True for
        uncased models and False for cased models.
      converter: Converter from text targets to points.
      use_open_vocab: Should MASK be inserted or phrases. Currently only True is
        supported.
      vocab_file: Path to BERT vocabulary file.
      converter_insertion: Converter for building an insertion example based on
        the tagger output. Optional.
      special_glue_string_for_sources: If there are multiple sources, this
        string is used to combine them into one string. The empty string is a
        valid value. Optional.
    """
    self.label_map = label_map
    inverse_label_map = {}
    for label, label_id in label_map.items():
      if label_id in inverse_label_map:
        raise ValueError(
            'Multiple labels with the same ID: {}'.format(label_id))
      inverse_label_map[label_id] = label
    self._inverse_label_map = frozendict.frozendict(inverse_label_map)
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file, do_lower_case=do_lower_case)
    self._max_seq_length = max_seq_length
    self._converter = converter
    self._pad_id = self._get_pad_id()
    self._do_lower_case = do_lower_case
    self._use_open_vocab = use_open_vocab
    self._converter_insertion = converter_insertion
    if special_glue_string_for_sources is not None:
      self._special_glue_string_for_sources = special_glue_string_for_sources
    else:
      self._special_glue_string_for_sources = ' '

  def build_bert_example(
      self,
      sources,
      target = None,
      is_test_time = False
  ):
    """Constructs a BERT tagging and insertion examples.

    Args:
      sources: List of source texts.
      target: Target text or None when building an example during inference. If
        the target is None then we don't calculate gold labels or tags, this is
        equivaltn to setting is_test_time to True.
      is_test_time: Controls whether the dataset is to be used at test time.
        Unlike setting target = None to indicate test time, this flags allows
        for saving the target in the tfrecord.

    Returns:
      A tuple with:
      1. BertExample for the tagging model or None if there's a tag not found in
      self.label_map or conversion from text to tags was infeasible.
      2. FeedDict for the insertion model or None if the BertExample or the
      insertion conversion failed.
    """

    merged_sources = self._special_glue_string_for_sources.join(sources)
    original_source = merged_sources
    merged_sources = merged_sources.strip()
    if self._do_lower_case:
      merged_sources = merged_sources.lower()
      # [SEP] Should always be uppercase.
      merged_sources = merged_sources.replace(constants.SEP.lower(),
                                              constants.SEP)
    tokens = self._split_to_wordpieces(merged_sources.split())
    tokens = self._truncate_list(tokens)

    input_tokens = [constants.CLS] + tokens + [constants.SEP]
    input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    if not target or is_test_time:
      example = BertExample(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          labels=[],
          point_indexes=[],
          labels_mask=[],
          input_tokens=input_tokens,
          source_text=original_source,
          target_text=target)
      example.pad_to_max_length(self._max_seq_length, self._pad_id)
      return example, None

    if self._do_lower_case:
      target = target.lower()

    output_tokens = self._split_to_wordpieces(target.split())

    output_tokens = self._truncate_list(output_tokens)
    output_tokens = [constants.CLS] + output_tokens + [constants.SEP]
    points = self._converter.compute_points(' '.join(input_tokens).split(),
                                            ' '.join(output_tokens))
    if not points:
      return None, None

    labels = [t.added_phrase for t in points]

    point_indexes = [t.point_index for t in points]
    point_indexes_set = set(point_indexes)
    try:
      new_labels = []
      for i, added_phrase in enumerate(labels):
        if i not in point_indexes_set:
          new_labels.append(self.label_map['DELETE'])
        elif not added_phrase:
          new_labels.append(self.label_map['KEEP'])
        else:
          if self._use_open_vocab:
            new_labels.append(self.label_map['KEEP|' +
                                             str(len(added_phrase.split()))])
          else:
            new_labels.append(self.label_map['KEEP|' + str(added_phrase)])
        labels = new_labels
    except KeyError:
      # added_phrase is not in label_map.
      return None, None

    if not labels:
      return None, None

    label_tokens = [
        self._inverse_label_map.get(label_id, constants.PAD)
        for label_id in labels
    ]
    label_counter = collections.Counter(labels)
    label_weight = {
        label: len(labels) / count / len(label_counter)
        for label, count in label_counter.items()
    }
    # Weight the labels inversely proportional to their frequency.
    labels_mask = [label_weight[label] for label in labels]
    example = BertExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=labels,
        point_indexes=point_indexes,
        labels_mask=labels_mask,
        input_tokens=input_tokens,
        label_tokens=label_tokens,
        source_text=merged_sources,
        target_text=target)
    example.pad_to_max_length(self._max_seq_length, self._pad_id)

    insertion_example = None
    if self._converter_insertion is not None:
      insertion_example = self._converter_insertion.create_insertion_example(
          input_tokens, labels, point_indexes, output_tokens)

    return example, insertion_example

  def _split_to_wordpieces(self, tokens):
    """Splits tokens to WordPieces.

    Args:
      tokens: Tokens to be split.

    Returns:
      List of WordPieces.
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    special_tokens = {'<::::>', constants.SEP.lower(), constants.CLS.lower()}
    for token in tokens:

      # Don't tokenize special tokens.
      if token.lower() not in special_tokens:
        pieces = self.tokenizer.tokenize(token)
      else:
        pieces = [token]

      bert_tokens.extend(pieces)

    return bert_tokens

  def _truncate_list(self, x):
    """Returns truncated version of x according to the self._max_seq_length."""
    # Save two slots for the first [CLS] token and the last [SEP] token.
    return x[:self._max_seq_length - 2]

  def _get_pad_id(self):
    """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
    try:
      return self.tokenizer.convert_tokens_to_ids([constants.PAD])[0]
    except KeyError:
      return 0
