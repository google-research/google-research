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

"""Build FelixInsert BERT Examples from text (source, target) pairs."""

import collections
import itertools
from typing import List, Mapping, Optional, Sequence, Tuple

import tensorflow as tf

from felix import converter_for_felix_insert as converter
from felix import felix_constants as constants
from felix import tokenization
from felix import utils


class TaggingBertExample:
  """Class for training and inference examples for BERT tagging model.

  Attributes:
    features: A dictionary of features with numeral lists as values.
    features_float: A dictionary of features with float lists as values.
    scalar_features: A dictionary of features with scalar values.
  """

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               labels = None,
               labels_mask = None,
               input_tokens = None,
               label_tokens = None,
               source_text = None,
               target_text = None,
               target_tokens = None,
              ):
    """Initializes an instance of TaggingBertExample.

    Args:
      input_ids: List of ids of source tokens.
      input_mask: List of 1s and 0s. 0 indicates a PAD token.
      segment_ids: List of segment ids for BERT.
      labels: List of label ids. Optional if we are at test time.
      labels_mask: List of 1s and 0s. Optional if we are at test time.
      input_tokens: List of tokens (as text). Optional. For building an
        insertion example and for debugging.
      label_tokens: List of labels (as text). Optional. For building an
        insertion example and for debugging.
      source_text: Raw string of source input. Optional. For debug purposes.
      target_text: Raw string of target output. Optional. For debug purposes.
      target_tokens: List of target tokens (as text). Optional. For debug
        purposes.
    """
    # Check if the labels(_mask) is None or an empty list.
    if not labels or not labels_mask:
      self.features = collections.OrderedDict([
          ('input_ids', list(input_ids)),
          ('input_mask', list(input_mask)),
          ('segment_ids', list(segment_ids)),
      ])
      self.features_float = {}
    else:
      self.features = collections.OrderedDict([
          ('input_ids', list(input_ids)),
          ('input_mask', list(input_mask)),
          ('segment_ids', list(segment_ids)),
          ('labels', list(labels)),
      ])

    self.features_float = collections.OrderedDict([
        ('labels_mask', labels_mask),
    ])

    self.scalar_features = collections.OrderedDict()

    self._debug_features = collections.OrderedDict()
    if input_tokens is not None:
      self._debug_features['input_tokens'] = input_tokens
    if label_tokens is not None:
      self._debug_features['label_tokens'] = label_tokens
    if source_text is not None:
      self._debug_features['text_source'] = [source_text]
    if target_text is not None:
      self._debug_features['text_target'] = [target_text]
    if target_tokens is not None:
      self._debug_features['target_tokens'] = target_tokens

  def pad_to_max_length(self, max_seq_length, pad_token_id):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """
    pad_len = max_seq_length - len(self.features['input_ids'])
    for key, feature in itertools.chain(self.features.items(),
                                        self.features_float.items()):
      if feature is None:
        continue

      pad_id = pad_token_id if key == 'input_ids' else 0

      if feature is not None:
        feature.extend([pad_id] * pad_len)
      if len(feature) != max_seq_length:
        raise ValueError(f'{key} has length {len(feature)} (should be '
                         f'{max_seq_length}).')

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
    for key, value in self._debug_features.items():
      tf_features[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[element.encode('utf8') for element in value]))

    return tf.train.Example(features=tf.train.Features(feature=tf_features))


class FelixInsertExampleBuilder:
  """Builder class for FelixInsert tagging and insertion examples.

  Attributes:
    label_map: Mapping from tags to tag IDs.
    tokenizer: A tokenization.FullTokenizer, which converts between strings and
      lists of tokens.
  """

  def __init__(self,
               label_map,
               vocab_file,
               do_lower_case,
               max_seq_length,
               max_predictions_per_seq,
               max_insertions_per_token,
               insert_after_token = True,
               special_glue_string_for_sources = None):
    """Initializes an instance of BertExampleBuilder.

    Args:
      label_map: Mapping from tags represented as (base_tag, num_insertions)
        tuples to tag IDs.
      vocab_file: Path to vocab file.
      do_lower_case: should text be lowercased.
      max_seq_length: Maximum sequence length.
      max_predictions_per_seq: Maximum number of tokens to insert per input.
      max_insertions_per_token: Maximum number of tokens/masks to insert per
        token.
      insert_after_token: Whether to insert tokens after the current token.
      special_glue_string_for_sources: If there are multiple sources, this
        string is used to combine them into one string. The empty string is a
        valid value. Optional.
    """
    self.label_map = label_map
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file, do_lower_case=do_lower_case)
    self._max_seq_length = max_seq_length
    self._max_predictions_per_seq = max_predictions_per_seq
    self._max_insertions_per_token = max_insertions_per_token
    self._insert_after_token = insert_after_token
    try:
      self._pad_id = self.tokenizer.convert_tokens_to_ids([constants.PAD])[0]
    except KeyError:
      self._pad_id = 0
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
    """Constructs a tagging and an insertion BERT Example.

    Args:
      sources: List of source texts.
      target: Target text or None when building an example during inference. If
        the target is None then we don't calculate gold labels or tags, this is
        equivaltn to setting is_test_time to True.
      is_test_time: Controls whether the dataset is to be used at test time.
        Unlike setting target = None to indicate test time, this flags allows
        for saving the target in the tfrecord.  For compatibility with old
        scripts, setting target to None has the same behavior as setting
        is_test_time to True.

    Returns:
      A tuple with:
      1. TaggingBertExample (or None if more than
      `self._max_insertions_per_token` insertions are required).
      2. A feed_dict object for creating the insertion BERT example (or None if
      `target` is None, `is_test_time` is True, or the above TaggingBertExample
      is None.

    Raises:
      KeyError: If a label not in `self.label_map` is produced.
    """
    merged_sources = self._special_glue_string_for_sources.join(sources).strip()
    input_tokens = self._tokenize_text(merged_sources)

    input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    if target is None or is_test_time:
      example = TaggingBertExample(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          labels=None,
          labels_mask=None,
          input_tokens=input_tokens,
          source_text=merged_sources,
          target_text=target)
      example.pad_to_max_length(self._max_seq_length, self._pad_id)
      return example, None

    output_tokens = self._tokenize_text(target)
    edits_and_insertions = converter.compute_edits_and_insertions(
        input_tokens, output_tokens, self._max_insertions_per_token,
        self._insert_after_token)
    if edits_and_insertions is None:
      return None, None
    else:
      edits, insertions = edits_and_insertions

    label_tokens = []  # Labels as strings.
    label_tuples = []  # Labels as (base_tag, num_insertions) tuples.
    labels = []  # Labels as IDs.
    for edit, insertion in zip(edits, insertions):
      label_token = edit
      if insertion:
        label_token += f'|{len(insertion)}'
      label_tokens.append(label_token)
      label_tuple = (edit, len(insertion))
      label_tuples.append(label_tuple)
      if label_tuple in self.label_map:
        labels.append(self.label_map[label_tuple])
      else:
        raise KeyError(
            f"Label map doesn't contain a computed label: {label_tuple}")

    label_counter = collections.Counter(labels)
    label_weight = {
        label: len(labels) / count / len(label_counter)
        for label, count in label_counter.items()
    }
    # Weight the labels inversely proportional to their frequency.
    labels_mask = [label_weight[label] for label in labels]
    if self._insert_after_token:
      # When inserting after the current token, we never need to insert after
      # the final [SEP] token and thus the edit label for that token is constant
      # ('KEEP') and could be excluded from loss computations.
      labels_mask[-1] = 0
    else:
      # When inserting before the current token, the first edit is constant.
      labels_mask[0] = 0

    example = TaggingBertExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=labels,
        labels_mask=labels_mask,
        input_tokens=input_tokens,
        label_tokens=label_tokens,
        source_text=merged_sources,
        target_text=target)
    example.pad_to_max_length(self._max_seq_length, self._pad_id)

    insertion_example = self.build_insertion_example(
        source_tokens=input_tokens,
        labels=label_tuples,
        target_insertions=insertions)
    return example, insertion_example

  def build_insertion_tokens(
      self,
      source_tokens,
      labels,
      target_insertions = None,
  ):
    """Constructs the masked input TF Example for the insertion model.

    Args:
      source_tokens: List of source tokens.
      labels: List of edit label tuples (base_tag, num_insertions).
      target_insertions: Inserted target tokens per source token. Only provided
        when constructing training examples.

    Returns:
      A tuple with a masked insertion token sequence and the unmasked target
      sequence.

    Raises:
      ValuError: If the number of insertions in a label doesn't match the
        number of elements in the corresponding target_insertions[i] list.
    """
    masked_tokens = []
    target_tokens = []
    deletion_started = False
    # Tokens to add after finishing the deleted span.
    mask_buffer = []
    target_buffer = []
    for i, (source_token,
            (base_tag,
             num_insertions)) in enumerate(zip(source_tokens, labels)):
      masks = [constants.MASK] * num_insertions
      if target_insertions is not None:
        insertions = target_insertions[i]
        if len(insertions) != num_insertions:
          raise ValueError(f'Inserted token counts do not match at {i}: '
                           f'{len(insertions)} vs. {num_insertions}')
      else:
        insertions = masks

      if base_tag == constants.KEEP:
        if deletion_started:
          deletion_started = False
          masked_tokens.extend([constants.DELETE_SPAN_END] + mask_buffer)
          target_tokens.extend([constants.DELETE_SPAN_END] + target_buffer)
          mask_buffer = []
          target_buffer = []
        if self._insert_after_token:
          masked_tokens.extend([source_token] + masks)
          target_tokens.extend([source_token] + insertions)
        else:
          masked_tokens.extend(masks + [source_token])
          target_tokens.extend(insertions + [source_token])
      else:  # base_tag == constants.DELETE
        if not deletion_started:
          masked_tokens.append(constants.DELETE_SPAN_START)
          target_tokens.append(constants.DELETE_SPAN_START)
        deletion_started = True
        mask_buffer.extend(masks)
        target_buffer.extend(insertions)
        masked_tokens.append(source_token)
        target_tokens.append(source_token)
    if deletion_started:
      # Source ended with deletion so we need to add the buffered tokens.
      masked_tokens.extend([constants.DELETE_SPAN_END] + mask_buffer)
      target_tokens.extend([constants.DELETE_SPAN_END] + target_buffer)

    assert len(masked_tokens) == len(target_tokens), (
        f"Masked token count ({len(masked_tokens)}) doesn't match the "
        f"target token count ({len(target_tokens)}).")

    # Truncate lists. Note that although `source_tokens` is already guaranteed
    # to be truncated, `masked_tokens` can still be too long since it contains
    # added MASK tokens.
    masked_tokens = masked_tokens[:self._max_seq_length]
    target_tokens = target_tokens[:self._max_seq_length]
    # Don't truncate [SEP] to match BERT input format.
    if masked_tokens[-1] != constants.SEP:
      masked_tokens[-1] = constants.SEP
      target_tokens[-1] = constants.SEP

    return masked_tokens, target_tokens

  def build_insertion_example(
      self,
      source_tokens,
      labels,
      target_insertions = None
  ):
    """Constructs the masked input TF Example for the insertion model.

    Args:
      source_tokens: List of source tokens.
      labels: List of edit label tuples (base_tag, num_insertions).
      target_insertions: Inserted target tokens per source token. Only provided
        when constructing training examples.

    Returns:
      A feed_dict containing input features to be fed to a predictor (for
      inference) or to be converted to a tf.Example (for model training). If the
      labels don't contain any insertions, returns None.
    """
    masked_tokens, target_tokens = self.build_insertion_tokens(
        source_tokens, labels, target_insertions)
    if constants.MASK not in masked_tokens:
      # No need to create an insertion example so return None.
      return None
    return utils.build_feed_dict(
        masked_tokens,
        self.tokenizer,
        target_tokens=target_tokens,
        max_seq_length=self._max_seq_length,
        max_predictions_per_seq=self._max_predictions_per_seq)

  def _tokenize_text(self, text):
    """Returns the tokenized text with special start and end tokens."""
    tokens = list(self.tokenizer.tokenize(text))
    # Truncate list and add [CLS] and [SEP] tokens to make the input resemble
    # the data BERT has seen during pretraining and to allow inserting tokens to
    # the beginning and to the end of the sequence regardless of
    # `self._insert_after_token`.
    tokens = tokens[:self._max_seq_length - 2]
    return [constants.CLS] + tokens + [constants.SEP]
