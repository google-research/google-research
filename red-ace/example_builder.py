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

"""Build RED-ACE BERT Examples from text (source, target) pairs."""

import collections
import itertools
import random

import constants
import lcs
import tensorflow as tf


class RedAceExample:
  """Class for training and inference examples for BERT tagging model.

  Attributes:
    features: A dictionary of features with numeral lists as values.
    features_float: A dictionary of features with float lists as values.
    features_int: A dictionary of features with int lists as values.
    scalar_features: A dictionary of features with scalar values.
    debug_features: A dictionary of debug features.
    features_joint: A dictionary of features for the joint model with numeral
      lists as values.
  """

  def __init__(
      self,
      input_ids,
      input_mask,
      segment_ids,
      confidence_scores,
      bucketed_token_confidence_scores,
      labels=None,
      labels_mask=None,
      input_tokens=None,
      label_tokens=None,
      source_text=None,
      target_text=None,
      target_tokens=None,
  ):
    """Initializes an instance of RedAceExample.

    Args:
      input_ids: List of ids of source tokens.
      input_mask: List of 1s and 0s. 0 indicates a PAD token.
      segment_ids: List of segment ids for BERT.
      confidence_scores: Confidence scores.
      bucketed_token_confidence_scores: Bucketed token confidence scores.
      labels: List of label ids. Optional if we are at test time.
      labels_mask: List of 1s and 0s. Optional if we are at test time.
      input_tokens: List of tokens (as text). Optional. For debugging.
      label_tokens: List of labels (as text). Optional. For debugging.
      source_text: Raw string of source input. Optional. For debug purposes.
      target_text: Raw string of target output. Optional. For debug purposes.
      target_tokens: List of target tokens (as text). Optional. For debug
        purposes.
    """
    self.features = collections.OrderedDict([
        ('input_ids', list(input_ids)),
        ('input_mask', list(input_mask)),
        ('segment_ids', list(segment_ids)),
        ('labels', list(labels)),
        ('bucketed_confidence_scores', bucketed_token_confidence_scores),
    ])

    self.features_float = collections.OrderedDict([
        ('labels_mask', labels_mask),
        ('confidence_scores', confidence_scores),
    ])

    self.debug_features = collections.OrderedDict()
    # if store_debug_features:
    if input_tokens is not None:
      self.debug_features['input_tokens'] = input_tokens
    if label_tokens is not None:
      self.debug_features['label_tokens'] = label_tokens
    if source_text is not None:
      self.debug_features['text_source'] = [source_text]
    if target_text is not None:
      self.debug_features['text_target'] = [target_text]
    if target_tokens is not None:
      self.debug_features['target_tokens'] = target_tokens

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
        feature.extend([pad_id] * pad_len)  # pytype: disable=attribute-error
      if len(feature) != max_seq_length:
        raise ValueError(
            f'{key} has length {len(feature)} (should be {max_seq_length}).')

  def _int_feature(self, values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  def _float_feature(self, values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

  def to_tf_example(self):
    """Returns this object as a tf.Example."""

    tf_features = collections.OrderedDict([
        (key, self._int_feature(val)) for key, val in self.features.items()
    ])

    # Add label mask feature.
    for key, value in self.features_float.items():
      tf_features[key] = self._float_feature(value)

    # Add debug fields.
    for key, value in self.debug_features.items():
      tf_features[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[element.encode('utf8') for element in value]))

    return tf.train.Example(features=tf.train.Features(feature=tf_features))


class RedAceExampleBuilder:
  """Builder class for RedAceExamples.

  Attributes:
    label_map: Mapping from tags to tag IDs.
    tokenizer: A tokenization.FullTokenizer, which converts between strings and
      lists of tokens.
  """

  def __init__(self, tokenizer, max_seq_length=128):
    """Initializes an instance of RedAceExampleBuilder.

    Args:
      tokenizer: Tokenizer used to split the sources and targets.
      max_seq_length: Maximum sequence length.
    """
    self.label_map = constants.LABEL_MAP
    self.tokenizer = tokenizer
    self._max_seq_length = max_seq_length
    try:
      self._pad_id = self.tokenizer.convert_tokens_to_ids([constants.PAD])[0]
    except KeyError:
      self._pad_id = 0
    random.seed(10)

  def build_redace_example(
      self,
      source,
      confidence_scores,
      target,
  ):
    """Constructs a RedAceExampleExample.

    Args:
      source: List of source texts.
      confidence_scores: Confidence scores.
      target: Target text or None when building an example during inference. If
        the target is None then we don't calculate gold labels or tags, this is
        equivaltn to setting is_test_time to True.

    Returns:
      RedAceExample.

    Raises:
      KeyError: If a label not in `self.label_map` is produced.
    """
    (
        input_tokens,
        token_confidence_scores,
        bucketed_token_confidence_scores,
    ) = self.tokenize_text(
        source, confidence_scores=confidence_scores)

    input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    output_tokens, _, _ = self.tokenize_text(target)

    edits = compute_edits(input_tokens, output_tokens)
    labels = []  # Labels as IDs.
    for edit in edits:
      if edit in self.label_map:
        labels.append(self.label_map[edit])
      else:
        raise KeyError(f"Label map doesn't contain a computed label: {edit}")

    label_counter = collections.Counter(labels)
    label_weight = {
        label: len(labels) / count / len(label_counter)
        for label, count in label_counter.items()
    }
    # Weight the labels inversely proportional to their frequency.
    labels_mask = [label_weight[label] for label in labels]
    # When inserting after the current token, we never need to insert after
    # the final [SEP] token and thus the edit label for that token is
    # constant ('KEEP') and could be excluded from loss computations.
    labels_mask[-1] = 0

    example = RedAceExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence_scores=token_confidence_scores,
        bucketed_token_confidence_scores=bucketed_token_confidence_scores,
        labels=labels,
        labels_mask=labels_mask,
        input_tokens=input_tokens,
        label_tokens=edits,
        source_text=source,
        target_text=target,
        target_tokens=output_tokens,
    )
    example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example

  def _add_start_end_tokens(
      self,
      value_list,
      start_value,
      end_value,
  ):
    return [start_value] + value_list[:self._max_seq_length - 2] + [end_value]

  def tokenize_text(self, text, confidence_scores=None):
    """Returns the tokenized text with special start and end tokens."""
    text = text.lower()
    # [SEP] and [CLS] Should always be uppercase.
    text = text.replace(constants.SEP.lower(), constants.SEP)
    text = text.replace(constants.CLS.lower(), constants.CLS)

    tokens, token_confidence_scores = self._split_to_wordpieces(
        text.split(), confidence_scores)

    if token_confidence_scores is not None:
      assert len(token_confidence_scores) == len(tokens), (
          'Tokenized text and confidence scores are of different lengths: {}, '
          '{}'.format(tokens, token_confidence_scores))
      token_confidence_scores = self._add_start_end_tokens(
          token_confidence_scores,
          constants.UNDEFINED_CONFIDENCE_SCORE,
          constants.UNDEFINED_CONFIDENCE_SCORE,
      )
      bucketed_token_confidence_scores = [
          int(v * 9) if v > 0 else 9 for v in token_confidence_scores
      ]
    else:
      bucketed_token_confidence_scores = []

    return (
        self._add_start_end_tokens(tokens, constants.CLS, constants.SEP),
        token_confidence_scores,
        bucketed_token_confidence_scores,
    )

  def _split_to_wordpieces(
      self,
      tokens,
      confidence_scores=None,
  ):
    """Splits tokens to WordPieces.

    Args:
      tokens: Tokens to be split.
      confidence_scores: Confidence scores.

    Returns:
      List of WordPieces.
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    special_tokens = {'<::::>', constants.SEP.lower(), constants.CLS.lower()}

    if confidence_scores is not None:
      assert len(confidence_scores) == len(tokens), (
          'Input text and confidence scores are of different lengths: {}, {}'
          .format(tokens, confidence_scores))
      token_confidence_scores = []
      for token, score in zip(tokens, confidence_scores):
        # Don't tokenize special tokens.
        if token.lower() not in special_tokens:
          pieces = self.tokenizer.tokenize(token)
          token_confidence_scores.extend([score] * len(pieces))
        else:
          pieces = [token]
          token_confidence_scores.append(constants.UNDEFINED_CONFIDENCE_SCORE)

        bert_tokens.extend(pieces)
    else:
      token_confidence_scores = None
      for token in tokens:
        # Don't tokenize special tokens.
        if token.lower() not in special_tokens:
          pieces = self.tokenizer.tokenize(token)
        else:
          pieces = [token]

        bert_tokens.extend(pieces)

    return bert_tokens, token_confidence_scores


def compute_edits(
    source_tokens,
    target_tokens,
):
  """Computes edit operations per source token.

  Args:
    source_tokens: List of source tokens.
    target_tokens: List of target tokens.

  Returns:
    List of edit operations ("KEEP" or "DELETE"), one per source token.
  """
  return _get_edits(
      lcs.compute_lcs(source_tokens, target_tokens), source_tokens)


def _get_edits(kept_tokens, source_tokens):
  """Returns edit operation per source token."""
  edit_operations = []
  kept_idx = 0
  for token in source_tokens:
    if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
      kept_idx += 1
      edit_operations.append(constants.KEEP)
    else:
      edit_operations.append(constants.DELETE)

  return edit_operations
