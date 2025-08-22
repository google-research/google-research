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

"""Library for Example class."""
import collections
import dataclasses
from typing import Any, Iterator, MutableSequence, Sequence, Tuple, Union

import ml_collections
import numpy as np
import tensorflow as tf

from mave.benchmark.utils import tokenization


@dataclasses.dataclass
class Paragraph:
  """A tokenized paragraph with attribute token level spans."""
  text: str
  source: str
  tokens: Sequence[str] = ()
  # Character level spans, inclusive indices.
  char_spans: Sequence[Tuple[int, int]] = ()
  # Character level spans, inclusive indices.
  token_spans: Sequence[Tuple[int, int]] = ()


@dataclasses.dataclass
class Example:
  """A example containing multiple tokenized paragraphs."""
  id: str
  category: str
  attribute_key: str
  paragraphs: Sequence[Paragraph] = ()

  @classmethod
  def from_json_example(cls, json_example):
    """Yields Examples from a json example loaded from file."""
    for attribute in json_example['attributes']:
      # If performance is an issue, it is also possible to tokenize paragraphs
      # with evidence char spans from all attributes together. Here for
      # readability purpose, we repeat the tokenization for each attribute.
      paragraphs = [
          Paragraph(p['text'], p['source']) for p in json_example['paragraphs']
      ]

      char_spans_by_pid = collections.defaultdict(list)
      for evidence in attribute.get('evidences', []):
        # Our tokenization util uses inclusive [begin, end] indices, while the
        # JSON lines data contains [begin, end) indices.
        char_spans_by_pid[evidence['pid']].append(
            (evidence['begin'], evidence['end'] - 1))

      for pid, char_spans in char_spans_by_pid.items():
        paragraphs[pid].char_spans = char_spans

      yield cls(
          json_example['id'],
          json_example['category'],
          attribute['key'],
          paragraphs=paragraphs)

  def tokenize(self,
               tokenizer):
    """Tokenizes paragraph texts and converts char spans to token spans."""
    for paragraph in self.paragraphs:
      paragraph.tokens, paragraph.token_spans = (
          tokenizer.tokenize(paragraph.text, paragraph.char_spans))


@dataclasses.dataclass
class InputFeatures:
  """Input features for the BERT or BiLSTM-CRF sequence tagging model."""
  config: ml_collections.FrozenConfigDict
  id: str
  input_ids: MutableSequence[int] = dataclasses.field(default_factory=list)
  input_mask: MutableSequence[int] = dataclasses.field(default_factory=list)
  # Not used for BiLSTM model.
  segment_ids: MutableSequence[int] = dataclasses.field(default_factory=list)
  label_ids: MutableSequence[int] = dataclasses.field(default_factory=list)
  # Required for inference.
  tokens: MutableSequence[str] = dataclasses.field(default_factory=list)

  @property
  def num_tokens(self):
    return len(self.input_ids)

  @property
  def space_left(self):
    """Num of more tokens can be added to the feature."""
    return self.config.model.seq_length - self.num_tokens

  def add_cls(self, segment_id):
    """Adds an [CLS] to the feature."""
    if not self.config.data.use_cls or self.space_left == 0:
      return
    self.input_ids.append(self.config.model.cls_id)
    self.input_mask.append(1)
    self.segment_ids.append(segment_id)
    self.label_ids.append(0)

  def add_sep(self, segment_id):
    """Adds an [SEP] to the feature."""
    if not self.config.data.use_sep or self.space_left == 0:
      return
    self.input_ids.append(self.config.model.sep_id)
    self.input_mask.append(1)
    self.segment_ids.append(segment_id)
    self.label_ids.append(0)

  def add_segment(self, token_ids, segment_id,
                  label_ids):
    """Adds a sequence of tokens."""
    end_sep = int(self.config.data.use_sep)
    num_new_tokens = min(len(token_ids), max(self.space_left - int(end_sep), 0))
    self.input_ids.extend(token_ids[:num_new_tokens])
    self.input_mask.extend([1] * num_new_tokens)
    self.segment_ids.extend([segment_id] * num_new_tokens)
    self.label_ids.extend(label_ids[:num_new_tokens])

  def pad(self):
    """Pads features to `seq_length`."""
    assert self.space_left >= 0, 'Too many tokens added.'
    num_paddings = self.space_left
    self.input_ids.extend([0] * num_paddings)
    self.input_mask.extend([0] * num_paddings)
    self.segment_ids.extend([0] * num_paddings)
    self.label_ids.extend([0] * num_paddings)

  def to_tf_example(self):
    """Returns a tf.train.Example."""
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'id':
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[self.id.encode()])),
                'input_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=self.input_ids)),
                'input_mask':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=self.input_mask)),
                'segment_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=self.segment_ids)),
                'label_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=self.label_ids)),
                'tokens':
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[t.encode() for t in self.tokens])),
            }))


class TFRecordConverter:
  """JSON example to TF Reccord converter."""

  def __init__(self, config):
    """Initializes the converter."""
    self._config = config
    self._tokenizer = tokenization.IndexMappingWordpieceTokenizer(
        config.bert.vocab_file)

  def convert(self, json_example):
    for example in Example.from_json_example(json_example):
      example.tokenize(self._tokenizer)
      yield self._create_tf_record(example)

  def _create_tf_record(self, example):
    """Creates a TF Record from an Example."""
    features = InputFeatures(self._config, example.id)
    # Adds a [ClS] token.
    features.add_cls(segment_id=0)

    # Adds category tokens followed by a [SEP].
    if self._config.data.use_category:
      category_tokens, _ = self._tokenizer.tokenize(example.category)
      category_token_ids = self._tokenizer.convert_tokens_to_ids(
          category_tokens)
      features.add_segment(
          category_token_ids,
          segment_id=self._config.model.category_attribute_type_id,
          label_ids=np.zeros_like(category_token_ids))
      features.add_sep(segment_id=self._config.model.category_attribute_type_id)

    # Adds attribute key tokens followed by a [SEP].
    if self._config.data.use_attribute_key:
      attribute_key_tokens, _ = self._tokenizer.tokenize(example.attribute_key)
      attribute_key_token_ids = self._tokenizer.convert_tokens_to_ids(
          attribute_key_tokens)
      features.add_segment(
          attribute_key_token_ids,
          segment_id=self._config.model.category_attribute_type_id,
          label_ids=np.zeros_like(attribute_key_token_ids))
      features.add_sep(segment_id=self._config.model.category_attribute_type_id)

    # Adds paragraph tokens followed by a [SEP].
    for paragraph in example.paragraphs:
      token_ids = self._tokenizer.convert_tokens_to_ids(paragraph.tokens)
      label_ids = np.zeros_like(token_ids)
      for begin, end in paragraph.token_spans:
        label_ids[begin:end + 1] = 1
      features.add_segment(
          token_ids,
          segment_id=self._config.model.paragraph_type_id,
          label_ids=label_ids)
    features.add_sep(segment_id=1)

    features.pad()

    if self._config.data.debug:
      features.tokens = self._tokenizer.convert_ids_to_tokens(
          features.input_ids)

    return features.to_tf_example()


@dataclasses.dataclass
class EtcInputFeatures:
  """Input features for the ETC sequenec tagging model."""
  config: ml_collections.FrozenConfigDict
  id: str
  global_token_ids: MutableSequence[int] = dataclasses.field(
      default_factory=list)
  global_breakpoints: MutableSequence[int] = dataclasses.field(
      default_factory=list)
  global_token_type_ids: MutableSequence[int] = dataclasses.field(
      default_factory=list)
  global_label_ids: MutableSequence[int] = dataclasses.field(
      default_factory=list)
  long_token_ids: MutableSequence[int] = dataclasses.field(default_factory=list)
  long_breakpoints: MutableSequence[int] = dataclasses.field(
      default_factory=list)
  long_token_type_ids: MutableSequence[int] = dataclasses.field(
      default_factory=list)
  # The paragraph ids for the long tokens. Each id `i` corresponds to a
  # paragraph global token, `global_token_ids[i]`. Category tokens and attribute
  # key tokens form two paragraphs.
  long_paragraph_ids: MutableSequence[int] = dataclasses.field(
      default_factory=list)
  long_label_ids: MutableSequence[int] = dataclasses.field(default_factory=list)

  long_tokens: MutableSequence[str] = dataclasses.field(default_factory=list)
  global_tokens: MutableSequence[str] = dataclasses.field(default_factory=list)

  @property
  def num_long_tokens(self):
    return len(self.long_token_ids)

  @property
  def num_global_tokens(self):
    return len(self.global_token_ids)

  @property
  def long_space_left(self):
    """Num of more long tokens can be added to the feature."""
    return self.config.etc.long_seq_length - self.num_long_tokens

  @property
  def global_space_left(self):
    """Num of more global tokens can be added to the feature."""
    return self.config.etc.global_seq_length - self.num_global_tokens

  def add_cls(self):
    """Adds an [CLS] token to the global feature."""
    if self.global_space_left == 0:
      return
    self.global_token_ids.append(self.config.model.cls_id)
    self.global_breakpoints.append(0)
    self.global_token_type_ids.append(self.config.etc.global_token_type_id)
    self.global_label_ids.append(0)

  def add_global_token(self, global_token_id, global_label_id):
    """Adds a globel token."""
    if self.global_space_left == 0:
      return
    self.global_token_ids.append(global_token_id)
    self.global_breakpoints.append(0)
    self.global_token_type_ids.append(self.config.etc.global_token_type_id)
    self.global_label_ids.append(global_label_id)

  def add_segment(self, global_token_id, long_token_ids,
                  long_token_type_id, log_label_ids):
    """Adds a sequence of long tokens and a corresponding global token."""
    if self.global_space_left == 0 or self.long_space_left == 0:
      return
    num_new_long_tokens = min(len(long_token_ids), self.long_space_left)
    global_label_id = int(any(log_label_ids[:num_new_long_tokens]))
    self.add_global_token(global_token_id, global_label_id)
    self.long_token_ids.extend(long_token_ids[:num_new_long_tokens])
    self.long_breakpoints.extend([0] * num_new_long_tokens)
    self.long_breakpoints[-1] = 1
    self.long_token_type_ids.extend([long_token_type_id] * num_new_long_tokens)
    paragraph_id = self.num_global_tokens - 1
    self.long_paragraph_ids.extend([paragraph_id] * num_new_long_tokens)
    self.long_label_ids.extend(log_label_ids[:num_new_long_tokens])

  def pad(self):
    """Pads features to `global_seq_length` and `long_seq_length`."""
    assert self.global_space_left >= 0, 'Too many global tokens added.'
    assert self.long_space_left >= 0, 'Too many long tokens added.'
    num_global_paddings = self.global_space_left
    self.global_token_ids.extend([0] * num_global_paddings)
    self.global_breakpoints[-1] = 1
    self.global_breakpoints.extend([0] * num_global_paddings)
    self.global_token_type_ids.extend([0] * num_global_paddings)
    self.global_label_ids.extend([0] * num_global_paddings)
    num_long_paddings = self.long_space_left
    self.long_token_ids.extend([0] * num_long_paddings)
    self.long_breakpoints.extend([0] * num_long_paddings)
    self.long_token_type_ids.extend([0] * num_long_paddings)
    self.long_paragraph_ids.extend([-1] * num_long_paddings)
    self.long_label_ids.extend([0] * num_long_paddings)

  def assert_shape(self):
    """Asserts shapes."""
    assert len(self.global_token_ids) == self.config.etc.global_seq_length
    assert len(self.global_breakpoints) == self.config.etc.global_seq_length
    assert len(self.global_token_type_ids) == self.config.etc.global_seq_length
    assert len(self.global_label_ids) == self.config.etc.global_seq_length
    assert len(self.long_token_ids) == self.config.etc.long_seq_length
    assert len(self.long_breakpoints) == self.config.etc.long_seq_length
    assert len(self.long_token_type_ids) == self.config.etc.long_seq_length
    assert len(self.long_paragraph_ids) == self.config.etc.long_seq_length
    assert len(self.long_label_ids) == self.config.etc.long_seq_length
    if self.config.data.debug:
      assert len(self.global_tokens) == self.config.etc.global_seq_length
      assert len(self.long_tokens) == self.config.etc.long_seq_length

  def to_tf_example(self):
    """Returns a tf.train.Example."""
    self.assert_shape()
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'id':
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[self.id.encode()])),
                'global_token_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.global_token_ids)),
                'global_breakpoints':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.global_breakpoints)),
                'global_token_type_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.global_token_type_ids)),
                'global_label_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.global_label_ids)),
                'long_token_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.long_token_ids)),
                'long_breakpoints':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.long_breakpoints)),
                'long_token_type_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.long_token_type_ids)),
                'long_paragraph_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.long_paragraph_ids)),
                'long_label_ids':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.long_label_ids)),
                'long_tokens':
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[t.encode() for t in self.long_tokens])),
                'global_tokens':
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[t.encode() for t in self.global_tokens])),
            }))


class EtcTFRecordConverter:
  """JSON example to ETC TF Reccord converter."""

  def __init__(self, config):
    """Initializes the converter."""
    self._config = config
    self._tokenizer = tokenization.IndexMappingWordpieceTokenizer(
        config.etc.vocab_file)

  def convert(self, json_example):
    for example in Example.from_json_example(json_example):
      example.tokenize(self._tokenizer)
      yield self._create_tf_record(example)

  def _create_tf_record(self, example):
    """Creates a ETC TF Record from an Example."""
    features = EtcInputFeatures(self._config, example.id)
    # Adds a [ClS] token to global.
    features.add_cls()

    # Adds category tokens.
    category_tokens, _ = self._tokenizer.tokenize(example.category)
    category_token_ids = self._tokenizer.convert_tokens_to_ids(category_tokens)
    features.add_segment(self._config.etc.category_global_token_id,
                         category_token_ids,
                         self._config.etc.category_token_type_id,
                         np.zeros_like(category_token_ids))

    # Adds attribute key tokens followed.
    attribute_key_tokens, _ = self._tokenizer.tokenize(example.attribute_key)
    attribute_key_token_ids = self._tokenizer.convert_tokens_to_ids(
        attribute_key_tokens)
    features.add_segment(self._config.etc.attribute_global_token_id,
                         attribute_key_token_ids,
                         self._config.etc.attribute_token_type_id,
                         np.zeros_like(attribute_key_token_ids))

    # Adds paragraph tokens
    for paragraph in example.paragraphs:
      long_token_ids = self._tokenizer.convert_tokens_to_ids(paragraph.tokens)
      long_label_ids = np.zeros_like(long_token_ids)
      for begin, end in paragraph.token_spans:
        long_label_ids[begin:end + 1] = 1
      if paragraph.source == 'title':
        long_token_type_id = self._config.etc.title_token_type_id
      elif paragraph.source == 'description':
        long_token_type_id = self._config.etc.description_token_type_id
      else:
        long_token_type_id = self._config.etc.other_token_type_id
      features.add_segment(self._config.etc.paragraph_global_token_id,
                           long_token_ids, long_token_type_id, long_label_ids)

    features.pad()

    if self._config.data.debug:
      features.global_tokens = self._tokenizer.convert_ids_to_tokens(
          features.global_token_ids)
      features.long_tokens = self._tokenizer.convert_ids_to_tokens(
          features.long_token_ids)

    return features.to_tf_example()


def get_tf_record_converter(
    config
):
  if config.model_type in ['bert', 'bilstm_crf']:
    return TFRecordConverter(config)
  elif config.model_type == 'etc':
    return EtcTFRecordConverter(config)
  else:
    raise ValueError(f'Unsupported model type: {config.model_type!r}')
