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

"""Data processing utility functions."""
import bisect
import collections
import copy
import enum
import functools
import math
import random
from typing import Iterable, Iterator, List, Optional, Text, Union

import attr
from bert import tokenization as bert_tokenization
import numpy as np
import tensorflow.compat.v1 as tf

from readtwice.data_utils import tokenization


def create_int_feature(values):
  """Creates TensorFlow int features.

  Args:
    values: A sequence of integers.

  Returns:
    An entry of int tf.train.Feature.
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


@functools.total_ordering
@attr.s
class Annotation(object):
  """Span-based annotation over a sentence."""
  # Begin index of the annotation
  begin = attr.ib()  # type: int

  # End index of the annotation (inclusive)
  end = attr.ib()  # type: int

  # Surface form of the annotation. Ideally, should be equal to the
  # text[start_index: end_index + 1]
  text = attr.ib(default=None)  # type: Optional[Text]

  # Label for the annotation.
  label = attr.ib(default=None)  # type: int

  type = attr.ib(default=0)

  def __eq__(self, other):
    return (self.begin, self.end) == (other.begin, other.end)

  def __ne__(self, other):
    return not self == other

  def __lt__(self, other):
    return (self.begin, self.end) < (other.begin, other.end)


@attr.s
class Sentence(object):
  """Raw sentence."""

  # Raw text of the sentence
  text = attr.ib()  # type: Text

  # (Optional) Annotations (in a sorted order).
  # Note that annotation begin / end indices are based on unicode code points
  # of the `text`.
  annotations = attr.ib(type=List[Annotation], factory=list)

  def __len__(self):
    return len(self.text)

  def strip_whitespaces(self):
    """Strip whitespaces from the sentence and update annotations if needed."""
    while self.text and self.text[0].isspace():
      self.text = self.text[1:]
      # Remove all annotations that will disappear after this operation
      self.annotations = list(
          filter(lambda a: a.begin > 0 or a.end > 0, self.annotations))
      for i in range(len(self.annotations)):
        if self.annotations[i].begin == 0:
          self.annotations[i].text = self.annotations[i].text[1:]
        self.annotations[i].begin = max(self.annotations[i].begin - 1, 0)
        self.annotations[i].end = max(self.annotations[i].end - 1, 0)
    while self.text and self.text[-1].isspace():
      self.text = self.text[:-1]
      # Remove all annotations that will disappear after this operation
      # pylint: disable=g-long-lambda
      self.annotations = list(
          filter(lambda a: a.begin < len(self.text) or a.end < len(self.text),
                 self.annotations))
      # pylint: enable=g-long-lambda
      for i in range(len(self.annotations)):
        if self.annotations[i].end == len(self.text):
          self.annotations[i].end = max(self.annotations[i].end - 1, 0)
    return self

  def num_annotations(self, annotation_type=None):
    counter = 0
    for annotation in self.annotations:
      if annotation_type is None or annotation.type == annotation_type:
        counter += 1
    return counter


class BertDocument(object):
  """A collection of sentences to be used as input to preprocessing.

  Attributes:
    document_id: An (optional) integer ID of the document.
    sentences: A list of Sentences objects -- the actual content of the document
  """

  def __init__(self,
               sentences,
               document_id = None):
    self.document_id = document_id

    if not sentences:
      self.sentences = []
    elif isinstance(sentences[0], str):
      self.sentences = [Sentence(sentence) for sentence in sentences]
    else:
      self.sentences = sentences

  def num_characters(self):
    """Computes the total number of characters (unicode code points)."""
    return sum(len(s.text) for s in self.sentences)

  def __len__(self):
    return len(self.sentences)

  def __eq__(self, other):
    if len(self) != len(other) or self.document_id != other.document_id:
      return False
    for i in range(len(self)):
      if self.sentences[i] != other.sentences[i]:
        return False
    return True

  def __str__(self):
    return 'BertDocument(id=%s, sentences=%s)' % (str(
        self.document_id), str(self.sentences))

  def num_annotations(self, annotation_type=None):
    return sum(s.num_annotations(annotation_type) for s in self.sentences)


@attr.s
class TokenizedSentence(object):
  """A tokenized sentence."""

  # The ids of the tokens in the sentence.
  token_ids = attr.ib()  # type: List[int]

  # (Optional) a list containing only 0 or 1 values. 1 for any token that's a
  # continuation of a word and 0 otherwise. Must be the same length as
  # `token_ids` if present.  For example, if "extensively" is broken into the
  # following tokens: ["exten", "##sive", "##ly"], then
  # is_continuation:  [      0,        1,      1]
  is_continuation = attr.ib(default=None)  # type: Optional[List[int]]

  # (Optional) the string tokens in the sentence corresponding to the token ids.
  # This may be helpful to include for debugging. Must be the same length as
  # `token_ids` if present.
  tokens = attr.ib(default=None)  # type: Optional[List[Text]]

  # (Optional) the raw sentence text. May be useful for inspecting the original
  # sentence.
  raw_text = attr.ib(default=None)  # type: Optional[Text]

  # (Optional) Annotations (in a sorted order).
  # Note that annotation begin / end indices are based on the tokens
  # (`token_ids` or `tokens`), not unicode code points
  # (like it is in `Sentence` class).
  annotations = attr.ib(default=None)  # type: Optional[List[Annotation]]

  def __len__(self):
    return len(self.token_ids)

  # Note, this is a linear, not logarithmic time function. That's intentional
  # since we have to iterate over all found annotations anyway.
  def get_annotations(
      self,
      begin,
      end,
      only_strictly_inside = True):
    """Get annotations within a given internval."""
    for annotation in self.annotations or []:
      if only_strictly_inside:
        if begin <= annotation.begin and annotation.end <= end:
          yield annotation
      else:
        if ((begin <= annotation.begin and annotation.begin <= end) or
            (begin <= annotation.end and annotation.end <= end)):
          yield annotation

  def num_annotations(self, annotation_type=None):
    counter = 0
    for annotation in self.annotations:
      if annotation_type is None or annotation.type == annotation_type:
        counter += 1
    return counter


@enum.unique
class BertDocumentGlobalMode(enum.Enum):
  """Specifies how global tokens are generated for BERT documents."""

  # Allocates a global token for every sentence.
  SENTENCE = 1

  # Allocates a global token for every fixed-length block of WordPieces
  # (e.g. 1 global token per 20 WordPiece tokens).
  FIXED_BLOCKS = 2


@attr.s
class TokenizedBertDocument(object):
  """A tokenized document divided by sentences, as in BERT pretraining."""

  # A list of tokenized sentences.
  sentences = attr.ib()  # type: List[TokenizedSentence]

  # Document ID
  document_id = attr.ib(default=None)  # type: int

  def num_tokens(self):
    """Returns the total number of tokens in this document."""
    return sum(len(sentence) for sentence in self.sentences)

  def num_annotations(self, annotation_type=None):
    return sum(s.num_annotations(annotation_type) for s in self.sentences)

  def to_tf_example(
      self,
      global_sentence_token_id,
      global_mode = BertDocumentGlobalMode.SENTENCE,
      fixed_blocks_num_tokens_per_block = 20,
      include_global_cls_token = False,
      global_cls_token_id = None):
    """Returns a TensorFlow Example for this document.

    The features will be neither padded nor truncated, so different examples
    will have different length features.

    Args:
      global_sentence_token_id: Token id for all global sentence tokens.
      global_mode: `BertDocumentGlobalMode` specifying how global tokens are
        generated for the example. The default SENTENCE mode generates 1 global
        token per natural language sentence.
      fixed_blocks_num_tokens_per_block: Only applicable for FIXED_BLOCKS
        `global_mode`. The number of WordPiece tokens per global token.
      include_global_cls_token: Whether to include a global CLS token at the
        beginning of the global memory.
      global_cls_token_id: Token id for all global CLS tokens. Must not be None
        if `include_global_cls_token` is True.

    Returns:
      A tf.train.Example object.

    Raises:
      ValueError: If `include_global_cls_token` is True but
        `global_cls_token_id` isn't given.
    """
    if include_global_cls_token and global_cls_token_id is None:
      raise ValueError(
          'Must specify `global_cls_token_id` if `include_global_cls_token` is '
          'True.')
    token_ids = []
    is_continuation = []
    sentence_ids = []
    global_token_ids = []
    if include_global_cls_token:
      global_token_ids.append(global_cls_token_id)
    for i, sentence in enumerate(self.sentences):
      token_ids.extend(sentence.token_ids)
      is_continuation.extend(sentence.is_continuation)
      if global_mode == BertDocumentGlobalMode.SENTENCE:
        sentence_id = i + 1 if include_global_cls_token else i
        sentence_ids.extend([sentence_id] * len(sentence))
        global_token_ids.append(global_sentence_token_id)

    if global_mode == BertDocumentGlobalMode.FIXED_BLOCKS:
      num_blocks = len(token_ids) // fixed_blocks_num_tokens_per_block
      if len(token_ids) % fixed_blocks_num_tokens_per_block != 0:
        num_blocks += 1
      for i in range(num_blocks):
        sentence_id = i + 1 if include_global_cls_token else i
        sentence_ids.extend([sentence_id] * fixed_blocks_num_tokens_per_block)
        global_token_ids.append(global_sentence_token_id)
      # Truncate any remainder in the last block.
      sentence_ids = sentence_ids[:len(token_ids)]

    feature_map = dict(
        token_ids=create_int_feature(token_ids),
        is_continuation=create_int_feature(is_continuation),
        sentence_ids=create_int_feature(sentence_ids),
        global_token_ids=create_int_feature(global_token_ids))
    return tf.train.Example(features=tf.train.Features(feature=feature_map))

  def to_tf_strided_large_example(
      self,
      overlap_length,
      block_length,
      padding_token_id,
      prefix_token_ids,
      max_num_annotations = None,
      answer_only_strictly_inside_annotations = True,
      entity_only_strictly_inside_annotations = False,
      default_annotation_label = 1):
    """Returns a TensorFlow Example for this document.

    The function prepares a "large" Example, which is supposed to be
    a concatenation of smaller examples of length `block_length`,
    so the features could always be reshaped into
    (`effective_batch_size`, `block_length`).

    The features will always be padded such that the length is divisible by
    `block_length`.

    Moreover, every block
    (1) will be prepended with a `prefix_token_ids` if provided. In this case
    an additional feaure "prefix_length" will be saved as part of the
    output tf.Example.
    (2) One can make block overlap by specifying `overlap_length`. Note, that
    stride prefix never cut actual words in half.
    (3) If `max_num_annotations` is set then annotations will also be provided
    for every block, including `annotation_begins`, `annotation_ends` and
    `annotation_labels`. The last feature indicates whether
    an annotation is padding (0) or an actual annotation (1).

    Note, currently we don't include annotations that appear in the "overlap"
    section of the block unless they are from the same sentence.

    Args:
      overlap_length: The overlap length (should be non-negative).
      block_length: Length of a single block (should be positive)
      padding_token_id: Token id for padding tokens.
      prefix_token_ids: Token ids to be prepended for every block. If it equal
        to None then no tokens will not be prepended.
      max_num_annotations: Maximum number of annotations per block.
      answer_only_strictly_inside_annotations: Whether to include annotations
         that only partially contained in the block (for answer annotations).
      entity_only_strictly_inside_annotations: Whether to include annotations
         that only partially contained in the block (for entity annotations).
      default_annotation_label: Default label for annotations without label.

    Returns:
      A tf.train.Example object.

    Raises:
      ValueError: If `overlap_length` is larger than `block_length`.
    """
    if prefix_token_ids is None:
      prefix_token_ids = []
    elif isinstance(prefix_token_ids, int):
      prefix_token_ids = [prefix_token_ids]
    if overlap_length < 0:
      raise ValueError('`overlap_length` should be non-negative')
    if block_length < 0:
      raise ValueError('`block_length` should be positive')
    if len(prefix_token_ids) + overlap_length >= block_length:
      raise ValueError('`overlap_length` is larger than `block_length`')
    if max_num_annotations and max_num_annotations < 0:
      raise ValueError('`max_num_annotations` should be posiive')

    token_ids = []
    is_continuation = []
    block_ids = []

    if prefix_token_ids:
      prefix_length = []

    if max_num_annotations:
      (answer_annotation_begins, answer_annotation_ends,
       answer_annotation_labels) = [], [], []
      (entity_annotation_begins, entity_annotation_ends,
       entity_annotation_labels) = [], [], []
      # All annotation_begins, annotation_ends, annotation_labels
      # should have the length exactly equal to expected_annotations_length.
      # Every time we add new block, we will increase
      # expected_annotations_length by max_num_annotations
      answer_expected_annotations_length = 0
      entity_expected_annotations_length = 0

    raw_token_ids = collections.deque()
    raw_is_continuation = collections.deque()
    current_block_length = 0

    for sentence in self.sentences:
      start_index = 0
      while start_index < len(sentence):
        # (1) Initialize the current block:
        if current_block_length == 0:
          if max_num_annotations:
            answer_expected_annotations_length += max_num_annotations
            entity_expected_annotations_length += max_num_annotations

          # (1.1) Add prefix if one was specified
          if prefix_token_ids:
            token_ids.extend(prefix_token_ids)
            is_continuation.extend([0] * len(prefix_token_ids))
            current_block_length += len(prefix_token_ids)
            prefix_length.append(len(prefix_token_ids))

          # (1.2) Prepend stride prefix from previous sentences
          assert len(raw_token_ids) == len(raw_is_continuation)
          while (raw_token_ids and (len(raw_token_ids) > overlap_length or
                                    raw_is_continuation[0] == 1)):
            raw_token_ids.popleft()
            raw_is_continuation.popleft()

          token_ids.extend(raw_token_ids)
          is_continuation.extend(raw_is_continuation)
          block_ids.append(self.document_id or 1)
          current_block_length += len(raw_token_ids)

        # (2) Add the current sentence to the current block
        length_to_add = min(
            len(sentence) - start_index, block_length - current_block_length)
        assert length_to_add > 0
        sentence_slice = slice(start_index, start_index + length_to_add)
        token_ids.extend(sentence.token_ids[sentence_slice])
        raw_token_ids.extend(sentence.token_ids[sentence_slice])
        is_continuation.extend(sentence.is_continuation[sentence_slice])
        raw_is_continuation.extend(sentence.is_continuation[sentence_slice])
        if max_num_annotations and sentence.annotations:
          # Include annotations from the current sentence.
          # Even though we're including a slice of the sentence
          # [start_index; start_index + length_to_add - 1] we need to
          # consider a bigger slice while searching for relevant annotations
          # since a part of the sentence could have been included as part of
          # the overlap section. Therefore, we consider annotations from the
          # following interval
          # [start_index - overlap_length; start_index + length_to_add - 1]
          for annotation in sentence.get_annotations(
              start_index - overlap_length, start_index + length_to_add - 1,
              answer_only_strictly_inside_annotations):
            if len(
                answer_annotation_begins) >= answer_expected_annotations_length:
              break
            if annotation.type != 0:
              continue
            answer_annotation_relative_begin = (
                annotation.begin + current_block_length - start_index)
            answer_annotation_relative_end = (
                annotation.end + current_block_length - start_index)
            # However, sometimes we include less than `overlap_length` tokens
            # in the overlap section because we try not to cut whole words.
            # So there could be a case that we included annotation that
            # does NOT start in the current block. Note that it could only
            # happen when annotation starts at a token with is_continuation=1.
            # It's easy to account for this case because
            # annotation_relative_begin would appear in the "prefix" section
            # of the block.
            if answer_only_strictly_inside_annotations:
              if answer_annotation_relative_begin < len(prefix_token_ids):
                continue
              if answer_annotation_relative_end >= block_length:
                continue
            else:
              answer_annotation_relative_begin = max(
                  answer_annotation_relative_begin, len(prefix_token_ids))
              answer_annotation_relative_end = min(
                  answer_annotation_relative_end, block_length - 1)
              if answer_annotation_relative_begin > answer_annotation_relative_end:
                continue
            answer_annotation_begins.append(answer_annotation_relative_begin)
            answer_annotation_ends.append(answer_annotation_relative_end)
            answer_annotation_labels.append(annotation.label or
                                            default_annotation_label)

          for annotation in sentence.get_annotations(
              start_index - overlap_length, start_index + length_to_add - 1,
              entity_only_strictly_inside_annotations):
            if len(
                entity_annotation_begins) >= entity_expected_annotations_length:
              break
            if annotation.type != 1:
              continue
            entity_annotation_relative_begin = (
                annotation.begin + current_block_length - start_index)
            entity_annotation_relative_end = (
                annotation.end + current_block_length - start_index)
            # However, sometimes we include less than `overlap_length` tokens
            # in the overlap section because we try not to cut whole words.
            # So there could be a case that we included annotation that
            # does NOT start in the current block. Note that it could only
            # happen when annotation starts at a token with is_continuation=1.
            # It's easy to account for this case because
            # annotation_relative_begin would appear in the "prefix" section
            # of the block.
            if entity_only_strictly_inside_annotations:
              if entity_annotation_relative_begin < len(prefix_token_ids):
                continue
              if entity_annotation_relative_end >= block_length:
                continue
            else:
              entity_annotation_relative_begin = max(
                  entity_annotation_relative_begin, len(prefix_token_ids))
              entity_annotation_relative_end = min(
                  entity_annotation_relative_end, block_length - 1)
              if entity_annotation_relative_begin > entity_annotation_relative_end:
                continue
            entity_annotation_begins.append(entity_annotation_relative_begin)
            entity_annotation_ends.append(entity_annotation_relative_end)
            entity_annotation_labels.append(annotation.label or
                                            default_annotation_label)

        start_index += length_to_add
        current_block_length += length_to_add

        # (3) End of the current block.
        if current_block_length == block_length:
          current_block_length = 0

          if max_num_annotations:
            while len(
                answer_annotation_begins) < answer_expected_annotations_length:
              answer_annotation_begins.append(0)
              answer_annotation_ends.append(0)
              answer_annotation_labels.append(0)
            assert len(
                answer_annotation_begins) == answer_expected_annotations_length
            assert len(
                answer_annotation_ends) == answer_expected_annotations_length
            assert len(
                answer_annotation_labels) == answer_expected_annotations_length

            while len(
                entity_annotation_begins) < entity_expected_annotations_length:
              entity_annotation_begins.append(0)
              entity_annotation_ends.append(0)
              entity_annotation_labels.append(0)
            assert len(
                entity_annotation_begins) == entity_expected_annotations_length
            assert len(
                entity_annotation_ends) == entity_expected_annotations_length
            assert len(
                entity_annotation_labels) == entity_expected_annotations_length

    # (4) Add padding to make the total length divisible by block_length
    padding_length = block_length - current_block_length
    if current_block_length > 0 and padding_length > 0:
      token_ids.extend([padding_token_id] * padding_length)
      is_continuation.extend([0] * padding_length)
      if max_num_annotations:
        while len(
            entity_annotation_begins) < entity_expected_annotations_length:
          entity_annotation_begins.append(0)
          entity_annotation_ends.append(0)
          entity_annotation_labels.append(0)
        assert len(
            entity_annotation_begins) == entity_expected_annotations_length
        assert len(entity_annotation_ends) == entity_expected_annotations_length
        assert len(
            entity_annotation_labels) == entity_expected_annotations_length

        while len(
            answer_annotation_begins) < answer_expected_annotations_length:
          answer_annotation_begins.append(0)
          answer_annotation_ends.append(0)
          answer_annotation_labels.append(0)
        assert len(
            answer_annotation_begins) == answer_expected_annotations_length
        assert len(answer_annotation_ends) == answer_expected_annotations_length
        assert len(
            answer_annotation_labels) == answer_expected_annotations_length

    assert len(token_ids) % block_length == 0
    assert len(token_ids) % len(block_ids) == 0
    assert len(block_ids) * block_length == len(token_ids)
    assert len(token_ids) == len(is_continuation)

    feature_map = dict(
        token_ids=create_int_feature(token_ids),
        is_continuation=create_int_feature(is_continuation),
        block_ids=create_int_feature(block_ids))

    if max_num_annotations:
      feature_map.update(
          dict(
              answer_annotation_begins=create_int_feature(
                  answer_annotation_begins),
              answer_annotation_ends=create_int_feature(answer_annotation_ends),
              answer_annotation_labels=create_int_feature(
                  answer_annotation_labels),
              entity_annotation_begins=create_int_feature(
                  entity_annotation_begins),
              entity_annotation_ends=create_int_feature(entity_annotation_ends),
              entity_annotation_labels=create_int_feature(
                  entity_annotation_labels),
          ))

    if prefix_token_ids:
      feature_map.update(dict(prefix_length=create_int_feature(prefix_length)))

    return tf.train.Example(features=tf.train.Features(feature=feature_map))

  def to_document_text(self):
    """Returns the document text if available; raises ValueError otherwise."""
    if any(sentence.raw_text is None for sentence in self.sentences):
      raise ValueError(
          'Cannot get document text when `raw_text` attribute is unavailable.')
    return '\n'.join(sentence.raw_text for sentence in self.sentences)


def expand_file_patterns(file_patterns,
                         ignore_unmatched_patterns = False):
  """Expands file patterns into a list of individual file paths.

  Args:
    file_patterns: A comma-separated string containing glob file patterns.
    ignore_unmatched_patterns: If True, unmatched patterns are ignored. The
      False default causes a ValueError if any pattern fails to match any files.

  Returns:
    A list of file paths.

  Raises:
    ValueError: If any file pattern returns 0 matching files and
      `ignore_unmatched_patterns` is False.
  """
  result = []
  for file_pattern in file_patterns.split(','):
    filenames = tf.io.gfile.glob(file_pattern)
    if not filenames and not ignore_unmatched_patterns:
      raise ValueError('No matching files for pattern:', file_pattern)
    result.extend(filenames)
  return result


def read_text_file_lines(filepath):
  """Reads a single text file into lines, ensuring each line is unicode.

  Args:
    filepath: The path to a single text file.

  Yields:
    Each line as unicode text, with left and right whitespace stripped.
  """
  # We open in binary mode here to avoid unicode decoding errors. The later
  # call to `bert_tokenization.convert_to_unicode()` ignores individual errors.
  with tf.io.gfile.GFile(filepath, 'rb') as binary_mode_file:
    for line in binary_mode_file:
      yield bert_tokenization.convert_to_unicode(line).strip()


def parse_bert_pretraining_text(
    lines,
    generate_document_ids = False):
  """Parses documents from lines in the BERT pretraining file format.

  The format is described in BERT as follows:
  (1) One sentence per line. These should ideally be actual sentences, not
    entire paragraphs or arbitrary spans of text.
  (2) Blank lines between documents.

  Args:
    lines: Text lines from a BERT pretraining file as described above.
    generate_document_ids: If True, then every output `TokenizedBertDocument`
        will have a random (not necessary unique) `document_id`.

  Yields:
    A BertDocument for each each document found in `lines`.
  """

  def get_document_id():
    if generate_document_ids:
      return random.randint(1, int(2e9))
    else:
      return None

  current_doc = []
  for line in lines:
    line = line.strip()
    if not line:
      if current_doc:
        yield BertDocument(current_doc, document_id=get_document_id())
      current_doc = []
      continue
    current_doc.append(Sentence(line))
  if current_doc:
    yield BertDocument(current_doc, document_id=get_document_id())


def realign_annotations(begin, end, offsets):
  """Re-aligns annotations from Sentences to TokenizedSentence using offsets."""

  if begin > end:
    raise ValueError('Invalid interval [{}; {}]'.format(begin, end))

  def find_max_that_less_or_equal(a, x):
    r = bisect.bisect_left(a, x)
    if r == len(a):
      return len(a) - 1
    if r > len(a) or (r == 0 and a[r] > x):
      raise ValueError('Cannot find %d in the %s' % (x, str(a)))
    if a[r] > x:
      r -= 1
    return r

  return (find_max_that_less_or_equal(offsets, begin),
          find_max_that_less_or_equal(offsets, end))


def tokenize_document_for_bert(
    document,
    tokenizer):
  """Tokenizes a `BertDocument`, dropping any sentences with empty results.

  Args:
    document: A `BertDocument` to tokenize.
    tokenizer: FullTokenizer object.

  Returns:
    A `TokenizedBertDocument`.
  """
  tokenized_sentences = []
  for sentence in document.sentences:
    tokenization_result = tokenizer.tokenize_full_output(sentence.text)
    tokens = tokenization_result.tokens
    is_continuation = tokenization_result.is_continuation
    annotations = []
    for annotation in (sentence.annotations or []):
      begin, end = realign_annotations(annotation.begin, annotation.end,
                                       tokenization_result.offsets)
      annotations.append(
          Annotation(begin, end,
                     tokenization_result.get_span_surface_form(begin, end),
                     annotation.label, annotation.type))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if not token_ids:
      continue

    tokenized_sentences.append(
        TokenizedSentence(
            token_ids=token_ids,
            is_continuation=is_continuation,
            tokens=tokens,
            raw_text=sentence.text,
            annotations=annotations))

  return TokenizedBertDocument(
      tokenized_sentences, document_id=document.document_id)


def split_tokenized_sentences(
    document, max_tokens,
    min_tokens_for_graceful_split):
  """Splits a document into shorter sentences meeting max length constraints."""
  sentences = []
  for sentence in document.sentences:
    if len(sentence) < max_tokens:
      sentences.append(sentence)
    else:
      annotations_mask = np.zeros(len(sentence.token_ids))
      for annotation in sentence.annotations or []:
        annotations_mask[annotation.begin:annotation.end + 1] = 1

      start_index = 0
      while start_index < len(sentence.token_ids):
        # Inclusive
        original_end_index = min(start_index + max_tokens - 1,
                                 len(sentence.token_ids) - 1)
        end_index = original_end_index
        while (end_index > start_index and
               ((sentence.is_continuation is not None and
                 end_index < len(sentence.token_ids) - 1 and
                 sentence.is_continuation[end_index + 1] == 1) or
                annotations_mask[end_index] == 1)):
          end_index -= 1
        if end_index - start_index + 1 < min_tokens_for_graceful_split:
          end_index = original_end_index

        if sentence.is_continuation is not None:
          is_continuation = sentence.is_continuation[start_index:end_index + 1]
        else:
          is_continuation = None
        annotations = []
        for annotation in sentence.get_annotations(start_index, end_index):
          annotation = copy.copy(annotation)
          annotation.begin -= start_index
          annotation.end -= start_index
          annotations.append(annotation)
        sentences.append(
            TokenizedSentence(
                token_ids=sentence.token_ids[start_index:end_index + 1],
                is_continuation=is_continuation,
                annotations=annotations))
        start_index = end_index + 1
  return TokenizedBertDocument(
      sentences=sentences, document_id=document.document_id)


def split_tokenized_documents(
    document, max_tokens,
    max_sentences):
  """Splits a document into smaller documents meeting max length constraints.

  Splitting happens only at sentence boundaries. Individual sentences will not
  be split.

  We try to distribute tokens fairly evenly so we don't end up with one document
  containing a small remainder of tokens. For example, if a document has 101
  sentences, each with 10 tokens, and `max_tokens` is 500 while `max_sentences`
  is 50, we know we must split into 3 documents. A greedy approach would split
  into documents with [50, 50, 1] sentences, but we try to split into something
  more like [34, 34, 33] sentences instead. The current heuristic should work
  fairly well when `max_tokens` is the bottleneck, but the results may not be as
  even if `max_sentences` is a major bottleneck instead.

  Args:
    document: The tokenized document to split.
    max_tokens: Maximum number of tokens in each output document.
    max_sentences: (Optional) Maximum number of sentences in each output
      document. If None then will be equal to max_tokens.

  Returns:
    The list of documents resulting from splitting.
  """
  if max_sentences is None:
    max_sentences = max_tokens
  result = []
  remaining_tokens = sum(len(x.token_ids) for x in document.sentences)
  current_sentences = []
  current_num_tokens = 0
  target_num_tokens = None
  for i, sentence in enumerate(document.sentences):
    if not current_sentences:
      if remaining_tokens == 0:
        # This should only happen if there are 1 or more empty sentences at
        # the end of the document.
        target_num_tokens = max_tokens
      else:
        num_splits = math.ceil(remaining_tokens / max_tokens)
        target_num_tokens = int(math.ceil(remaining_tokens / num_splits))

    current_sentences.append(sentence)
    current_num_tokens += len(sentence)
    remaining_tokens -= len(sentence)

    if (i == len(document.sentences) - 1 or
        current_num_tokens >= target_num_tokens or
        current_num_tokens + len(document.sentences[i + 1]) > max_tokens or  #
        len(current_sentences) == max_sentences):
      result.append(
          TokenizedBertDocument(
              current_sentences, document_id=document.document_id))
      current_sentences = []
      current_num_tokens = 0

  assert not current_sentences
  assert remaining_tokens == 0
  return result
