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

"""Tokenization classes."""
from typing import List, Optional, Sequence, Text

import dataclasses
from bert import tokenization as bert_tokenization
import numpy as np
import tensorflow.compat.v1 as tf

import sentencepiece.sentencepiece_pb2 as sentencepiece_pb2
import sentencepiece as spm

# Special symbol '▁'
SPIECE_UNDERLINE = u'\u2581'


@dataclasses.dataclass(frozen=True)
class TokenizationResult(object):
  """Full results of SentencePiece tokenization."""
  tokens: List[Text]
  token_ids: List[int]
  byte_offsets: Optional[List[int]]
  offsets: List[int]
  surface_forms: List[Text]
  is_continuation: List[int]

  def get_span_surface_form(self, begin, end):
    return ''.join(self.surface_forms[begin:end + 1])


def get_byte_to_char_offset_mapping(text):
  """Get mapping from bytes positions to unicode code point positions."""
  if not text:
    return {}, {}
  bytes_per_char = [len(c.encode('utf-8')) for c in text]
  bytes_per_char_cumsum = np.cumsum(bytes_per_char)
  char_begin_offset = np.roll(bytes_per_char_cumsum, 1)
  char_begin_offset[0] = 0
  byte_to_char_begin_offset_mapping = {
      offset: char_idx for char_idx, offset in enumerate(char_begin_offset)
  }
  char_end_offset = bytes_per_char_cumsum - 1
  byte_to_char_end_offset_mapping = {
      offset: char_idx for char_idx, offset in enumerate(char_end_offset)
  }
  return byte_to_char_begin_offset_mapping, byte_to_char_end_offset_mapping


class FullTokenizer(object):
  """Runs end-to-end tokenziation.

  The class provides functions to tokenize text and, additionally, return
  the offsets to the original, untokenized text. This information can be useful
  if one decided to re-align span annotations from the original text against
  the tokenized text.

  Currently only supports SentencePiece tokenization which is similar to ALBERT,
  but the latter performs some additional postprocessing.
  We might want to extend it to the BERT tokenization as well.
  """

  # TODO(urikz): Support `force_tokenization_to_words`
  def __init__(self,
               spm_model_file = None,
               vocab_path = None,
               do_lower_case = False):
    if spm_model_file is not None and vocab_path is not None:
      raise ValueError('You cannot specify both `spm_model_file` and '
                       '`vocab_path` at the same time.')
    if spm_model_file is None and vocab_path is None:
      raise ValueError('You need to specify exactly one from `spm_model_file` '
                       'and `vocab_path`.')
    if spm_model_file is not None:
      self.tokenizer = None
      self.sp_model = spm.SentencePieceProcessor()
      tf.logging.info('Loading sentence piece model from %s', spm_model_file)
      # Handle cases where SP can't load the file, but gfile can.
      sp_model_ = tf.gfile.GFile(spm_model_file, 'rb').read()
      self.sp_model.LoadFromSerializedProto(sp_model_)
      # Note(mingdachen): For the purpose of consisent API, we are
      # generating a vocabulary for the sentence piece tokenizer.
      self.vocab = {
          self.sp_model.IdToPiece(i): i
          for i in range(self.sp_model.GetPieceSize())
      }
      self.inv_vocab = {v: k for k, v in self.vocab.items()}
    else:
      self.sp_model = None
      self.tokenizer = bert_tokenization.FullTokenizer(
          vocab_file=vocab_path, do_lower_case=do_lower_case)

  def tokenize_full_output(self, text):
    if self.sp_model is not None:
      return self.tokenize_full_output_sp(text)
    else:
      return self.tokenize_full_output_bert(text)

  def tokenize_full_output_sp(self, text):
    """Tokenize text. See `TokenizationResult` for details."""
    # TODO(urikz): Shall we sample word pieces? That might make the model
    # more robust to all tokenization discrepancies between pre-training /
    # fine-tuning stages.
    processed_text = sentencepiece_pb2.SentencePieceText.FromString(
        self.sp_model.EncodeAsSerializedProto(text))

    # Note that SentencePiece returns offsets in bytes.
    # So we have to manually convert to offset in unicode code points.
    byte_to_char_offset_mapping, _ = get_byte_to_char_offset_mapping(text)

    tokens, token_ids = [], []
    offsets, byte_offsets = [], []
    surface_forms, is_continuation = [], []
    for piece in processed_text.pieces:
      tokens.append(piece.piece)
      token_ids.append(piece.id)
      byte_offsets.append(piece.begin)
      if piece.begin not in byte_to_char_offset_mapping:
        raise ValueError(
            'SentencePiece tokenized text in-between unicode character:' + text)
      offsets.append(byte_to_char_offset_mapping[piece.begin])
      surface_forms.append(piece.surface)
      is_continuation.append(int(not piece.piece.startswith(SPIECE_UNDERLINE)))
    return TokenizationResult(tokens, token_ids, byte_offsets, offsets,
                              surface_forms, is_continuation)

  def tokenize_full_output_bert(self, text):
    """Tokenize text. See `TokenizationResult` for details."""

    def is_whitespace(c):
      if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
        return True
      return False

    words = []
    word_to_char_offset = []
    prev_is_whitespace = True
    for char_index, c in enumerate(text):
      if is_whitespace(c):
        prev_is_whitespace = True
      else:
        if prev_is_whitespace:
          words.append(c)
          word_to_char_offset.append(char_index)
        else:
          words[-1] += c
        prev_is_whitespace = False

    tokens = []
    token_ids = []
    offsets = []
    surface_forms, is_continuation = [], []

    for word_index, word in enumerate(words):
      word_tokens = self.tokenizer.tokenize(word)
      current_word_offset = 0
      for token_index, token in enumerate(word_tokens):
        surface_forms_prefix = ' ' if token_index == 0 else ''
        tokens.append(token)
        offsets.append(current_word_offset + word_to_char_offset[word_index])
        if token.startswith('##'):
          is_continuation.append(1)
          current_word_offset += len(token) - 2
          surface_forms.append(surface_forms_prefix + token[2:])
        else:
          is_continuation.append(0)
          current_word_offset += len(token)
          surface_forms.append(surface_forms_prefix + token)
      token_ids.extend(self.tokenizer.convert_tokens_to_ids(word_tokens))
    assert len(tokens) == len(token_ids)
    return TokenizationResult(tokens, token_ids, None, offsets, surface_forms,
                              is_continuation)

  def tokenize(self, text):
    return self.tokenize_full_output(text).tokens

  def convert_tokens_to_ids(self, tokens):
    if self.sp_model is not None:
      return [self.sp_model.PieceToId(token) for token in tokens]
    else:
      return self.tokenizer.convert_tokens_to_ids(tokens)

  def convert_ids_to_tokens(self, ids):
    if self.sp_model is not None:
      return [self.sp_model.IdToPiece(id_) for id_ in ids]
    else:
      return self.tokenizer.convert_ids_to_tokens(ids)
