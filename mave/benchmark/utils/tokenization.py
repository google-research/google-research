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

"""Utils for tokenization."""
from typing import List, Sequence, Tuple
import unicodedata

from bert import tokenization


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically control characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _index_mapping_whitespace_tokenize(
    text):
  """Basic whitespace tokenizaion with index mapping."""
  words = []
  word_by_char = []
  char_by_word = []
  prev_is_separator = True
  for char_index, char in enumerate(text):
    if _is_whitespace(char):
      prev_is_separator = True
    else:
      if prev_is_separator:
        words.append(char)
        char_by_word.append(char_index)
        prev_is_separator = False
      else:
        words[-1] += char
    word_by_char.append(len(words) - 1)
  return words, word_by_char, char_by_word


class IndexMappingWordpieceTokenizer:
  """A WordPiece tokenizer that keeps index mapping."""

  def __init__(self, vocab_file, **kwargs):
    self.tokenizer = tokenization.FullTokenizer(vocab_file, **kwargs)

  def tokenize(
      self, text, char_spans = ()
  ):
    """Returns text Wordpiece tokens and token level spans.

    Example usage:
    >>> tokenizer = IndexMappingWordpieceTokenizer(vocab_file)
    >>> tokens, spans = tokenizer.tokenize('a b c', [(0, 2), (4, 4)])
    >>> print(tokens)
    ['a', 'b', 'c']
    >>> print(spans)
    [(0, 1), (2, 2)]

    Args:
      text: The text string to tokenize.
      char_spans: The character level spans to be converted to token leven
        spans.

    Returns:
      (wordpiece tokens, token level spans).
    """
    words, word_by_char, _ = _index_mapping_whitespace_tokenize(text)
    tokens, token_by_word, _ = self._index_mapping_wordpiece_tokenize(words)

    def _improve_token_span(char_span_text,
                            token_span):
      """Improves token spans to more fine-grained."""
      char_span_tokens = tuple(self.tokenizer.tokenize(char_span_text))
      begin, end = token_span
      for new_begin in range(begin, end + 1):
        for new_end in range(end, new_begin - 1, -1):
          new_span_tokens = tuple(tokens[new_begin:new_end + 1])
          if new_span_tokens == char_span_tokens:
            return new_begin, new_end
      return token_span

    token_spans = []
    for char_begin, char_end in char_spans:
      word_begin, word_end = (word_by_char[char_begin], word_by_char[char_end])
      token_span = (token_by_word[word_begin], token_by_word[word_end + 1] - 1)
      char_span_text = text[char_begin:char_end + 1]
      token_spans.append(_improve_token_span(char_span_text, token_span))

    return tokens, token_spans

  def convert_tokens_to_ids(self, tokens):
    return self.tokenizer.convert_tokens_to_ids(tokens)

  def convert_ids_to_tokens(self, ids):
    return self.tokenizer.convert_ids_to_tokens(ids)

  def _index_mapping_wordpiece_tokenize(
      self, words
  ):
    """Wordpiece tokenizaion with index mapping."""
    tokens = []
    token_by_word = []
    word_by_token = []
    for word_index, word in enumerate(words):
      token_by_word.append(len(tokens))
      sub_tokens = self.tokenizer.tokenize(word)
      word_by_token.extend([word_index] * len(sub_tokens))
      tokens.extend(sub_tokens)
    token_by_word.append(len(tokens))  # Adds a placeholder word.
    return tokens, token_by_word, word_by_token
