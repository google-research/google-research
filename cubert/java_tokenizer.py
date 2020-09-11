# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A Python tokenizer subclass of CuBertTokenizer."""
from typing import List
from typing import Sequence
from typing import Tuple


from absl import logging
from javalang import tokenizer


from cubert import cubert_tokenizer
from cubert import unified_tokenizer


class JavaTokenizer(cubert_tokenizer.CuBertTokenizer):
  """Tokenizer that extracts Python's lexical elements preserving strings."""
  _TOKEN_TYPE_MAP = {
      tokenizer.EndOfInput: unified_tokenizer.TokenKind.EOS,

      tokenizer.Keyword: unified_tokenizer.TokenKind.KEYWORD,
      tokenizer.Modifier: unified_tokenizer.TokenKind.KEYWORD,

      tokenizer.Separator: unified_tokenizer.TokenKind.PUNCTUATION,
      tokenizer.Operator: unified_tokenizer.TokenKind.PUNCTUATION,

      tokenizer.Annotation: unified_tokenizer.TokenKind.IDENTIFIER,
      tokenizer.Identifier: unified_tokenizer.TokenKind.IDENTIFIER,
      tokenizer.BasicType: unified_tokenizer.TokenKind.IDENTIFIER,

      tokenizer.Integer: unified_tokenizer.TokenKind.NUMBER,
      tokenizer.DecimalInteger: unified_tokenizer.TokenKind.NUMBER,
      tokenizer.OctalInteger: unified_tokenizer.TokenKind.NUMBER,
      tokenizer.BinaryInteger: unified_tokenizer.TokenKind.NUMBER,
      tokenizer.HexInteger: unified_tokenizer.TokenKind.NUMBER,
      tokenizer.FloatingPoint: unified_tokenizer.TokenKind.NUMBER,
      tokenizer.DecimalFloatingPoint: unified_tokenizer.TokenKind.NUMBER,
      tokenizer.HexFloatingPoint: unified_tokenizer.TokenKind.NUMBER,

      tokenizer.Boolean: unified_tokenizer.TokenKind.STRING,
      tokenizer.Character: unified_tokenizer.TokenKind.STRING,
      tokenizer.String: unified_tokenizer.TokenKind.STRING,
      tokenizer.Null: unified_tokenizer.TokenKind.STRING,
  }

  def tokenize_and_abstract(
      self,
      source_code):
    """As per the superclass."""
    try:
      java_tokens = tokenizer.tokenize(source_code)
    except tokenizer.LexerError as e:
      logging.warn('The tokenizer raised exception `%s` while parsing %s', e,
                   source_code)
      return (
          (cubert_tokenizer.quote_special(
              unified_tokenizer.TokenKind.ERROR.name),
           unified_tokenizer.TokenKind.ERROR),
          (cubert_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS),
           unified_tokenizer.TokenKind.EOS),
      )

    agnostic_tokens: List[Tuple[str, unified_tokenizer.TokenKind]] = []

    for token in java_tokens:
      # The token kind is the subclass type of the token.
      token_type = type(token)
      if token_type not in JavaTokenizer._TOKEN_TYPE_MAP:
        raise ValueError('Received Java token type %s, but it was unexpected, '
                         'while tokenizing \n%s\n' % (token_type, source_code))

      agnostic_tokens.append(
          (token.value, JavaTokenizer._TOKEN_TYPE_MAP[token_type]))

    return agnostic_tokens

  def untokenize_abstract(self, whole_tokens):
    tokens: List[str] = []

    for token in whole_tokens[:-1]:  # Skip EOS. The caller checked it's there.
      if token == cubert_tokenizer.quote_special(
          unified_tokenizer.TokenKind.NEWLINE.name):
        tokens.append('\n')
      else:
        tokens.append(token)
    return ''.join(tokens)
