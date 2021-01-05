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

"""A Java tokenizer subclass of CuBertTokenizer.

This tokenizer uses an extension of the tokenizer from the javalang GitHub
repository. The extension enables the javalang tokenizer to return end positions
as well as end-of-sequence tokens and comments.
"""
from typing import List
from typing import Sequence


from absl import logging
import dataclasses
from javalang import tokenizer as javalang


from cubert import cubert_tokenizer
from cubert import extended_javalang_tokenizer
from cubert import unified_tokenizer


class JavaTokenizer(cubert_tokenizer.CuBertTokenizer):
  """Tokenizer that extracts Python's lexical elements preserving strings."""
  _TOKEN_TYPE_MAP = {
      javalang.EndOfInput:
          unified_tokenizer.TokenKind.EOS,
      javalang.Keyword:
          unified_tokenizer.TokenKind.KEYWORD,
      javalang.Modifier:
          unified_tokenizer.TokenKind.KEYWORD,
      javalang.Separator:
          unified_tokenizer.TokenKind.PUNCTUATION,
      javalang.Operator:
          unified_tokenizer.TokenKind.PUNCTUATION,
      javalang.Annotation:
          unified_tokenizer.TokenKind.IDENTIFIER,
      javalang.Identifier:
          unified_tokenizer.TokenKind.IDENTIFIER,
      javalang.BasicType:
          unified_tokenizer.TokenKind.IDENTIFIER,
      javalang.Integer:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.DecimalInteger:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.OctalInteger:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.BinaryInteger:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.HexInteger:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.FloatingPoint:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.DecimalFloatingPoint:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.HexFloatingPoint:
          unified_tokenizer.TokenKind.NUMBER,
      javalang.Boolean:
          unified_tokenizer.TokenKind.STRING,
      javalang.Character:
          unified_tokenizer.TokenKind.STRING,
      javalang.String:
          unified_tokenizer.TokenKind.STRING,
      javalang.Null:
          unified_tokenizer.TokenKind.STRING,

      # New JavalangTokenizerExtended token kinds:
      extended_javalang_tokenizer.Comment:
          unified_tokenizer.TokenKind.COMMENT,
      extended_javalang_tokenizer.Whitespace:
          unified_tokenizer.TokenKind.WHITESPACE,
      extended_javalang_tokenizer.ErrorToken:
          unified_tokenizer.TokenKind.ERROR,
  }

  def tokenize_and_abstract(
      self,
      source_code):
    """As per the superclass."""
    agnostic_tokens: List[unified_tokenizer.AbstractToken] = []

    try:
      java_tokens = list(
          extended_javalang_tokenizer.tokenize_extended(source_code))
    except (javalang.LexerError, TypeError) as e:
      # Sometimes, javalang returns a TypeError when reading a number.
      # See
      # https://github.com/c2nes/javalang/blob/0664afb7f4d40254312693f2e833c1ed4ac551c7/javalang/tokenizer.py#L370
      logging.warning('The tokenizer raised exception `%r` while parsing %s', e,
                      source_code)

      # We don't try to do recovery from errors quite yet. Mark the error as
      # occurring at whatever position we are in and terminate
      agnostic_tokens.append(
          unified_tokenizer.AbstractToken(
              unified_tokenizer.quote_special(
                  unified_tokenizer.TokenKind.ERROR.name),
              unified_tokenizer.TokenKind.ERROR,
              unified_tokenizer.TokenMetadata(
                  start=unified_tokenizer.Position(
                      line=0, column=0),
                  end=unified_tokenizer.Position(
                      line=0, column=0))))
      agnostic_tokens.append(
          unified_tokenizer.AbstractToken(
              unified_tokenizer.quote_special(
                  unified_tokenizer.TokenKind.EOS.name),
              unified_tokenizer.TokenKind.EOS,
              unified_tokenizer.TokenMetadata(
                  start=unified_tokenizer.Position(
                      line=0, column=0),
                  end=unified_tokenizer.Position(
                      line=0, column=0))))
    else:
      start_line = 0
      start_column = 0
      for token in java_tokens:
        # The token kind is the subclass type of the token.
        token_type = type(token)
        if token_type not in JavaTokenizer._TOKEN_TYPE_MAP:
          raise ValueError(
              'Received Java token type %s, but it was unexpected, '
              'while tokenizing \n%s\n' % (token_type, source_code))

        # JavaTokenizer counts lines and columns from 1.
        start_line = token.position.line - 1
        start_column = token.position.column - 1

        # The tokenizer seems to take some liberties with Unicode, returning
        # invalid characters. This cleans spellings up.
        spelling = token.value.encode('utf-8', errors='replace').decode('utf-8')
        agnostic_tokens.append(
            unified_tokenizer.AbstractToken(
                spelling, JavaTokenizer._TOKEN_TYPE_MAP[token_type],
                unified_tokenizer.TokenMetadata(
                    start=unified_tokenizer.Position(
                        line=start_line, column=start_column))))

    # At this point, we have all the tokens, either as produced and abstracted,
    # or a placeholder error and eos in case of an exception. However, the
    # tokens only have start positions. Since the extended tokenizer guarantees
    # that tokens abut, we take a second pass, backwards, setting the end
    # position of a token from the start position of token following it. The
    # final token, `EOS` already has an end position, so we don't modify it.
    eos = agnostic_tokens[-1]
    if not eos.metadata.start:
      # This should be there. Raise an exception
      raise AssertionError('The end of input token is missing positioning '
                           'information: %s' % eos)
    later_token_start: unified_tokenizer.Position = eos.metadata.start

    # The EOS token has an empty extent, so the end and the start are set to be
    # the same.
    filled_agnostic_tokens = [
        dataclasses.replace(
            eos,
            metadata=dataclasses.replace(eos.metadata, end=eos.metadata.start))
    ]
    # Go backwards, from the element before `eos` to the beginning.
    for token in (
        agnostic_tokens[i] for i in range(len(agnostic_tokens) - 2, -1, -1)):
      filled_token = dataclasses.replace(
          token,
          metadata=dataclasses.replace(token.metadata, end=later_token_start))
      filled_agnostic_tokens.append(filled_token)
      later_token_start = token.metadata.start

    # Now we have the tokens, including end position, but they're reversed.
    # The final step is to break down whitespace tokens into primitive
    # WHITESPACE tokens and NEWLINE tokens.
    with_broken_whitespace = []
    for token in filled_agnostic_tokens[::-1]:
      if token.kind is not unified_tokenizer.TokenKind.WHITESPACE:
        with_broken_whitespace.append(token)
      else:
        # This is whitespace. Replace it with primitive tokens.
        with_broken_whitespace.extend(
            unified_tokenizer.fill_range_with_whitespace(
                token.metadata.start, token.metadata.end))

    return with_broken_whitespace

  def untokenize_abstract(self, whole_tokens):
    tokens: List[str] = []

    for token in whole_tokens[:-1]:  # Skip EOS. The caller checked it's there.
      if token == unified_tokenizer.quote_special(
          unified_tokenizer.TokenKind.NEWLINE.name):
        tokens.append('\n')
      else:
        tokens.append(token)
    return ''.join(tokens)
