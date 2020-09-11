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
import keyword
import re
import tokenize
import typing
from typing import Any
from typing import List
from typing import Sequence
from typing import Text
from typing import Tuple
from absl import logging
import six
from cubert import cubert_tokenizer
from cubert import unified_tokenizer


class PythonTokenizer(cubert_tokenizer.CuBertTokenizer):
  """Tokenizer that extracts Python's lexical elements preserving strings."""
  _TOKEN_TYPE_MAP = {
      tokenize.COMMENT: unified_tokenizer.TokenKind.COMMENT,
      tokenize.DEDENT: unified_tokenizer.TokenKind.KEYWORD,
      tokenize.ENDMARKER: unified_tokenizer.TokenKind.EOS,
      tokenize.ERRORTOKEN: unified_tokenizer.TokenKind.ERROR,
      tokenize.INDENT: unified_tokenizer.TokenKind.KEYWORD,
      tokenize.NEWLINE: unified_tokenizer.TokenKind.NEWLINE,
      tokenize.NL: unified_tokenizer.TokenKind.PUNCTUATION,
      tokenize.NUMBER: unified_tokenizer.TokenKind.NUMBER,
      tokenize.OP: unified_tokenizer.TokenKind.PUNCTUATION,
      tokenize.STRING: unified_tokenizer.TokenKind.STRING,
  }
  _REVERSE_TOKEN_MAP = {
      cubert_tokenizer.token_from_token_type(tokenize.INDENT):
          tokenize.INDENT,
      cubert_tokenizer.token_from_token_type(tokenize.DEDENT):
          tokenize.DEDENT,
      cubert_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS.name):
          tokenize.ENDMARKER,
      cubert_tokenizer.quote_special(unified_tokenizer.TokenKind.ERROR.name):
          tokenize.ERRORTOKEN,
      cubert_tokenizer.quote_special(unified_tokenizer.TokenKind.NEWLINE.name):
          tokenize.NEWLINE,
      cubert_tokenizer.token_from_token_type(tokenize.NL):
          tokenize.NL,
  }
  # Adding the end-of-string anchor \Z below, since re.fullmatch wasn't
  # available in Python2.
  _NUMBERS = re.compile('(' + tokenize.Number + r')\Z')  # pytype: disable=module-attr
  _SINGLE_STRINGS = re.compile('(' + tokenize.String + r')\Z')  # pytype: disable=module-attr
  _TRIPLE_STRING_BEGINNINGS = re.compile(tokenize.Triple)  # pytype: disable=module-attr
  _COMMENTS = re.compile('(' + tokenize.Comment + r')\Z')  # pytype: disable=module-attr

  _EXACT_TOKEN_TYPES = tokenize.EXACT_TOKEN_TYPES.keys()  # pytype: disable=module-attr

  # Token types that CubertTokenizer will tokenize by their type and not
  # content.
  _TOKEN_TYPES_TO_TOKENIZE_BY_TYPE = [
      tokenize.NEWLINE, tokenize.DEDENT, tokenize.NL
  ]

  def __init__(self, *args, **kwargs):
    super(PythonTokenizer, self).__init__(*args, **kwargs)

    # By default, we drop COMMENT tokens.
    self.update_types_to_skip([unified_tokenizer.TokenKind.COMMENT])

  def tokenize_and_abstract(
      self,
      source_code):
    """Produces a language-agnostic tokenization of the input code."""
    token_pairs = []  # type: List[Tuple[Text, int]]
    try:
      token_tuples = unified_tokenizer.code_to_tokens(source_code)
      token_pairs = [(six.ensure_text(token_name), token_type)
                     for token_type, token_name, _, _, _ in token_tuples]
    except (tokenize.TokenError, IndentationError) as e:
      logging.warning('The tokenizer raised exception `%s` while parsing %s', e,
                      source_code)
      token_pairs = [
          (cubert_tokenizer.quote_special(
              unified_tokenizer.TokenKind.ERROR.name), tokenize.ERRORTOKEN),
          ('', tokenize.ENDMARKER),
      ]
    agnostic_tokens = []  # type: List[Tuple[Text, unified_tokenizer.TokenKind]]

    for spelling, kind in token_pairs:
      adjusted_spelling = spelling
      token_kind = unified_tokenizer.TokenKind.NONE
      if kind == tokenize.NAME:
        # Disambiguate identifiers from keywords.
        if keyword.iskeyword(spelling):
          token_kind = unified_tokenizer.TokenKind.KEYWORD
        else:
          token_kind = unified_tokenizer.TokenKind.IDENTIFIER
      else:
        if kind in PythonTokenizer._TOKEN_TYPES_TO_TOKENIZE_BY_TYPE:
          # Replace spelling with type.
          adjusted_spelling = cubert_tokenizer.token_from_token_type(kind)
        elif kind is tokenize.INDENT:
          # For INDENT, in particular, we also record the actual spelling too.
          adjusted_spelling = '{indent}{spelling}'.format(
              indent=cubert_tokenizer.token_from_token_type(kind),
              spelling=spelling)
        elif kind == tokenize.ENDMARKER:
          adjusted_spelling = cubert_tokenizer.quote_special(
              unified_tokenizer.TokenKind.EOS.name)

        # Map everything according to table.
        try:
          token_kind = PythonTokenizer._TOKEN_TYPE_MAP[kind]
        except KeyError as ke:
          # It's possible we're here because of async/await. Those kept being
          # turned into keywords and then removed from keywords, so we can't
          # rely on knowing which they are. We'll check by spelling.
          # See: https://bugs.python.org/issue30406
          # and https://bugs.python.org/issue33260
          # and https://bugs.python.org/issue35975
          if spelling in ('async', 'await'):
            token_kind = unified_tokenizer.TokenKind.KEYWORD
          else:
            raise ValueError('While trying to turn Python token %r into an '
                             'agnostic one, raised %r.' %
                             ((spelling, kind), ke))

      agnostic_tokens.append((adjusted_spelling, token_kind))

    return agnostic_tokens

  def untokenize_abstract(self, whole_tokens):
    # Reconstruct Python tokenizer tuples, so that Python's untokenize can be
    # invoked.
    token_tuples = []  # type: List[Tuple[int, Text]]

    for whole_token in whole_tokens:
      if whole_token in PythonTokenizer._EXACT_TOKEN_TYPES:
        token_tuples.append((tokenize.OP, whole_token))
      elif cubert_tokenizer.token_from_token_type(
          tokenize.INDENT) in whole_token:
        # We baked the type and spelling into one token. Break them up.
        spelling = whole_token.replace(
            cubert_tokenizer.token_from_token_type(tokenize.INDENT), '')
        token_tuples.append((tokenize.INDENT, spelling))
      elif whole_token in PythonTokenizer._REVERSE_TOKEN_MAP:
        python_kind = PythonTokenizer._REVERSE_TOKEN_MAP[whole_token]
        if python_kind in (tokenize.DEDENT, tokenize.ENDMARKER,
                           tokenize.ERRORTOKEN):
          spelling = ''
        else:  # python_kind in (tokenize.NEWLINE, tokenize.NL)
          spelling = '\n'
        token_tuples.append((python_kind, spelling))
      elif keyword.iskeyword(whole_token):
        token_tuples.append((tokenize.NAME, whole_token))
      elif PythonTokenizer._NUMBERS.match(whole_token):
        token_tuples.append((tokenize.NUMBER, whole_token))
      elif PythonTokenizer._SINGLE_STRINGS.match(whole_token):
        token_tuples.append((tokenize.STRING, whole_token))
      elif PythonTokenizer._TRIPLE_STRING_BEGINNINGS.match(whole_token):
        token_tuples.append((tokenize.STRING, whole_token))
      elif PythonTokenizer._COMMENTS.match(whole_token):
        token_tuples.append((tokenize.COMMENT, whole_token))
      else:
        # Everything else we map back to NAME.
        token_tuples.append((tokenize.NAME, whole_token))

    reconstructed = tokenize.untokenize(typing.cast(Any, token_tuples))
    return reconstructed
