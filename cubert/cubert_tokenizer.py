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

"""This module contains utilities for source code tokenization."""

import abc
import keyword
import re
import tokenize
import typing
from typing import Any, Dict, List, Sequence, Text, Tuple, Union

from absl import logging
import six
from cubert import unified_tokenizer

# Quote string for special tokens. Must make the resulting string a valid Python
# token.
SPECIAL_QUOTE = '___'


def quote_special(content):
  return '{q}{t}{q}'.format(q=SPECIAL_QUOTE, t=content)

ENDMARKER = 'ENDMARKER'

NEWLINE = quote_special('NEWLINE')

# After all splitting, the longest a token is of the following length.
MAX_OUTPUT_TOKEN_LENGTH = 15


@six.add_metaclass(abc.ABCMeta)
class Tokenizer(object):
  """A tokenizer that implements a language-agnostic tokenization.

  The tokenizer implements a language-agnostic tokenization. This is available
  as `tokenize_and_abstract()`.
  """

  def __init__(self, max_output_token_length = MAX_OUTPUT_TOKEN_LENGTH,
               reserved = ()):
    self.types_to_skip = []
    self.reserved = reserved
    self.mappings = dict()
    self.max_output_token_length = max_output_token_length

  @abc.abstractmethod
  def tokenize_and_abstract(
      self,
      source_code):
    """Produces a language-agnostic tokenization of the input code.

    Args:
      source_code: Source code stored in a string.

    Returns:
      A list of pairs of a token (string) and a token kind in the given source
        code. It always includes an end of sequence token. That is, an empty
        input always returns a list of size 1.

    Raises:
      ValueError: if `source_code` cannot be tokenized.
    """

  @abc.abstractmethod
  def untokenize_abstract(self, whole_tokens):
    """Applies language-specific rules to an abstract untokenized list.

    Args:
      whole_tokens: Abstract tokens, reconstituted and unsanitized by
        `untokenize` before passed to this language-specific logic.

    Returns:
      A string representing the untokenized text.
    """

  def update_types_to_skip(
      self, types_to_skip):
    """Replaces the set of token types that are ignored.

    Each tokenizer may provide different semantics with respect to this list,
    and may ignore it altogether.

    Args:
      types_to_skip: List of types (from the constants in the `token` module) or
        `unified_tokenizer.TokenKind`. Note that some of those constants are
        actually defined in the `tokenize` module.
    """
    self.types_to_skip = types_to_skip

  def replace_reserved_keywords(self, reserved):
    """Replaces the reserved keywords with the supplied list of strings.

    Each tokenizer may provide different semantics with respect to the list
    of reserved keywords, or ignore them altogether.

    Args:
      reserved: List of strings.
    """
    self.reserved = reserved  # Replace the old one entirely.

  def update_mappings(self, mappings):
    """Replaces the character mappings with the supplied dictionary.

    The intent for character mappings is to enable tokenizers that support them
    to sanitize dangerous characters, such as newline and carriage return,
    with a nicer symbol.

    Each tokenizer may provide different semantics with respect to the
    mappings, or ignore them altogether.

    Args:
      mappings: Dictionary of original to sanitized strings. Keys are expected
        to have length 1.

    Raises:
      ValueError: if a key has length different from 1.
    """
    unified_tokenizer.check_mappings(mappings)
    self.mappings = mappings

  def condition_full_tokens(
      self, agnostic
  ):
    """Applies reserved keywords and character sanitization."""
    filtered = [(spelling, kind) for spelling, kind in agnostic
                if kind not in self.types_to_skip]

    # Now turn all reserved words, regardless of kind, into keywords.
    with_reserved = [(spelling, unified_tokenizer.TokenKind.KEYWORD
                      if spelling in self.reserved else kind)
                     for spelling, kind in filtered]
    return with_reserved

  def subtokenize_full_tokens(
      self, agnostic
  ):
    """Performs heuristic splitting of full tokens."""
    subtoken_lists = unified_tokenizer.subtokenize_agnostic_tokens_in_place(
        agnostic_tokens=agnostic,
        max_output_token_length=self.max_output_token_length,
        sanitization_mapping=self.mappings,
        sentinel=unified_tokenizer.SENTINEL)
    return subtoken_lists

  def tokenize(self, source_code):
    """Tokenizes via `tokenize_and_abstract`."""
    try:
      agnostic = self.tokenize_and_abstract(source_code)
    except Exception as e:
      raise ValueError('While trying to do language-specific tokenization for '
                       'the string:\n\n\n%r\n\n\n%s\n\n\n'
                       'we received error %r.' % (source_code, source_code, e))

    conditioned = self.condition_full_tokens(agnostic)

    subtoken_lists = self.subtokenize_full_tokens(conditioned)

    subtokens = unified_tokenizer.flatten_subtoken_lists(subtoken_lists)
    return subtokens

  def untokenize(self, token_list):
    """Untokenizes via `untokenize_abstract`."""
    # Untokenize agnostic.
    if (not token_list or token_list[-1] != quote_special(
        unified_tokenizer.TokenKind.EOS.name)):
      raise ValueError(
          'Token list %r should end with the EOS token %r.' %
          (token_list, quote_special(unified_tokenizer.TokenKind.EOS.name)))

    whole_tokens = unified_tokenizer.reconstitute_full_unsanitary_tokens(
        token_list,
        sanitization_mapping=self.mappings,
        sentinel=unified_tokenizer.SENTINEL)

    return self.untokenize_abstract(whole_tokens)


def _token_from_token_type(token_type):
  """Turns a token type into a reserved token string."""
  # We use the tok_name dict from tokenize, not token. The former has
  # NL and COMMENT and such, whereas the latter doesn't.
  return quote_special(tokenize.tok_name[token_type])


class CuBertTokenizer(Tokenizer):
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
      _token_from_token_type(tokenize.INDENT): tokenize.INDENT,
      _token_from_token_type(tokenize.DEDENT): tokenize.DEDENT,
      quote_special(
          unified_tokenizer.TokenKind.EOS.name): tokenize.ENDMARKER,
      quote_special(
          unified_tokenizer.TokenKind.ERROR.name): tokenize.ERRORTOKEN,
      quote_special(
          unified_tokenizer.TokenKind.NEWLINE.name): tokenize.NEWLINE,
      _token_from_token_type(tokenize.NL): tokenize.NL,
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
    super(CuBertTokenizer, self).__init__(*args, **kwargs)

    # By default, we drop COMMENT tokens.
    self.update_types_to_skip([unified_tokenizer.TokenKind.COMMENT])
    self.update_mappings({
        # By default, replace \n and \r. We choose special names that are
        # different from the Python token types (i.e., NL).
        '\n':
            quote_special('NLCHAR'),
        '\r':
            quote_special('CR'),
        unified_tokenizer.SENTINEL:
            quote_special(unified_tokenizer.SENTINEL_ESCAPE),
    })

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
      logging.warning('The tokenizer raised exception `%s` while parsing %s',
                      e, source_code)
      token_pairs = [
          (quote_special(unified_tokenizer.TokenKind.ERROR.name),
           tokenize.ERRORTOKEN),
          ('',
           tokenize.ENDMARKER),
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
        if kind in CuBertTokenizer._TOKEN_TYPES_TO_TOKENIZE_BY_TYPE:
          # Replace spelling with type.
          adjusted_spelling = _token_from_token_type(kind)
        elif kind is tokenize.INDENT:
          # For INDENT, in particular, we also record the actual spelling too.
          adjusted_spelling = '{indent}{spelling}'.format(
              indent=_token_from_token_type(kind),
              spelling=spelling)
        elif kind == tokenize.ENDMARKER:
          adjusted_spelling = quote_special(
              unified_tokenizer.TokenKind.EOS.name)

        # Map everything according to table.
        try:
          token_kind = CuBertTokenizer._TOKEN_TYPE_MAP[kind]
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
                             'agnostic one, raised %r.' % ((spelling, kind),
                                                           ke))

      agnostic_tokens.append((adjusted_spelling, token_kind))

    return agnostic_tokens

  def untokenize_abstract(self, whole_tokens):
    # Reconstruct Python tokenizer tuples, so that Python's untokenize can be
    # invoked.
    token_tuples = []  # type: List[Tuple[int, Text]]

    for whole_token in whole_tokens:
      if whole_token in CuBertTokenizer._EXACT_TOKEN_TYPES:
        token_tuples.append((tokenize.OP, whole_token))
      elif _token_from_token_type(tokenize.INDENT) in whole_token:
        # We baked the type and spelling into one token. Break them up.
        spelling = whole_token.replace(
            _token_from_token_type(tokenize.INDENT), '')
        token_tuples.append((tokenize.INDENT, spelling))
      elif whole_token in CuBertTokenizer._REVERSE_TOKEN_MAP:
        python_kind = CuBertTokenizer._REVERSE_TOKEN_MAP[whole_token]
        if python_kind in (tokenize.DEDENT, tokenize.ENDMARKER,
                           tokenize.ERRORTOKEN):
          spelling = ''
        else:  # python_kind in (tokenize.NEWLINE, tokenize.NL)
          spelling = '\n'
        token_tuples.append((python_kind, spelling))
      elif keyword.iskeyword(whole_token):
        token_tuples.append((tokenize.NAME, whole_token))
      elif CuBertTokenizer._NUMBERS.match(whole_token):
        token_tuples.append((tokenize.NUMBER, whole_token))
      elif CuBertTokenizer._SINGLE_STRINGS.match(whole_token):
        token_tuples.append((tokenize.STRING, whole_token))
      elif CuBertTokenizer._TRIPLE_STRING_BEGINNINGS.match(whole_token):
        token_tuples.append((tokenize.STRING, whole_token))
      elif CuBertTokenizer._COMMENTS.match(whole_token):
        token_tuples.append((tokenize.COMMENT, whole_token))
      else:
        # Everything else we map back to NAME.
        token_tuples.append((tokenize.NAME, whole_token))

    reconstructed = tokenize.untokenize(typing.cast(Any, token_tuples))
    return reconstructed
