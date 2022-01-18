# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
import dataclasses
import tokenize
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import Text
from typing import Union

from cubert import unified_tokenizer

# After all splitting, the longest a token is of the following length.
MAX_OUTPUT_TOKEN_LENGTH = 15


class CuBertTokenizer(abc.ABC):
  """A tokenizer that implements a language-agnostic tokenization.

  The tokenizer implements a language-agnostic tokenization. This is available
  as `tokenize_and_abstract()`.
  """

  def __init__(self, max_output_token_length = MAX_OUTPUT_TOKEN_LENGTH,
               reserved = ()):
    self.types_to_skip = ()
    self.reserved = reserved
    self.mappings: Dict[str, str]
    self.update_mappings({
        # By default, replace \n and \r. This is meant primarily for literals.
        '\n':
            unified_tokenizer.quote_special('NLCHAR'),
        '\r':
            unified_tokenizer.quote_special('CR'),
        unified_tokenizer.SENTINEL:
            unified_tokenizer.quote_special(unified_tokenizer.SENTINEL_ESCAPE),
    })
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
      self, types_to_skip
  ):
    """Replaces the set of token types that are ignored.

    Each tokenizer may provide different semantics with respect to this list,
    and may ignore it altogether.

    Args:
      types_to_skip: Types (from the constants in the `token` module) or
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

  def get_mappings(self):
    return self.mappings

  def condition_full_tokens(
      self, agnostic
  ):
    """Applies reserved keywords and character sanitization."""
    filtered: Iterable[unified_tokenizer.AbstractToken] = (
        a for a in agnostic if a.kind not in self.types_to_skip)

    # Now turn all reserved words, regardless of kind, into keywords.
    with_reserved: Sequence[unified_tokenizer.AbstractToken] = tuple(
        dataclasses.replace(a, kind=unified_tokenizer.TokenKind.KEYWORD)
        if a.spelling in self.reserved else a
        for a in filtered)
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

    multi_tokens = self.subtokenize_full_tokens(conditioned)

    subtokens = unified_tokenizer.flatten_subtoken_lists(multi_tokens)
    return subtokens

  def untokenize_agnostic(self, token_list):
    """Turns CuBERT subtokens into whole tokens."""
    # Untokenize agnostic.
    if (not token_list or token_list[-1] != unified_tokenizer.quote_special(
        unified_tokenizer.TokenKind.EOS.name)):
      raise ValueError('Token list %r should end with the EOS token %r.' %
                       (token_list,
                        unified_tokenizer.quote_special(
                            unified_tokenizer.TokenKind.EOS.name)))

    whole_tokens = unified_tokenizer.reconstitute_full_unsanitary_tokens(
        token_list,
        sanitization_mapping=self.mappings,
        sentinel=unified_tokenizer.SENTINEL)
    return whole_tokens

  def untokenize(self, token_list):
    """Untokenizes via `untokenize_abstract`."""
    whole_tokens = self.untokenize_agnostic(token_list)
    return self.untokenize_abstract(whole_tokens)


def token_from_token_type(token_type):
  """Turns a token type into a reserved token string."""
  # We use the tok_name dict from tokenize, not token. The former has
  # NL and COMMENT and such, whereas the latter doesn't.
  return unified_tokenizer.quote_special(tokenize.tok_name[token_type])
