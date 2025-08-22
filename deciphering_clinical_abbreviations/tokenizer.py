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

"""Helper functions for the tokenization of abbreviated and expanded text."""

import collections
from collections.abc import Mapping, Sequence
import enum
import re
from typing import Optional

# Type aliases.
AbbrevExpansionDict = Mapping[str, Sequence[str]]

# Tokenization constants and enums.
START_STRINGS = {"-", "#", "(", "[", "'", "–", '"', "=", "non-"}
END_STRINGS = {
    ",", ".", ":", ")", "]", "/", ";", "-", "?",
    "'", "–", '"', "=", "-like", "'s"}
ONLY_START_STRINGS = START_STRINGS - END_STRINGS
ONLY_END_STRINGS = END_STRINGS - START_STRINGS


class TokenizationStringType(enum.Enum):
  """things."""
  ORIGINAL = enum.auto()
  EXPANDED = enum.auto()


def create_word_finder_regex(words):
  """Creates regex pattern that matches any of the provided words.

  The regex pattern is comprised of 3 components: the prefix, the words, and the
  suffix. The words are a concatenation of all provided words separated by the
  OR operator. The prefix and suffix ensure that the matched word represents a
  distinct, delineated instance of that word.
    - prefix: A word must be preceded by any one of the following patterns:
      1. The string start
      2. Any character OTHER than alphanumeric or apostrophe
      3. An apostrophe preceded by a non-alphanumeric character (this
        ensures contractions and possessives do not lead to false positives)
    - suffix: A word must be followed by any one of the following patterns:
      1. The string end
      2. Any non-alphanumeric character
  Note: the order of the words matters, as the regex pattern will greedily
  match with the first word in the list. This has implications for instances
  in which a word is a distinguishable substring of another word.

  Args:
    words: A list of words that the pattern will find.
  Returns:
    A regex pattern that will match any instance of the provided words.
  """
  words_re = "|".join([re.escape(word) for word in words])
  prefix_re = "^|[^\\w']|(?:^|\\W)'"
  suffix_re = "(?:$|\\W)"
  # We exclude the suffix from capture so that consecutive abbreviations
  # separated by spaces can be matched.
  re_string = f"(?:{prefix_re})({words_re})(?={suffix_re})"
  return re.compile(re_string)


class Tokenizer:
  """A class for tokenizing text with abbreviations and expansions.

  This class can be provided with an abbreviation expansions dictionary at
  initialization, which will be used to enforce that certain strings remain as
  atomic tokens after tokenization, which aids alignment.

  Note: Tokenizing and detokenizing is not lossless. If tokens are bordered by
  strings that are classified as both start and end strings, they will be broken
  off during tokenization but then left separate during detokenization, since it
  is ambiguous which adjacent token to attach them to. E.g.
  detokenize(tokenize("hello- you"))="hello - you". If a string that is
  classified exclusively as a start or end string is provided separately, it
  will still be prepended/appended to the adjacent token during detokenization.
  E.g. detokenize(tokenize("hello : you"))="hello: you".
  """

  def __init__(
      self,
      abbreviation_expansions_dict = None):
    """Initializes the Tokenizer.

    Args:
      abbreviation_expansions_dict: An optional dictionary mapping known
        abbreviations to their valid expansions. This is used to ensure that
        abbreviations are never broken up into separate tokens, and that
        expansions containing their own abbreviations are also kept as single
        tokens.
    """
    self._abbreviation_expansions_dict = abbreviation_expansions_dict
    if abbreviation_expansions_dict is not None:
      self._atomic_tokens_provided = True
      abbreviations = sorted(
          abbreviation_expansions_dict.keys(), key=len, reverse=True)
      expansions_containing_abbrev = sorted(
          self._get_expansions_containing_abbreviation(), key=len, reverse=True)
      self._orig_regex = create_word_finder_regex(abbreviations)
      self._expanded_regex = create_word_finder_regex(
          expansions_containing_abbrev + abbreviations)
    else:
      self._atomic_tokens_provided = False

  def _get_expansions_containing_abbreviation(self):
    """Gets expansions that contain their abbreviations within them."""
    if self._abbreviation_expansions_dict is None:
      raise ValueError("No abbreviation expansions dictionary provided.")
    expansions_containing_abbrev = []
    for abbrev, expansions in self._abbreviation_expansions_dict.items():
      abbrev_re = create_word_finder_regex([abbrev])
      for expansion in expansions:
        if abbrev_re.findall(expansion):
          expansions_containing_abbrev.append(expansion)
    return expansions_containing_abbrev

  def _standard_tokenize(self, string):
    """Separates tokens at spaces and breaks off start and end strings."""
    # Split into coarse tokens at any number of spaces.
    coarse_tokens = string.split()
    fine_tokens = collections.deque()
    while coarse_tokens:
      token = coarse_tokens.pop()
      # Break off suffixes.
      for end_string in END_STRINGS:
        if len(token) >= len(end_string) and token.endswith(end_string):
          fine_tokens.appendleft(end_string)
          token = token[:-len(end_string)]
      # Break off prefixes.
      prefixes = []
      for start_string in START_STRINGS:
        if len(token) > len(start_string) and token.startswith(start_string):
          prefixes.append(start_string)
          token = token[len(start_string):]
      if token:
        fine_tokens.appendleft(token)
      while prefixes:
        fine_tokens.appendleft(prefixes.pop())
    return list(fine_tokens)

  def tokenize(
      self, string, string_type = None
      ):
    """Decomposes string into separate tokens.

    The rules of tokenization:
      - If class was initialized with a nonempty list of abbreviations, all
        provided abbreviations are kept as atomic tokens.
      - The rest of the string is separated into 'coarse' tokens by splitting at
        any number of space chars.
      - For each coarse token, any suffixes/prefixes recorded as end/start
        strings are broken off to obtain fine tokens.

    Args:
      string: The string to be tokenized.
      string_type: {None, TokenizationStringType.ORIGINAL} The type of string.
        If provided, indicates which regex pattern to use to keep matched
        phrases as atomic tokens.
    Returns:
      A list of string tokens.
    """
    word_re = None
    # If there were no tokens provided to keep atomic, we skip directly to
    # standard tokenization.
    if not self._atomic_tokens_provided or string_type is None:
      return self._standard_tokenize(string)
    if string_type == TokenizationStringType.ORIGINAL:
      word_re = self._orig_regex
    elif string_type == TokenizationStringType.EXPANDED:
      word_re = self._expanded_regex
    else:
      raise ValueError(f"'{string_type}' is not a valid string_type.")
    abbrev_matches = word_re.finditer(string)
    tokens = []
    # Recursively tokenize text around matches.
    # Reverse so that string modification doesn't shift indices.
    for match in reversed(list(abbrev_matches)):
      matched_abbrev = match.group(1)
      tokens_after = self._standard_tokenize(string[match.end(1):].strip())
      tokens = [matched_abbrev] + tokens_after + tokens
      string = string[:match.start(1)].strip()
    return self._standard_tokenize(string) + tokens

  def detokenize(self, tokens):
    """Combines separate tokens back into a string.

    The rules of detokenization:
      - If any token is classified as exclusively a start or end token, it will
        be prepended or appended, respectively, to its adjacent token.

    Args:
      tokens: The tokens to combine.
    Returns:
      The combined string.
    """
    tokens = list(tokens)
    processed_tokens = collections.deque()
    while tokens:
      token = tokens.pop()
      if tokens and token in ONLY_END_STRINGS:
        processed_tokens.appendleft(tokens.pop() + token)
      elif processed_tokens and token in ONLY_START_STRINGS:
        processed_tokens[0] = token + processed_tokens[0]
      else:
        processed_tokens.appendleft(token)
    return " ".join(processed_tokens).strip()
