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

"""Tokenizer and utils for parsing of assessment and plan sections."""

import dataclasses
import enum
import re
from typing import List


@enum.unique
class TokenType(enum.Enum):
  UNKNOWN = 0
  WORD = 1
  PUNCT = 2
  NUM = 3
  DEID = 4
  SPACE = 5


@dataclasses.dataclass
class Token:
  char_start: int
  char_end: int
  token_type: TokenType
  token_text: str


_TOKENIZER_REGEXP = re.compile("|".join([
    r"(?P<DEID>\[\*\*.*?\*\*\])", r"(?P<WORD>[A-Za-z]+)", r"(?P<NUM>\d+)",
    r"(?P<SPACE>\s)", r"(?P<PUNCT>[^\w\s])"
]))


def tokenize(text):
  """Separates text into a list of tokens via regex.

  Token types are words, punctutation, numbers, de-id tokens, spaces.
  de-id tokens take the MIMIC3 form of "[** deid attribute text **]"
  Args:
    text: str

  Returns:
    tokens: List[Token], Token objects with character indices, token type and
    text.
  Raises:
    KeyError: When a token with an invalid type was identified.
  """

  tokens = []
  for match in re.finditer(_TOKENIZER_REGEXP, text):
    kind = match.lastgroup
    match_text = match.group()
    char_start, char_end = match.start(), match.end()  # [start, end)
    try:
      tokens.append(Token(char_start, char_end, TokenType[kind], match_text))
    except KeyError as error:
      raise KeyError("type %s is not a valid token type" % kind) from error
  return tokens
