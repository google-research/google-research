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

"""Utilitis for AP Parsing."""

import bisect
import copy
from typing import Iterable, List, Optional, Tuple

from absl import logging

from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import tokenizer_lib

_TokenStartEndTuple = Tuple[int, int]
_CharStartEndTuple = Tuple[int, int]


def labeled_token_span_to_labeled_char_span(
    labeled_token_span,
    tokens):
  """Converts labeled spans from token to character level.

  Args:
    labeled_token_span: A token level span.
    tokens: Document tokens.

  Returns:
    LabeledCharSpan: Character level labeled span.
  """
  start_char, end_char = token_span_to_char_span(
      tokens, (labeled_token_span.start_token, labeled_token_span.end_token))
  labeled_char_span = ap_parsing_lib.LabeledCharSpan(
      span_type=labeled_token_span.span_type,
      action_item_type=labeled_token_span.action_item_type,
      start_char=start_char,
      end_char=end_char)
  return labeled_char_span


def labeled_char_span_to_labeled_token_span(
    labeled_char_span,
    tokens):
  """Converts labeled spans from character to token level.

  Args:
    labeled_char_span: A character level span.
    tokens: Document tokens.

  Returns:
    LabeledTokenSpan: Character level labeled span.
  """
  start_token, end_token = char_span_to_token_span(
      tokens, (labeled_char_span.start_char, labeled_char_span.end_char))

  labeled_token_span = ap_parsing_lib.LabeledTokenSpan(
      start_token=start_token,
      end_token=end_token,
      span_type=labeled_char_span.span_type)
  if labeled_char_span.action_item_type:
    labeled_token_span.action_item_type = labeled_char_span.action_item_type
  return labeled_token_span


def char_span_to_token_span(
    tokens,
    char_span):
  token_start_chars = [token.char_start for token in tokens]
  token_end_chars = [token.char_start for token in tokens]
  return (
      # Rightmost token starting before char_span.
      bisect.bisect_right(token_start_chars, char_span[0]) - 1,
      # Leftmost token ending after char_span.
      bisect.bisect_right(token_end_chars, char_span[1] - 1))


def token_span_to_char_span(
    tokens,
    token_span):
  return (tokens[token_span[0]].char_start, tokens[token_span[1] - 1].char_end)


def normalize_token_span(
    token_span,
    tokens):
  """Truncates span to remove flanking space (and other) tokens."""

  def keep_token(i):
    return not (tokens[i].token_type == tokenizer_lib.TokenType.SPACE or
                tokens[i].token_text in ".,:-#*")

  start_index = max(range(token_span[0], token_span[1]), key=keep_token)
  end_index = max(
      range(token_span[1], start_index - 1, -1),
      key=lambda i: keep_token(i - 1))
  return (start_index, end_index)


def normalize_labeled_char_span(
    label, tokens
):
  """Performs token based normalization for a labeled character span.

  Args:
    label: LabeledCharSpan pre-normalization.
    tokens: Tokens of the A&P section.

  Returns:
    LabeledCharSpan only if normalized span is valid (larger than 0 tokens).
  """
  token_start, token_end = char_span_to_token_span(
      tokens, (label.start_char, label.end_char))
  token_start, token_end = normalize_token_span((token_start, token_end),
                                                tokens)
  if token_end > token_start:
    char_start, char_end = token_span_to_char_span(tokens,
                                                   (token_start, token_end))
    new_labeled_char_span = copy.deepcopy(label)
    new_labeled_char_span.start_char = char_start
    new_labeled_char_span.end_char = char_end
    return new_labeled_char_span

  # Implicit else, token span is invalid:
  logging.debug(
      "Invalid labeled span after normalization. token span: (%d, %d)",
      token_start, token_end)


def normalize_labeled_char_spans_iterable(
    labeled_char_spans,
    tokens):
  """Normalizes multiple labeled_char_spans.

  Args:
    labeled_char_spans: An iterable of LabeledCharSpans pre-normalization.
    tokens: Tokens of the A&P section.

  Returns:
    List of LabeledCharSpans only if normalized span is valid (larger than 0
    tokens).
  """
  labeled_char_spans = [
      normalize_labeled_char_span(label, tokens) for label in labeled_char_spans
  ]
  return list(filter(None, labeled_char_spans))


def token_span_size_nonspaces(token_span,
                              tokens):
  assert token_span[1] >= token_span[0], token_span
  return len([
      token for token in tokens[token_span[0]:token_span[1]]
      if token.token_type != tokenizer_lib.TokenType.SPACE
  ])
