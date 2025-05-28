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

"""Tests for converters."""

import functools
from absl.testing import absltest
from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import ap_parsing_utils
from assessment_plan_modeling.ap_parsing import tokenizer_lib


class NormalizationTest(absltest.TestCase):

  def test_usage(self):
    text = "a&p:\n # dm2 "
    tokens = tokenizer_lib.tokenize(text)
    token_span = (3, 11)  # ":\n # dm2 "
    self.assertEqual(
        ap_parsing_utils.normalize_token_span(token_span, tokens),
        (8, 10))  # "dm2"

  def test_one_sided(self):
    text = "a&p:\n # dm2. "
    tokens = tokenizer_lib.tokenize(text)
    token_span = (6, 10)
    self.assertEqual(
        ap_parsing_utils.normalize_token_span(token_span, tokens),
        (8, 10))  # "dm2"

    token_span = (8, 11)
    self.assertEqual(
        ap_parsing_utils.normalize_token_span(token_span, tokens),
        (8, 10))  # "dm2"


class CharSpanToTokenSpanTest(absltest.TestCase):

  def test_usage(self):
    #       0   12     34     56 78   90
    text = "some longer tokens in this test"
    tokens = tokenizer_lib.tokenize(text)

    self.assertEqual(
        ap_parsing_utils.char_span_to_token_span(tokens, (5, 18)), (2, 5))

  def test_midtoken(self):
    #       0   12     34     56 78   90
    text = "some longer tokens in this test"
    tokens = tokenizer_lib.tokenize(text)

    # Mid word from the left.
    self.assertEqual(
        ap_parsing_utils.char_span_to_token_span(tokens, (7, 21)), (2, 7))

    # Mid word from the right.
    self.assertEqual(
        ap_parsing_utils.char_span_to_token_span(tokens, (5, 20)), (2, 7))


class NormalizeLabeledCharSpanTest(absltest.TestCase):

  def test_usage(self):
    text = "a&p:\n # dm2 "
    tokens = tokenizer_lib.tokenize(text)
    labeled_char_span = ap_parsing_lib.LabeledCharSpan(
        start_char=3,
        end_char=12,
        span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE)  # ":\n # dm2 "
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_span(labeled_char_span, tokens),
        ap_parsing_lib.LabeledCharSpan(
            start_char=8,
            end_char=11,
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE))  # "dm2"

  def test_metadata(self):
    text = "a&p:\n - nebs "
    tokens = tokenizer_lib.tokenize(text)
    labeled_char_span = ap_parsing_lib.LabeledCharSpan(
        start_char=3,
        end_char=11,
        span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
        action_item_type=ap_parsing_lib.ActionItemType
        .MEDICATIONS  # ":\n - neb"
    )
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_span(labeled_char_span, tokens),
        ap_parsing_lib.LabeledCharSpan(
            start_char=8,
            end_char=12,
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS)
    )  # "nebs"

  def test_one_sided(self):
    text = "a&p:\n # dm2 "
    tokens = tokenizer_lib.tokenize(text)

    # Remove suffix only.
    labeled_char_span = ap_parsing_lib.LabeledCharSpan(
        start_char=8,
        end_char=12,
        span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE)  # "dm2 "
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_span(labeled_char_span, tokens),
        ap_parsing_lib.LabeledCharSpan(
            start_char=8,
            end_char=11,
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE))  # "dm2"

    # Remove prefix only.
    labeled_char_span = ap_parsing_lib.LabeledCharSpan(
        start_char=3,
        end_char=11,
        span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE)  # ":\n # dm2"
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_span(labeled_char_span, tokens),
        ap_parsing_lib.LabeledCharSpan(
            start_char=8,
            end_char=11,
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE))  # "dm2"

  def test_midword(self):
    text = "a&p:\n # COPD: on nebs "
    tokens = tokenizer_lib.tokenize(text)

    # Extend word boundry right.
    labeled_char_span = ap_parsing_lib.LabeledCharSpan(
        start_char=6,
        end_char=11,
        span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE)  # "# COP"
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_span(labeled_char_span, tokens),
        ap_parsing_lib.LabeledCharSpan(
            start_char=8,
            end_char=12,
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE))  # "COPD"

    # Extend word boundry left.
    labeled_char_span = ap_parsing_lib.LabeledCharSpan(
        start_char=9,
        end_char=14,
        span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE)  # "OPD: "
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_span(labeled_char_span, tokens),
        ap_parsing_lib.LabeledCharSpan(
            start_char=8,
            end_char=12,
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE))  # "COPD"

    # Extend word boundry both directions.
    labeled_char_span = ap_parsing_lib.LabeledCharSpan(
        start_char=9,
        end_char=11,
        span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE)  # "OP"
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_span(labeled_char_span, tokens),
        ap_parsing_lib.LabeledCharSpan(
            start_char=8,
            end_char=12,
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE))  # "COPD"


class NormalizeLabeledCharSpansIterableTest(absltest.TestCase):

  def test_usage(self):
    #       0   12     34     56 78   90
    text = "some longer tokens in this test"
    #       0123456789012345678901234567890
    #       0         1         2         3

    tokens = tokenizer_lib.tokenize(text)

    labeled_char_spans = [
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE,
            start_char=5,
            end_char=18),  # "longer tokens" - already normalized.
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE,
            start_char=14,
            end_char=25),  # "kens in thi" -> "tokens in this"
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE,
            start_char=18,
            end_char=19),  # Invalid - only space.
    ]
    expected = [
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE,
            start_char=5,
            end_char=18),  # "longer tokens"
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE,
            start_char=12,
            end_char=26)  # "tokens in this"
    ]
    self.assertEqual(
        ap_parsing_utils.normalize_labeled_char_spans_iterable(
            labeled_char_spans, tokens), expected)


class ConvertersTest(absltest.TestCase):

  def test_labeled_char_spans_to_token_spans(self):
    #  Char space:
    #       0         1          2          3
    #       01234567890123456 78901234 56789012345
    text = "# DM2: on insulin\n # COPD\n- nebs prn"
    # Token:012 3456 78       90123    4567   89
    #       0                  1
    tokens = tokenizer_lib.tokenize(text)

    labeled_char_spans = [
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=2,
            end_char=17),  # "DM2: on insulin"
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=21,
            end_char=25),  # "COPD"
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            start_char=28,
            end_char=32),  # "nebs"
    ]

    labeled_token_spans = [
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_token=2,
            end_token=9),  # "DM2: on insulin"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_token=13,
            end_token=14),  # "COPD"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            start_token=17,
            end_token=18),  # "nebs"
    ]

    labeled_char_span_to_labeled_token_span = functools.partial(
        ap_parsing_utils.labeled_char_span_to_labeled_token_span, tokens=tokens)
    self.assertEqual(labeled_token_spans, [
        labeled_char_span_to_labeled_token_span(labeled_char_span)
        for labeled_char_span in labeled_char_spans
    ])

    labeled_token_span_to_labeled_char_span = functools.partial(
        ap_parsing_utils.labeled_token_span_to_labeled_char_span, tokens=tokens)
    self.assertEqual(labeled_char_spans, [
        labeled_token_span_to_labeled_char_span(labeled_token_span)
        for labeled_token_span in labeled_token_spans
    ])


if __name__ == "__main__":
  absltest.main()
