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

# -*- coding: utf-8 -*
"""Tests for tokenization utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from qanet.util import tokenizer_util


class TokenizerUtilTest(tf.test.TestCase):

  def testConvertToSpans(self):
    raw_text = "Convert to spans."
    tokens = ["Convert", "to", "spans", "."]
    output = tokenizer_util.convert_to_spans(raw_text, tokens)
    expected_output = [(0, 7), (8, 10), (11, 16), (16, 17)]
    self.assertEqual(expected_output, output)

  def testConvertToSpansMismatches(self):
    raw_text = u"America — Harvard, Yale"  # "—": em-dash (\u2014)
    tokens = ["America", u"\u2013", "Harvard", ",", "Yale"]  # \u2013: en-dash
    output = tokenizer_util.convert_to_spans(raw_text, tokens)
    expected_output = [(0, 7), (8, 9), (10, 17), (17, 18), (19, 23)]
    self.assertEqual(expected_output, output)

  def testGetAnswer(self):
    context = u"America — Harvard, Yale"  # "—": em-dash (\u2014)
    # \u2013: en-dash
    context_words = ["America", u"\u2013", "Harvard", ",", "Yale"]
    word_answer_start = 0
    word_answer_end = 2
    output = tokenizer_util.get_answer(
        context, context_words, word_answer_start, word_answer_end)
    expected_output = u"America — Harvard"
    self.assertEqual(expected_output, output)

  def testGetAnswerBytes(self):
    """Test when context and context_words have byte sequences."""
    # "—": em-dash (\u2014)
    context = u"America — Harvard, Yale".encode("utf-8")
    # \u2013: en-dash
    context_words = ["America", u"\u2013".encode("utf-8"), "Harvard", ",",
                     "Yale"]
    word_answer_start = 0
    word_answer_end = 2
    output = tokenizer_util.get_answer(
        context, context_words, word_answer_start, word_answer_end,
        is_byte=True)
    expected_output = u"America — Harvard".encode("utf-8")
    self.assertEqual(expected_output, output)

if __name__ == "__main__":
  tf.test.main()
