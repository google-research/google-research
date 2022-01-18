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

# coding=utf-8
"""Tests for tokenizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import six
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
from summae import tokenizer

FLAGS = tf.flags.FLAGS

_TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")


class TokenizerTest(tf.test.TestCase):

  def test_encode(self):
    self.assertListEqual(
        [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."],
        tokenizer.encode(u"Dude - that's so cool."))
    self.assertListEqual([u"Łukasz", u"est", u"né", u"en", u"1981", u"."],
                         tokenizer.encode(u"Łukasz est né en 1981."))
    self.assertListEqual([u" ", u"Spaces", u"at", u"the", u"ends", u" "],
                         tokenizer.encode(u" Spaces at the ends "))
    self.assertListEqual([u"802", u".", u"11b"], tokenizer.encode(u"802.11b"))
    self.assertListEqual([u"two", u". \n", u"lines"],
                         tokenizer.encode(u"two. \nlines"))

  def test_decode(self):
    self.assertEqual(
        u"Dude - that's so cool.",
        tokenizer.decode(
            [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]))

  def test_invertibility_on_random_strings(self):
    for _ in range(1000):
      s = u"".join(six.unichr(random.randint(0, 65535)) for _ in range(10))
      self.assertEqual(s, tokenizer.decode(tokenizer.encode(s)))


class TestTokenCounts(tf.test.TestCase):

  def setUp(self):
    super(TestTokenCounts, self).setUp()
    self.corpus_path = os.path.join(_TESTDATA, "corpus-*.txt")
    self.vocab_path = os.path.join(_TESTDATA, "vocab-*.txt")

  def test_corpus_token_counts_split_on_newlines(self):
    token_counts = tokenizer.corpus_token_counts(
        self.corpus_path, corpus_max_lines=0, split_on_newlines=True)

    expected = {
        u"'": 2,
        u".": 2,
        u". ": 1,
        u"... ": 1,
        u"Groucho": 1,
        u"Marx": 1,
        u"Mitch": 1,
        u"Hedberg": 1,
        u"I": 3,
        u"in": 2,
        u"my": 2,
        u"pajamas": 2,
    }
    self.assertDictContainsSubset(expected, token_counts)
    self.assertNotIn(u".\n\n", token_counts)
    self.assertNotIn(u"\n", token_counts)

  def test_corpus_token_counts_no_split_on_newlines(self):
    token_counts = tokenizer.corpus_token_counts(
        self.corpus_path, corpus_max_lines=0, split_on_newlines=False)

    self.assertDictContainsSubset({u".\n\n": 2, u"\n": 3}, token_counts)

  def test_corpus_token_counts_split_with_max_lines(self):
    token_counts = tokenizer.corpus_token_counts(
        self.corpus_path, corpus_max_lines=5, split_on_newlines=True)

    self.assertIn(u"slept", token_counts)
    self.assertNotIn(u"Mitch", token_counts)

  def test_corpus_token_counts_no_split_with_max_lines(self):
    token_counts = tokenizer.corpus_token_counts(
        self.corpus_path, corpus_max_lines=5, split_on_newlines=False)

    self.assertIn(u"slept", token_counts)
    self.assertNotIn(u"Mitch", token_counts)
    self.assertDictContainsSubset({
        u".\n\n": 1,
        u"\n": 2,
        u".\n": 1
    }, token_counts)

  def test_vocab_token_counts(self):
    token_counts = tokenizer.vocab_token_counts(self.vocab_path, 0)

    expected = {
        u"lollipop": 8,
        u"reverberated": 12,
        u"kattywampus": 11,
        u"balderdash": 10,
        u"jiggery-pokery": 14,
    }
    self.assertDictEqual(expected, token_counts)

  def test_vocab_token_counts_with_max_lines(self):
    # vocab-1 has 2 lines, vocab-2 has 3
    token_counts = tokenizer.vocab_token_counts(self.vocab_path, 5)

    expected = {
        u"lollipop": 8,
        u"reverberated": 12,
        u"kattywampus": 11,
        u"balderdash": 10,
    }
    self.assertDictEqual(expected, token_counts)


if __name__ == "__main__":
  tf.test.main()
