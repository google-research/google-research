# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
from absl.testing import flagsaver
import six
import tensorflow.compat.v1 as tf

from smith.bert import tokenization

FLAGS = flags.FLAGS


class TokenizationTest(tf.test.TestCase):

  def tokenize_with_full_tokenizer(self):
    """Returns tokens and ids processed with FullTokenizer."""
    text = u"UNwant\u00E9d,running [unused0] [CLS] [unused55]"
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing", ",", "[unused0]"
    ]
    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
      if six.PY2:
        vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
      else:
        vocab_writer.write("".join(
            [x + "\n" for x in vocab_tokens]).encode("utf-8"))

      vocab_file = vocab_writer.name

    tokenizer = tokenization.FullTokenizer(vocab_file)
    os.unlink(vocab_file)
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, ids

  def test_full_tokenizer(self):
    tokens, ids = self.tokenize_with_full_tokenizer()

    self.assertAllEqual(tokens, [
        "un", "##want", "##ed", ",", "runn", "##ing", "[UNK]", "[UNK]", "[UNK]",
        "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"
    ])

    self.assertAllEqual(ids, [7, 4, 5, 10, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  def test_full_tokenizer_preserve_unused(self):
    with flagsaver.flagsaver(preserve_unused_tokens=True):
      tokens, ids = self.tokenize_with_full_tokenizer()

    self.assertAllEqual(tokens, [
        "un", "##want", "##ed", ",", "runn", "##ing", "[unused0]", "[UNK]",
        "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"
    ])

    self.assertAllEqual(ids, [7, 4, 5, 10, 8, 9, 11, 0, 0, 0, 0, 0, 0])

  def test_chinese(self):
    tokenizer = tokenization.BasicTokenizer()

    self.assertAllEqual(
        tokenizer.tokenize(u"ah\u535A\u63A8zz"),
        [u"ah", u"\u535A", u"\u63A8", u"zz"])

  def test_basic_tokenizer_lower(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

    self.assertAllEqual(
        tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["hello", "!", "how", "are", "you", "?"])
    self.assertAllEqual(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])

  def test_basic_tokenizer_no_lower(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

    self.assertAllEqual(
        tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["HeLLo", "!", "how", "Are", "yoU", "?"])

  def test_wordpiece_tokenizer(self):
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing"
    ]

    vocab = {}
    for (i, token) in enumerate(vocab_tokens):
      vocab[token] = i
    tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

    self.assertAllEqual(tokenizer.tokenize(""), [])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running"),
        ["un", "##want", "##ed", "runn", "##ing"])

    self.assertAllEqual(
        tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

  def test_convert_tokens_to_ids(self):
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing"
    ]

    vocab = {}
    for (i, token) in enumerate(vocab_tokens):
      vocab[token] = i

    self.assertAllEqual(
        tokenization.convert_tokens_to_ids(
            vocab, ["un", "##want", "##ed", "runn", "##ing"]), [7, 4, 5, 8, 9])

  def test_is_whitespace(self):
    self.assertTrue(tokenization._is_whitespace(u" "))
    self.assertTrue(tokenization._is_whitespace(u"\t"))
    self.assertTrue(tokenization._is_whitespace(u"\r"))
    self.assertTrue(tokenization._is_whitespace(u"\n"))
    self.assertTrue(tokenization._is_whitespace(u"\u00A0"))

    self.assertFalse(tokenization._is_whitespace(u"A"))
    self.assertFalse(tokenization._is_whitespace(u"-"))

  def test_is_control(self):
    self.assertTrue(tokenization._is_control(u"\u0005"))

    self.assertFalse(tokenization._is_control(u"A"))
    self.assertFalse(tokenization._is_control(u" "))
    self.assertFalse(tokenization._is_control(u"\t"))
    self.assertFalse(tokenization._is_control(u"\r"))
    self.assertFalse(tokenization._is_control(u"\U0001F4A9"))

  def test_is_punctuation(self):
    self.assertTrue(tokenization._is_punctuation(u"-"))
    self.assertTrue(tokenization._is_punctuation(u"$"))
    self.assertTrue(tokenization._is_punctuation(u"`"))
    self.assertTrue(tokenization._is_punctuation(u"."))

    self.assertFalse(tokenization._is_punctuation(u"A"))
    self.assertFalse(tokenization._is_punctuation(u" "))


if __name__ == "__main__":
  tf.test.main()
