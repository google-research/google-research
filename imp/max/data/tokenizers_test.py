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

"""Tests for tokenizers."""

import copy

from absl.testing import absltest
import tensorflow as tf

from imp.max.data import tokenizers


class TokenizersTest(tf.test.TestCase):

  def test_tokenize(self):
    parsing_feature_name = 'raw_string'
    features = {
        'something': 'else',
        parsing_feature_name: ['This is the raw string']
    }
    tokenizer = tokenizers.get_tokenizer('bert_uncased_en')
    tokenizer.initialize()
    expected_with_raw = copy.deepcopy(features)
    output_feature_name = 'output_tokens'
    expected_with_raw[output_feature_name] = tf.convert_to_tensor([[2023,
                                                                    2003]])
    expected_no_raw = {'something': 'else'}
    expected_no_raw[output_feature_name] = tf.convert_to_tensor(
        [[101, 2023, 2003, 1996, 6315]])

    output_with_raw = tokenizers.tokenize(
        features,
        tokenizer,
        parsing_feature_name,
        output_feature_name,
        prepend_bos=False,
        append_eos=False,
        max_num_tokens=2,
        keep_raw_string=True)
    self.assertDictEqual(output_with_raw, expected_with_raw)

    output_no_raw = tokenizers.tokenize(
        features,
        tokenizer,
        parsing_feature_name,
        output_feature_name,
        prepend_bos=True,
        append_eos=True,
        max_num_tokens=5,
        keep_raw_string=False)
    self.assertDictEqual(output_no_raw, expected_no_raw)

  def test_crop_or_pad_words(self):
    words = tf.convert_to_tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    self.assertAllEqual(
        tokenizers.crop_or_pad_words(words, 1),
        tf.convert_to_tensor([[0], [3], [6], [9]]))
    self.assertAllEqual(tokenizers.crop_or_pad_words(words, 3), words)
    self.assertAllEqual(
        tokenizers.crop_or_pad_words(words, 4, 3),
        tf.convert_to_tensor([[0, 1, 2, 3], [3, 4, 5, 3], [6, 7, 8, 3],
                              [9, 10, 11, 3]]))
    with self.assertRaises(ValueError):
      tokenizers.crop_or_pad_words(words, 0)

  def test_get_tokenizer(self):
    self.assertIsInstance(
        tokenizers.get_tokenizer('bErt_uncased_en'), tokenizers.BertTokenizer)
    self.assertIsInstance(
        tokenizers.get_tokenizer('howto100M_en'), tokenizers.WordTokenizer)
    with self.assertRaises(ValueError):
      tokenizers.get_tokenizer('something')
    with self.assertRaises(ValueError):
      tokenizers.get_tokenizer('bert_de')

  def test_text_tokenize_t5_en(self):
    tokenizer = tokenizers.get_tokenizer('t5_en')
    tokenizer.initialize()

    example = 'This is the raw string'
    tokenized = tokenizer.string_tensor_to_indices([example], max_num_tokens=8)
    decoded = tokenizer._tf_sp_model.detokenize(tokenized)
    decoded = decoded[0].numpy().decode('utf8')

    self.assertAllEqual(tokenized, [[100, 19, 8, 5902, 6108, 0, 0, 0]])
    self.assertEqual(decoded, example)


if __name__ == '__main__':
  absltest.main()
