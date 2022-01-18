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

"""Tests for code_to_subtokenized_sentences."""
from typing import Sequence


from absl.testing import absltest
from absl.testing import parameterized
from tensor2tensor.data_generators import text_encoder

from cubert import code_to_subtokenized_sentences
from cubert import python_tokenizer


class CodeToSubtokenizedSentencesTest(parameterized.TestCase):

  _CODE = 'def foo_bar(): return("ab, cd")'
  _RECONSTITUTED_CODE = 'def foo_bar ():return ("ab, cd")\n'
  _WORDPIECE_VOCABULARY = ('def_', 'foo\\u^_', 'bar_', '(_', ')_', ':_', 'ret',
                           'urn_', '"^_', 'ab^_', ',^_', ' ^_', 'cd^_', '"_',
                           '\\u\\u\\uNEWLINE\\u\\u\\u_', 'uv', 'xy', 'yz_',
                           'ww^_', 'www_')
  _QUOTED_VOCABULARY = (f'"{w}"' for w in _WORDPIECE_VOCABULARY)
  _VOCABULARY_CONTENT = '\n'.join(_QUOTED_VOCABULARY)
  _WORDPIECE_SUBTOKENS = [
      'def_', 'foo\\u^_', 'bar_', '(_', ')_', ':_', 'ret', 'urn_', '(_', '"^_',
      'ab^_', ',^_', ' ^_', 'cd^_', '"_', ')_', '\\u\\u\\uNEWLINE\\u\\u\\u_'
  ]

  def setUp(self):
    super().setUp()
    self.tokenizer = python_tokenizer.PythonTokenizer()
    subword_vocabulary_path = self.create_tempfile(
        content=self._VOCABULARY_CONTENT).full_path
    self.subword_text_encoder = text_encoder.SubwordTextEncoder(
        subword_vocabulary_path)

  def test_code_to_sentences(self):
    sentences = code_to_subtokenized_sentences.code_to_cubert_sentences(
        self._CODE, self.tokenizer, self.subword_text_encoder)
    self.assertEqual([self._WORDPIECE_SUBTOKENS], sentences)

  def test_wordpiece_tokens_to_code(self):
    code = code_to_subtokenized_sentences.wordpiece_subtokens_to_code(
        self._WORDPIECE_SUBTOKENS, self.tokenizer, self.subword_text_encoder)
    self.assertEqual(self._RECONSTITUTED_CODE, code)

  @parameterized.named_parameters(
      ('complete_wordpiece_only', ('uv', 'yz_'), 'uvyz', 2),
      ('complete_wordpiece_prefix', ('uv', 'yz_', 'yz_'), 'uvyz', 2),
      ('complete_wordpiece_before_incomplete', ('uv', 'yz_', 'xy'), 'uvyz', 2),
      ('complete_cubert_only', ('uv', 'ww^_', 'yz_'), 'uvwwyz', 3),
      ('complete_cubert_prefix', ('uv', 'ww^_', 'yz_', 'yz_'), 'uvwwyz', 3),
      ('complete_cubert_before_incomplete',
       ('uv', 'ww^_', 'yz_', 'xy'), 'uvwwyz', 3),
      ('complete_cubert_before_incomplete_cubert',
       ('uv', 'ww^_', 'yz_', 'ww^_'), 'uvwwyz', 3),
      # Here SubwordTextEncoder accepts a partial WordPiece subtoken, and we
      # interpret it as best we can.
      ('complete_cubert_subtoken_before_incomplete_wordpiece',
       ('uv', 'ww^_', 'xy'), 'uvwwxy', 3),
      # The SubwordTextEncoder accepts a partial WordPiece subtoken.
      ('incomplete_wordpiece_only', ('uv',), 'uv', 1),
  )
  def test_next_whole_token(self, subtokens,
                            expected_whole_token,
                            expected_end_index):
    actual_whole_result = code_to_subtokenized_sentences.next_whole_token(
        subtokens, self.tokenizer, self.subword_text_encoder)
    self.assertEqual((expected_whole_token, expected_end_index),
                     actual_whole_result)

  @parameterized.named_parameters(
      ('incomplete_cubert_only', ('uv', 'ww^_')),
  )
  def test_next_whole_token_fails(self, subtokens):
    with self.assertRaises(ValueError):
      _ = code_to_subtokenized_sentences.next_whole_token(
          subtokens, self.tokenizer, self.subword_text_encoder)


if __name__ == '__main__':
  absltest.main()
