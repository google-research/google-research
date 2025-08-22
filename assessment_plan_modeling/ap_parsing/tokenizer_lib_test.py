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

"""Tests for tokenizer."""

from absl.testing import absltest

from assessment_plan_modeling.ap_parsing import tokenizer_lib


class TokenizerTest(absltest.TestCase):

  def test_usage(self):
    text = "Word \t\n12.2[**Name**]"
    tokenizer = tokenizer_lib
    self.assertEqual(
        tokenizer.tokenize(text), [
            tokenizer_lib.Token(0, 4, tokenizer_lib.TokenType.WORD, "Word"),
            tokenizer_lib.Token(4, 5, tokenizer_lib.TokenType.SPACE, " "),
            tokenizer_lib.Token(5, 6, tokenizer_lib.TokenType.SPACE, "\t"),
            tokenizer_lib.Token(6, 7, tokenizer_lib.TokenType.SPACE, "\n"),
            tokenizer_lib.Token(7, 9, tokenizer_lib.TokenType.NUM, "12"),
            tokenizer_lib.Token(9, 10, tokenizer_lib.TokenType.PUNCT, "."),
            tokenizer_lib.Token(10, 11, tokenizer_lib.TokenType.NUM, "2"),
            tokenizer_lib.Token(11, 21, tokenizer_lib.TokenType.DEID,
                                "[**Name**]"),
        ])


if __name__ == "__main__":
  absltest.main()
