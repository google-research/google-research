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

"""Tests for java_tokenizer."""
from typing import Sequence


from absl.testing import absltest
from absl.testing import parameterized
from cubert import java_tokenizer
from cubert import unified_tokenizer


class JavaTokenizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'nothing',
          '',
          (),
          (),
      ),
      (
          'same_line',
          """TokenA TokenB""",
          (0, 0),
          (0, 7),
      ),
      (
          'different_lines',
          """TokenA
TokenB TokenC""",
          (0,
           0,  # NEWLINE
           1, 1),
          (0,
           java_tokenizer._MAX_COLUMN,  # NEWLINE
           0, 7),
      ),
      (
          'comment',
          '''TokenA /** comment */ TokenB''',
          (0, 0),
          (0, 22),
      ),
      (
          'multi_line_comment',
          '''TokenA /** comment

*/ TokenB''',
          (0,
           0, 1,  # NEWLINEs.
           2),
          (0,
           java_tokenizer._MAX_COLUMN, java_tokenizer._MAX_COLUMN,  # NEWLINEs.
           3),
      ),
  )
  def test_tokenization_returns_expected_positions(
      self, source, expected_lines,
      expected_columns):
    tokenizer = java_tokenizer.JavaTokenizer()

    # Produce multi-tokens, right before flattening.
    agnostic = tokenizer.tokenize_and_abstract(source)
    conditioned = tokenizer.condition_full_tokens(agnostic)
    multi_tokens = tokenizer.subtokenize_full_tokens(conditioned)[:-1]
    actual_lines_and_columns = tuple(
        (m.metadata.start.line, m.metadata.start.column) for m in multi_tokens)
    expected_lines_and_columns = tuple(zip(expected_lines, expected_columns))

    self.assertSequenceEqual(expected_lines_and_columns,
                             actual_lines_and_columns)

  @parameterized.named_parameters(
      (
          'same_line',
          """TokenA TokenB""",
          (),
      ),
      (
          'different_lines',
          """TokenA
TokenB

TokenC""",
          (0, 1, 2),
      ),
  )
  def test_tokenization_returns_expected_newlines(
      self, source, expected_newline_lines):
    tokenizer = java_tokenizer.JavaTokenizer()

    # Produce multi-tokens, right before flattening.
    agnostic = tokenizer.tokenize_and_abstract(source)
    conditioned = tokenizer.condition_full_tokens(agnostic)
    multi_tokens = tokenizer.subtokenize_full_tokens(conditioned)[:-1]
    actual_newline_lines = tuple(
        m.metadata.start.line
        for m in multi_tokens
        if m.kind == unified_tokenizer.TokenKind.NEWLINE)

    self.assertSequenceEqual(expected_newline_lines,
                             actual_newline_lines)


if __name__ == '__main__':
  absltest.main()
