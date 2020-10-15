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
from typing import Sequence, Tuple


from absl.testing import absltest
from absl.testing import parameterized
from cubert import java_tokenizer
from cubert import unified_tokenizer


_NEWLINE_NAME = unified_tokenizer.quote_special(
    unified_tokenizer.TokenKind.NEWLINE.name)


class JavaTokenizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'nothing',
          '',
          (),
      ),
      (
          'same_line',
          """TokenA TokenB""",
          #  0     67
          (
              (0, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 6, unified_tokenizer.TokenKind.WHITESPACE),
              (0, 7, unified_tokenizer.TokenKind.IDENTIFIER),
              # We skip the EOS token.
          )),
      (
          'different_lines',
          """TokenA
TokenB TokenC""",
          (
              (0, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 6, unified_tokenizer.TokenKind.NEWLINE),
              (1, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (1, 6, unified_tokenizer.TokenKind.WHITESPACE),
              (1, 7, unified_tokenizer.TokenKind.IDENTIFIER),
          )),
      (
          'comment',
          '''TokenA /** comment */ TokenB''',
          (
              (0, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 6, unified_tokenizer.TokenKind.WHITESPACE),
              (0, 7, unified_tokenizer.TokenKind.COMMENT),
              (0, 21, unified_tokenizer.TokenKind.WHITESPACE),
              (0, 22, unified_tokenizer.TokenKind.IDENTIFIER),
          )),
      (
          'multi_line_comment',
          '''TokenA /** comment

*/ TokenB''',
          (
              (0, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 6, unified_tokenizer.TokenKind.WHITESPACE),
              (0, 7, unified_tokenizer.TokenKind.COMMENT),
              (2, 2, unified_tokenizer.TokenKind.WHITESPACE),
              (2, 3, unified_tokenizer.TokenKind.IDENTIFIER),
          )),
      (
          'multi_line_whitespace',
          # pylint:disable = trailing-whitespace
          # After "TokenA" there are 3 spaces that should be dropped by
          # the tokenizer.
          # Before "TokenB" there are 2 spaces that should not be dropped.
          '''TokenA

  TokenB''',
          # pylint:enable = trailing-whitespace
          (
              (0, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 6, unified_tokenizer.TokenKind.NEWLINE),
              (1, 0, unified_tokenizer.TokenKind.NEWLINE),
              (2, 0, unified_tokenizer.TokenKind.WHITESPACE),
              (2, 2, unified_tokenizer.TokenKind.IDENTIFIER),
          )),
      (
          'first_character_invalid',
          'TokenA\n#TokenB',
          (
              (0, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 6, unified_tokenizer.TokenKind.NEWLINE),
              (1, 0, unified_tokenizer.TokenKind.ERROR),
              (1, 1, unified_tokenizer.TokenKind.IDENTIFIER),
          )),
      (
          'middle_character_invalid',
          'TokenA#TokenB',
          (
              (0, 0, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 6, unified_tokenizer.TokenKind.ERROR),
              (0, 7, unified_tokenizer.TokenKind.IDENTIFIER),
          )),
      (
          'unterminated_string',
          'TokenA "mystring',
          (
              # For any lexer errors, we just abort and return an error.
              (0, 0, unified_tokenizer.TokenKind.ERROR),
          )),
      (
          'concluding_comment_with_newline',
          """  /* comment */
 """,
          (
              (0, 0, unified_tokenizer.TokenKind.WHITESPACE),
              (0, 2, unified_tokenizer.TokenKind.COMMENT),
              (0, 15, unified_tokenizer.TokenKind.NEWLINE),
              (1, 0, unified_tokenizer.TokenKind.WHITESPACE),
          )),
  )
  def test_abstraction_returns_expected(
      self, source,
      expected_starts_and_kinds
  ):
    tokenizer = java_tokenizer.JavaTokenizer()

    agnostic = tokenizer.tokenize_and_abstract(source)
    actual_starts_and_kinds = tuple(
        (m.metadata.start.line, m.metadata.start.column, m.kind)
        for m in agnostic[:-1])

    self.assertSequenceEqual(expected_starts_and_kinds,
                             actual_starts_and_kinds)

  @parameterized.named_parameters(
      ('single_line', 'package com.aye.bee.cee;',
       ('package', ' ', 'com', '.', 'aye', '.', 'bee', '.', 'cee', ';', '')),
      ('with_subtokenization', 'public   class CamelCase {',
       ('public', '   ', 'class', ' ', 'Camel^', 'Case', ' ', '{', '')),
      ('multiple_lines', """
public class CamelCase {

    private static final String SNAKE_CASE = "text string.continued";
    /* comment */

""",
       (
           # Line 0.
           _NEWLINE_NAME,

           # Line 1.
           'public', ' ', 'class', ' ', 'Camel^', 'Case', ' ', '{',
           _NEWLINE_NAME,

           # Line 2.
           _NEWLINE_NAME,  # Ignores extraneous whitespace on empty line.

           # Line 3.
           '    ', 'private', ' ', 'static', ' ', 'final', ' ', 'String', ' ',
           'SNAKE_^', 'CASE', ' ', '=', ' ', '"^', 'text^', ' ^', 'string^',
           '.^', 'continued^', '"', ';', _NEWLINE_NAME,

           # Line 4.
           '    ', '/*^', ' ^', 'comment^', ' ^', '*/', _NEWLINE_NAME,

           # Empty line 5.
           _NEWLINE_NAME,

           # Line 6.
           '',
           )),
      )
  def test_tokenize_returns_expected(
      self, source,
      expected
  ):
    tokenizer = java_tokenizer.JavaTokenizer()

    actual = tokenizer.tokenize(source)

    self.assertSequenceEqual(expected, actual)


if __name__ == '__main__':
  absltest.main()
