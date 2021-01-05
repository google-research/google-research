# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

import tokenize
from typing import List, Sequence, Tuple
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from cubert import python_tokenizer
from cubert import unified_tokenizer
_EOS_NAME = unified_tokenizer.quote_special(
    unified_tokenizer.TokenKind.EOS.name)
_ERROR_NAME = unified_tokenizer.quote_special(
    unified_tokenizer.TokenKind.ERROR.name)
_NEWLINE_NAME = unified_tokenizer.quote_special(
    unified_tokenizer.TokenKind.NEWLINE.name)


class PythonTokenizerTest(parameterized.TestCase):

  @mock.patch.object(
      unified_tokenizer,
      'code_to_tokens',
      return_value=[
          tokenize.TokenInfo(
              type=1000, string='spelling', start=(0, 0), end=(0, 0), line='')
      ])
  def test_python_tokenize_raises_on_unknown_python_token_kind(
      self, unused_function_mock):
    tokenizer = python_tokenizer.PythonTokenizer()
    with self.assertRaisesRegex(ValueError, 'While trying to turn'):
      tokenizer.tokenize('source')

  @parameterized.named_parameters(
      (
          'nothing',
          '',
          (
              (0, 0, 0, 0, unified_tokenizer.TokenKind.EOS),
          )),
      (
          'same_line',
          """TokenA TokenB""",
          #  0     67
          (
              (0, 0, 0, 6, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 7, 0, 13, unified_tokenizer.TokenKind.IDENTIFIER),
              # Python tokenizer always puts EOS in the beginning of the line
              # after the end of the file.
              (1, 0, 1, 0, unified_tokenizer.TokenKind.EOS),
          )),
      (
          'different_lines',
          """TokenA TokC\nTokenB""",
          #  0     67   11     6
          (
              (0, 0, 0, 6, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 7, 0, 11, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 11, 1, 0, unified_tokenizer.TokenKind.NEWLINE),
              (1, 0, 1, 6, unified_tokenizer.TokenKind.IDENTIFIER),
              (2, 0, 2, 0, unified_tokenizer.TokenKind.EOS),
          )),
      (
          'nl_tokens',
          """def f(\n): pass""",
          #  0   45  01 3   7
          (
              (0, 0, 0, 3, unified_tokenizer.TokenKind.KEYWORD),
              (0, 4, 0, 5, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 5, 0, 6, unified_tokenizer.TokenKind.PUNCTUATION),
              # Python's NL tokens are PUNCTUATION, not NEWLINE. However we
              # still treat them similarly to NEWLINE tokens, in that they
              # start at the end of a line and end at the beginning of the next
              # line.
              (0, 6, 1, 0, unified_tokenizer.TokenKind.PUNCTUATION),

              (1, 0, 1, 1, unified_tokenizer.TokenKind.PUNCTUATION),
              (1, 1, 1, 2, unified_tokenizer.TokenKind.PUNCTUATION),
              (1, 3, 1, 7, unified_tokenizer.TokenKind.KEYWORD),
              (2, 0, 2, 0, unified_tokenizer.TokenKind.EOS),
          )),
      (
          'nl_character',
          'a = "1\\n2" +  "10"',
          #  2 4567 89 11 14
          (
              (0, 0, 0, 1, unified_tokenizer.TokenKind.IDENTIFIER),
              (0, 2, 0, 3, unified_tokenizer.TokenKind.PUNCTUATION),
              # The NL character is like any other character, but of course it
              # counts as a single character.
              (0, 4, 0, 10, unified_tokenizer.TokenKind.STRING),
              (0, 11, 0, 12, unified_tokenizer.TokenKind.PUNCTUATION),
              (0, 14, 0, 18, unified_tokenizer.TokenKind.STRING),
              (1, 0, 1, 0, unified_tokenizer.TokenKind.EOS),
          )),
      (
          'unterminated_string',
          '"""ABC',
          #  2 4 6 8
          (
              (0, 0, 0, 0, unified_tokenizer.TokenKind.ERROR),
              (0, 0, 0, 0, unified_tokenizer.TokenKind.EOS),
          )),
      (
          'mismatched_indentation',
          """
class A():
  def f(): pass
 def g(): pass
""",
          (
              (0, 0, 0, 0, unified_tokenizer.TokenKind.ERROR),
              (0, 0, 0, 0, unified_tokenizer.TokenKind.EOS),
          )),
  )
  def test_python_tokenize_abstract_returns_positioning(
      self, source,
      expected_positions_and_kinds
  ):
    tokenizer = python_tokenizer.PythonTokenizer()

    agnostic = tokenizer.tokenize_and_abstract(source)
    actual_positions_and_kinds = tuple(
        (m.metadata.start.line, m.metadata.start.column, m.metadata.end.line,
         m.metadata.end.column, m.kind) for m in agnostic)

    self.assertSequenceEqual(expected_positions_and_kinds,
                             actual_positions_and_kinds)

  @parameterized.named_parameters(
      (
          'strings_are_tokenized',
          [],
          '"this is a string"',
          [
              '"^', 'this^', ' ^', 'is^', ' ^', 'a^', ' ^', 'string^', '"',
              _EOS_NAME
          ],
      ),
      (
          'newlines_in_strings_are_nlchar',
          [],
          '"""this\nis"""',
          ['"""^', 'this^', '___NLCHAR___^', 'is^', '"""', _EOS_NAME],
      ),
      (
          'snake_case_is_split',
          [],
          'a_b_c',
          ['a_^', 'b_^', 'c', _EOS_NAME],
      ),
      (
          'python_tokens_are_distinct',
          [],
          'def foo(bar)',
          ['def', 'foo', '(', 'bar', ')', _EOS_NAME],
      ),
      (
          'comments_are_skipped',
          [unified_tokenizer.TokenKind.COMMENT],
          'a = b\n#comment\nc = d',
          ['a', '=', 'b', _NEWLINE_NAME, '___NL___', 'c', '=', 'd', _EOS_NAME],
      ),
      (
          'comments_are_skipped_by_default',
          None,
          'a = b\n#comment\nc = d',
          ['a', '=', 'b', _NEWLINE_NAME, '___NL___', 'c', '=', 'd', _EOS_NAME],
      ),
      (
          'comments_are_not_skipped_when_desired',
          [],
          '# Comment.',
          # This NL token is preserved as ___NL___ and not replaced by
          # ___NLCHAR___ because it's introduced by the Python lexer for
          # well-formedness, and does not correspond to an actual \n character.
          # The corresponding token tuple has type NL and name ''.
          ['#^', ' ^', 'Comment^', '.', '___NL___', _EOS_NAME],
      ),
      (
          'explicit_newline_after_comment_is_always_nl_token',
          [],
          '# Comment 1.\n# Comment 2.',
          # All NL tokens are kept as special tokens, rather than representing
          # explicit ones with NLCHAR and implicit ones with NL.
          [
              '#^', ' ^', 'Comment^', ' ^', '1^', '.', '___NL___', '#^', ' ^',
              'Comment^', ' ^', '2^', '.', '___NL___', _EOS_NAME
          ],
      ),
      (
          'indent_dedent_replaced_with_keyword',
          [],
          'a\n\tb',
          [
              'a', '___NEWLINE___', '___INDENT___\t', 'b', '___DEDENT___',
              _EOS_NAME
          ],
      ),
      (
          'async_await_are_always_name_tokens',
          [],
          'async f():\n\tawait g()',
          [
              'async', 'f', '(', ')', ':', _NEWLINE_NAME, '___INDENT___\t',
              'await', 'g', '(', ')', '___DEDENT___', _EOS_NAME
          ],
      ),
  )
  def test_python_tokenize_returns_expected(self, types_to_skip, source,
                                            expected):
    tokenizer = python_tokenizer.PythonTokenizer()
    if types_to_skip is not None:
      tokenizer.update_types_to_skip(types_to_skip)
    actual = tokenizer.tokenize(source)
    self.assertListEqual(expected, actual)

  @parameterized.named_parameters(
      ('incomplete_multiline_token', '"""ABC'),
      ('mismatched_indents', """
class A(object):
  def f():
    pass
 def b():
   pass"""),
  )
  def test_python_tokenize_handles_tokenization_errors(self, bad_code):
    tokenizer = python_tokenizer.PythonTokenizer()
    actual = tokenizer.tokenize(bad_code)
    self.assertListEqual([_ERROR_NAME, _EOS_NAME], actual)

  @parameterized.named_parameters(
      (
          'reserved_python_token_is_preserved',
          ['ReServed', 'bad_mojo'],
          'ReServed == bad_mojo',
          ['ReServed', '==', 'bad_mojo', _EOS_NAME],
      ),
      (
          'unreserved_python_token_is_split',
          [],
          'ReServed == bad_mojo',
          ['Re^', 'Served', '==', 'bad_^', 'mojo', _EOS_NAME],
      ),
      (
          'reserved_substrings_are_split',
          ['bad_mojo'],
          'bad_mojo == "bad_mojo"',
          ['bad_mojo', '==', '"^', 'bad^', '_^', 'mojo^', '"', _EOS_NAME],
      ),
  )
  def test_python_tokenize_respects_reserved(self, reserved, source, expected):
    tokenizer = python_tokenizer.PythonTokenizer()
    tokenizer.replace_reserved_keywords(reserved)
    actual = tokenizer.tokenize(source)
    self.assertListEqual(expected, actual)

  @parameterized.named_parameters(
      (
          'mappings_are_respected',
          {
              '^': 'SENTINEL',
              'a': 'b',
          },
          'a == b',
          ['b', '==', 'b', _EOS_NAME],
      ),
      (
          'mappings_work_inside_tokens',
          {
              '^': 'SENTINEL',
              'o': 'ZZZ'
          },
          'for a in b:',
          ['fZZZr', 'a', 'in', 'b', ':', _EOS_NAME],
      ),
      (
          'mappings_work_after_token_retokenization',
          {
              '^': 'SENTINEL',
              'Z': 'xxx'
          },
          'a = "rZboing"',
          [
              'a',
              '=',
              '"^',
              # If white-space tokenization happened first, the following two
              # tokens would have ended up as one, since they're all lower-case.
              'r^',
              'xxxboing^',
              '"',
              _EOS_NAME
          ],
      ),
  )
  def test_python_tokenize_respects_mappings(self, mappings, source, expected):
    tokenizer = python_tokenizer.PythonTokenizer()
    tokenizer.update_mappings(mappings)
    actual = tokenizer.tokenize(source)
    self.assertListEqual(expected, actual)

  @parameterized.named_parameters(
      (
          '_empty_string',
          '',
          '',
      ),
      (
          '_subtokenization',
          'f_b_d = "blah blah"',
          'f_b_d ="blah blah"',
      ),
      (
          '_indent_dedent',
          'def f():\n'
          '  for i in B:\n'
          '         pass',  # Unconventional indent.
          'def f ():\n'
          '  for i in B :\n'
          '         pass ',
      ),
      (
          '_newlines',
          'def f(a,\n'
          'b):\n'
          '  a = "\\n"',
          'def f (a ,\n'
          'b ):\n'
          '  a ="\\n"',
      ),
      (
          '_comments',
          'def f():\n'
          '  # My comment\n'
          '  pass',
          'def f ():\n'
          '\n'
          '  pass ',
      ),
      (
          '_triple_quotes',
          '"""blah"""',
          '"""blah"""',
      ),
      (
          '_async_await',
          'async def f():\n'
          '  await g()',
          'async def f ():\n'
          '  await g ()',
      ),
  )
  def test_python_roundtrip(self, source, expected):
    tokenizer = python_tokenizer.PythonTokenizer()
    self.assertEqual(expected, tokenizer.untokenize(tokenizer.tokenize(source)))

  @parameterized.named_parameters(
      (
          '_comments',
          'def f():\n'
          '  # My comment\n'
          '  pass',
          'def f ():\n'
          '# My comment\n'  # Comments are not indented.
          '  pass ',
      ),
      (
          '_comments_after_code',
          'def f():\n'
          '  a = 1 + 13    # My comment\n'
          '  pass',
          'def f ():\n'
          '  a =1 +13 # My comment\n'  # No space preservation after comment.
          '  pass ',
      ),
  )
  def test_python_with_comment_roundtrip(self, source,
                                         expected):
    tokenizer = python_tokenizer.PythonTokenizer()
    tokenizer.update_types_to_skip([])
    self.assertEqual(expected, tokenizer.untokenize(tokenizer.tokenize(source)))

  @parameterized.named_parameters(
      ('_no_tokens', []),
      ('_no_eos', ['other']),
  )
  def test_python_untokenize_raises_as_expected(self,
                                                tokens):
    tokenizer = python_tokenizer.PythonTokenizer()
    with self.assertRaises(ValueError):
      tokenizer.untokenize(tokens)


if __name__ == '__main__':
  absltest.main()
