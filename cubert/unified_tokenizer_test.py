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

# coding=utf-8
"""The tests for unified_tokenizer.py."""

from absl.testing import absltest
from absl.testing import parameterized
import six

from cubert import unified_tokenizer


class SubtokenizeIdentifierTest(parameterized.TestCase):

  # We use these Unicode characters as representatives of different property
  # categories. Note that some characters have been added in subsequent
  # editions of the Unicode standard, and may be known by Python 3 but not by
  # Python 2. That was the case, for example for `Ꞧ` (U+A7A6, LATIN CAPITAL
  # LETTER R WITH OBLIQUE STROKE), which was in category Lu in Python 3, but
  # in the unknown category in Python2. We select examples that are known to
  # Python 2 as well.
  # Lu (letter uppercase, but not ASCII):
  #     `𝕁` (U+1D541, MATHEMATICAL DOUBLE-STRUCK CAPITAL J)
  # Lt (letter title case):
  #     `ᾩ` (U+1FA9, GREEK CAPITAL LETTER OMEGA WITH DASIA AND PROSGEGRAMMENI)
  # Ll (letter lowercase, but not ASCII):
  #     `ᾶ` (U+1FB6, GREEK SMALL LETTER ALPHA WITH PERISPOMENI)
  # Nc (decimal number):
  #     `൬` (U+0D6C, MALAYALAM DIGIT SIX)
  @parameterized.named_parameters(
      (
          'single_title_case',
          'ᾩ',
          ['ᾩ'],
      ),
      (
          'single_capital',
          'A',
          ['A'],
      ),
      (
          'single_uppercase',
          '𝕁',
          ['𝕁'],
      ),
      (
          'uppercase_and_titlecase',
          'ᾩT𝕁E',
          ['ᾩ', 'T𝕁E'],
      ),
      (
          'uppercase_unicode',
          'T𝕁E',
          ['T𝕁E'],
      ),
      (
          'all_capital_unicode',
          '__T𝕁E_𝕁HECK_E𝕁__',
          ['__', 'T𝕁E_', '𝕁HECK_', 'E𝕁__'],
      ),
      (
          'all_capital',
          '__THE_CHECK_EQ__',
          ['__', 'THE_', 'CHECK_', 'EQ__'],
      ),
      (
          'all_capital_and_titlecase_unicode',
          '__ᾩT𝕁E_𝕁HEᾩCK_E𝕁ᾩ__',
          #  ^        ^     ^
          ['__', 'ᾩ', 'T𝕁E_', '𝕁HE', 'ᾩ', 'CK_', 'E𝕁', 'ᾩ__'],
          #       ^ equivalent to <uppercase><lowercase>, so ᾩTRE is split.
      ),
      (
          'mixed_conventions',
          'CHECK__snABCke__caSSe5_8A3__',
          ['CHECK__', 'sn', 'AB', 'Cke__', 'ca', 'S', 'Se5_', '8', 'A3__'],
      ),
      (
          'mixed_conventions_unicode',
          'CH𝕁CK__sᾶABCke__caSᾩe5_൬A3__',
          #  ^Lu   ^Ll        ^Lt  ^Nl
          ['CH𝕁CK__', 'sᾶ', 'AB', 'Cke__', 'ca', 'S', 'ᾩe5_', '൬', 'A3__'],
      ),
      (
          'just_underscores',
          '__',
          ['__'],
      ),
      (
          'just_capitals',
          'ABC',
          ['ABC'],
      ),
      (
          'just_capitals_unicode',
          'A𝕁C',
          ['A𝕁C'],
      ),
      (
          'just_capitals_unicode_first',
          '𝕁CA',
          ['𝕁CA'],
      ),
      (
          'just_capitals_unicode_last',
          'AC𝕁',
          ['AC𝕁'],
      ),
      (
          'regular_snake',
          'abc_efg',
          ['abc_', 'efg'],
      ),
      (
          'regular_snake_unicode',
          'abcᾶ൬_cᾶ൬ab',
          ['abcᾶ൬_', 'cᾶ൬ab'],
      ),
      (
          'regular_camel',
          'abcEfg',
          ['abc', 'Efg'],
      ),
      (
          'regular_camel_unicode_uppercase',
          'abcᾶ൬𝕁cᾶ൬ab',
          #      ^^
          ['abcᾶ൬', '𝕁cᾶ൬ab'],
      ),
      (
          'regular_camel_unicode_titlecase',
          'abcᾶ൬ᾩcᾶ൬ab',
          #       ^
          ['abcᾶ൬', 'ᾩcᾶ൬ab'],
      ),
      (
          'regular_pascal',
          'AbcEfg',
          ['Abc', 'Efg'],
      ),
      (
          'regular_pascal_unicode_uppercase',
          '𝕁abcᾶ൬𝕁cᾶ൬ab',
          #^      ^
          ['𝕁abcᾶ൬', '𝕁cᾶ൬ab'],
      ),
      (
          'regular_pascal_unicode_titlecase',
          'ᾩabcᾶ൬ᾩcᾶ൬ab',
          #^       ^
          ['ᾩabcᾶ൬', 'ᾩcᾶ൬ab'],
      ),
  )
  def test_subtokenize_identifier_returns_expected(self, identifier,
                                                   components):
    actual_components = unified_tokenizer.subtokenize_identifier(identifier)
    self.assertListEqual([six.ensure_text(c) for c in components],
                         [six.ensure_text(c) for c in actual_components])


class SanitizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_mappings', 'abcabc', {}, 'abcabc'),
      ('no_text', '', {
          'a': 'ALPHA'
      }, ''),
      ('found', 'abcabc', {
          'b': 'BETA'
      }, 'aBETAcaBETAc'),
      ('not_found', 'abcabc', {
          'z': 'ZETA'
      }, 'abcabc'),
      ('two', 'abcabc', {
          'b': 'BETA',
          'z': 'ZETA',
      }, 'aBETAcaBETAc'),
  )
  def test_sanitize_returns_expected(self, text, mappings, expected):
    sanitized = unified_tokenizer.sanitize(text, mappings)
    self.assertEqual(expected, sanitized)

  @parameterized.named_parameters(
      ('no_mappings', 'abcabc', {}, 'abcabc'),
      ('no_text', '', {
          'a': 'ALPHA'
      }, ''),
      ('found', 'aBETAcaBETAc', {
          'b': 'BETA'
      }, 'abcabc'),
      ('not_found', 'abcabc', {
          'z': 'ZETA'
      }, 'abcabc'),
      ('two', 'aBETAcaBETAc', {
          'b': 'BETA',
          'z': 'ZETA',
      }, 'abcabc'),
  )
  def test_unsanitize_returns_expected(self, sanitized, mappings, expected):
    unsanitized = unified_tokenizer.unsanitize(sanitized, mappings)
    self.assertEqual(expected, unsanitized)


class SplitLongTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          '_short_token',
          'abc',
          ['abc'],
      ),
      (
          '_long_token',
          'abcde',
          ['abc', 'de'],
      ),
      (
          '_long_even_token',
          'abcdef',
          ['abc', 'def'],
      ),
  )
  def test_split_long_returns_expected(self, token, components):
    actual_components = unified_tokenizer.split_long_token(
        token_string=token, max_output_token_length=3)
    self.assertSequenceEqual(components, actual_components)

  def test_split_long_raises_on_empty(self):
    with self.assertRaises(ValueError):
      unified_tokenizer.split_long_token(
          token_string='', max_output_token_length=3)


class SplitAgnosticTest(parameterized.TestCase):

  _COMMENT = unified_tokenizer.TokenKind.COMMENT
  _IDENTIFIER = unified_tokenizer.TokenKind.IDENTIFIER
  _NUMBER = unified_tokenizer.TokenKind.NUMBER
  _STRING = unified_tokenizer.TokenKind.STRING

  @parameterized.named_parameters(
      (
          'empty_results_in_empty',
          [],
          10,
          [],
      ),
      (
          'unsplittable_identifiers',
          [('a1', _IDENTIFIER), ('bb', _IDENTIFIER)],
          10,
          [(['a1'], _IDENTIFIER), (['bb'], _IDENTIFIER)],
      ),
      (
          'heuristically_splittable_identifiers',
          [('a1_b2', _IDENTIFIER), ('bb', _IDENTIFIER)],
          10,
          [(['a1_', 'b2'], _IDENTIFIER), (['bb'], _IDENTIFIER)],
      ),
      (
          'heuristically_unsplittable_text',
          [
              ('z', _STRING),
              ('b', _COMMENT),
              ('zc', _STRING),
              ('bb', _COMMENT),
          ],
          10,
          [
              (['z'], _STRING),
              (['b'], _COMMENT),
              (['zc'], _STRING),
              (['bb'], _COMMENT),
          ],
      ),
      (
          'heuristically_splittable_text',
          [
              ('zz1', _STRING),
              ('b b', _COMMENT),
              ('zc', _STRING),
              ('bb', _COMMENT),
          ],
          10,
          [
              (['zz', '1'], _STRING),
              (['b', ' ', 'b'], _COMMENT),
              (['zc'], _STRING),
              (['bb'], _COMMENT),
          ],
      ),
      (
          'size_splittable_subtokens',
          [
              ('a123_b2', _IDENTIFIER),
              ('bbcd', _IDENTIFIER),
              ('zzx12', _STRING),
              ('b b', _COMMENT),
              ('zc', _STRING),
              ('1234', _NUMBER),
              ('bb', _COMMENT),
              ('11', _NUMBER),
          ],
          2,
          [
              (['a1', '23', '_', 'b2'], _IDENTIFIER),
              (['bb', 'cd'], _IDENTIFIER),
              (['zz', 'x', '12'], _STRING),
              (['b', ' ', 'b'], _COMMENT),
              (['zc'], _STRING),
              (['12', '34'], _NUMBER),
              (['bb'], _COMMENT),
              (['11'], _NUMBER),
          ],
      ),
      (
          'language_strings_unaffected',
          [
              ('punctuation', unified_tokenizer.TokenKind.PUNCTUATION),
              ('keyworddddd', unified_tokenizer.TokenKind.KEYWORD),
              ('newlineeeee', unified_tokenizer.TokenKind.NEWLINE),
              ('eosssssssss', unified_tokenizer.TokenKind.EOS),
              ('errrrorrrrr', unified_tokenizer.TokenKind.ERROR),
          ],
          2,
          [
              (['punctuation'], unified_tokenizer.TokenKind.PUNCTUATION),
              (['keyworddddd'], unified_tokenizer.TokenKind.KEYWORD),
              (['newlineeeee'], unified_tokenizer.TokenKind.NEWLINE),
              (['eosssssssss'], unified_tokenizer.TokenKind.EOS),
              (['errrrorrrrr'], unified_tokenizer.TokenKind.ERROR),
          ],
      ),
  )
  def test_split_agnostic_returns_expected(self, labelled_tokens, max_length,
                                           expected_labelled_subtokens):
    tokens = [
        unified_tokenizer.AbstractToken(s, k, unified_tokenizer.TokenMetadata())
        for s, k in labelled_tokens
    ]
    labelled_subtokens = unified_tokenizer.split_agnostic_tokens(
        tokens, max_length)

    expected_multi_tokens = []
    for spelling_list, kind in expected_labelled_subtokens:
      expected_multi_tokens.append(
          unified_tokenizer.AbstractMultiToken(
              # We cast spellings to tuples, since we know that
              # `split_agnostic_tokens` creates multi tokens with tuples rather
              # than lists.
              spellings=tuple(spelling_list),
              kind=kind,
              metadata=unified_tokenizer.TokenMetadata()))

    self.assertSequenceEqual(expected_multi_tokens, labelled_subtokens)


class FlattenUnflattenTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'singleton_subtoken_list',
          [['ab'], ['bc']],
          {
              '^': 'CARET',
          },
          ['ab', 'bc'],
      ),
      (
          'subtokenized_token_list',
          [['ab', '1', '2'], ['bc']],
          {
              '^': 'CARET',
          },
          ['ab^', '1^', '2', 'bc'],
      ),
      (
          'sanitize_one',
          [['ab', '1', 'a'], ['bc']],
          {
              '^': 'CARET',
              'a': 'ALPHA',
          },
          ['ALPHAb^', '1^', 'ALPHA', 'bc'],
      ),
      (
          'sanitize_two',
          [['ab', '1', 'a'], ['bc']],
          {
              '^': 'CARET',
              'a': 'ALPHA',
              'b': 'BETA',
          },
          ['ALPHABETA^', '1^', 'ALPHA', 'BETAc'],
      ),
  )
  def test_flatten_returns_expected(self, subtoken_lists, mappings,
                                    expected_subtoken_list):
    subtokens = unified_tokenizer.flatten_and_sanitize_subtoken_lists(
        subtoken_lists, mappings, sentinel='^')
    self.assertSequenceEqual(expected_subtoken_list, subtokens)

  @parameterized.named_parameters(
      (
          'empty_input',
          [],
          {
              '^': 'CARET'
          },
      ),
      (
          'empty_sublist',
          [[]],
          {
              '^': 'CARET'
          },
      ),
      (
          'interspersed_empty_sublist',
          [['a'], [], ['c', 'd']],
          {
              '^': 'CARET'
          },
      ),
      (
          'missing_sentinel',
          [['a'], [], ['c', 'd']],
          {},
      ),
  )
  def test_flatten_raises_when_expected(self, list_of_lists, mapping):
    with self.assertRaises(ValueError):
      unified_tokenizer.flatten_and_sanitize_subtoken_lists(
          list_of_lists, sanitization_mapping=mapping, sentinel='^')

  @parameterized.named_parameters(
      (
          'single',
          ['ab'],
          {
              '^': 'CARET',
          },
          ['ab'],
      ),
      (
          'multiple',
          ['ab^', '1', '2^', 'bc'],
          {
              '^': 'CARET',
          },
          ['ab1', '2bc'],
      ),
      (
          'unsanitize_one',
          ['ALPHAb^', '1', 'ALPHA^', 'bc'],
          {
              '^': 'CARET',
              'a': 'ALPHA',
          },
          ['ab1', 'abc'],
      ),
      (
          'unsanitize_two',
          ['ALPHAb^', 'ONE', 'b^', 'ALPHAc'],
          {
              '^': 'CARET',
              'a': 'ALPHA',
              '1': 'ONE',
          },
          ['ab1', 'bac'],
      ),
      (
          'sentinel_reconstruction',
          ['CARET^', '1'],
          {
              '^': 'CARET',
          },
          ['^1'],
      ),
  )
  def test_reconstitute_returns_expected(self, subtokens, mappings,
                                         expected_tokens):
    whole_tokens = unified_tokenizer.reconstitute_full_unsanitary_tokens(
        subtokens, mappings, sentinel='^')
    self.assertSequenceEqual(expected_tokens, whole_tokens)

  @parameterized.named_parameters(
      (
          'missing_sentinel',
          ['a', 'c', 'd'],
          {},
          #^
      ),
      (
          'empty_input',
          [],
          {
              '^': 'CARET'
          },
      ),
      (
          'empty_subtoken',
          [''],
          {
              '^': 'CARET'
          },
      ),
      (
          'interspersed_empty_subtoken',
          ['a', '', 'c', 'd'],
          #    ^^^^
          {
              '^': 'CARET'
          },
      ),
      (
          'unterminated_subtoken',
          ['a^', 'b', 'c^'],
          #            ^^^ should be followed by a subtoken without sentinel.
          {
              '^': 'CARET'
          },
      ),
      (
          'unsanitary_character_in_input',
          ['^^', '1'],
          # ^ Can't have a caret there.
          {
              '^': 'CARET',
          },
      ),
  )
  def test_reconstitute_raises_when_expected(self, subtokens, mappings):
    with self.assertRaises(ValueError):
      unified_tokenizer.reconstitute_full_unsanitary_tokens(
          subtokens, mappings, sentinel='^')


class CheckMappingsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'empty',
          {},
      ),
      (
          'clean',
          {
              'a': 'ALPHA',
              'b': 'BETA',
          },
      ),
  )
  def test_check_mappings_accepts_legitimate(self, mappings):
    unified_tokenizer.check_mappings(mappings)

  @parameterized.named_parameters(
      (
          'one_long_key',
          {
              'ab': 'ALPHABETA',
          },
      ),
      (
          'interspersed_long_key',
          {
              'ab': 'ALPHABETA',
              'c': 'CEE',
          },
      ),
      (
          'key_in_own_value',
          {
              'a': 'ALPHa',
          },
      ),
      (
          'key_in_other_value',
          {
              'a': 'ALPHA',
              'b': 'BETa',
          },
      ),
      (
          'second_pre_image',
          {
              'a': 'ALPHA',
              'b': 'ALPHA',
          },
      ),
      (
          'empty_value',
          {
              'a': '',
              'b': 'ALPHA',
          },
      ),
      (
          'value_in_other_value',
          {
              'a': 'ALPHA',
              'b': 'LP',
          },
      ),
  )
  def test_check_mappings_raises_as_expected(self, mappings):
    with self.assertRaises(ValueError):
      unified_tokenizer.check_mappings(mappings)


if __name__ == '__main__':
  absltest.main()
