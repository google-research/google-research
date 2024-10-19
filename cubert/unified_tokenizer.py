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

"""Cross-language tokenization library."""
import dataclasses
import enum
import token as python_token
import tokenize
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Text
from typing import Tuple

from absl import logging
import regex  # Using instead of `re` because it handles Unicode classes.
import six


# Quote string for special tokens.
SPECIAL_QUOTE = '___'


def quote_special(content):
  return '{q}{t}{q}'.format(q=SPECIAL_QUOTE, t=content)


# Log level of pedantic messages.
_PEDANTIC = 5

# Punctuation for tokenization.
SENTINEL = '^'
SENTINEL_ESCAPE = 'CARET'


@enum.unique
class TokenKind(enum.Enum):
  """The kind of language-agnostic tokens."""
  NONE = 0  # Sadly, Python2 doesn't support enum.auto()
  PUNCTUATION = 1
  KEYWORD = 2
  IDENTIFIER = 3
  STRING = 4
  COMMENT = 5
  NEWLINE = 6
  EOS = 7
  ERROR = 8
  NUMBER = 9
  WHITESPACE = 10


NEWLINE = quote_special(TokenKind.NEWLINE.name)


@dataclasses.dataclass(frozen=True)
class Position():
  line: int
  column: int

  def __lt__(self, other):
    return (self.line, self.column) < (other.line, other.column)


@dataclasses.dataclass(frozen=True)
class TokenMetadata():
  """Metadata about abstract tokens.

  Attributes:
    start: The position of the first character of the token.
    end: The position right after the last character of the token. The line is
      the same as the line of the last character and the column is the
      column immediately following the last column of the token.
  """
  start: Optional[Position] = None
  end: Optional[Position] = None


@dataclasses.dataclass(frozen=True)
class AbstractToken():
  spelling: str
  kind: TokenKind
  metadata: TokenMetadata


@dataclasses.dataclass(frozen=True)
class AbstractMultiToken():
  # We force `spellings` to be a concrete `Tuple`, to simplify equality checks
  # and hashing. Otherwise, `spellings=[1, 2, 3]` and `spellings=(1, 2, 3)`
  # would result in different multi-tokens.
  spellings: Tuple[str]
  kind: TokenKind
  metadata: TokenMetadata


def multi_token_from_token(token):
  return AbstractMultiToken(spellings=(token.spelling,),
                            kind=token.kind,
                            metadata=token.metadata)


# TODO(maniatis): Add a test for this one, and migrate other copies to use
# the same implementation.
def fill_range_with_whitespace(start,
                               end):
  """Yields primitive whitespace/newline tokens to fill a text range.

  We translate multi-line whitespace into single-line whitespace and newlines,
  in a *destructive* canonical fashion. Only space preceding a non-whitespace
  token is preserved. Lines with only whitespace are replaced by a single
  newline token.

  Args:
    start: The beginning of the range.
    end: The end (exclusive) of the range.

  Yields:
    WHITESPACE and NEWLINE abstract tokens.

  Raises:
    ValueError: if `start` does not precede `end`.
  """
  current_line = start.line
  current_column = start.column
  end_column = end.column
  end_line = end.line
  if (current_line, current_column) >= (end_line, end_column):
    raise ValueError('`start` must precede `end`, but we received start %s '
                     'and end %s.' % (start, end))

  while current_line < end_line:
    next_line = current_line + 1
    yield AbstractToken(
        NEWLINE,
        TokenKind.NEWLINE,
        TokenMetadata(
            # A NEWLINE starts at the colum where it occurs and ends
            # at the first character of the next line.
            start=Position(line=current_line, column=current_column),
            end=Position(line=next_line, column=0)))
    current_column = 0
    current_line = next_line

  # At this point, we have consumed all newlines. Add any remaining
  # space until the next, non-whitespace token.
  number_of_final_spaces = end_column - current_column
  if number_of_final_spaces:
    # Note that we canonicalize all column differences as space characters.
    # This, for example, will discard any '\t' characters and replace them
    # with ' '.
    yield AbstractToken(
        ' ' * number_of_final_spaces, TokenKind.WHITESPACE,
        TokenMetadata(
            start=Position(line=current_line, column=current_column),
            end=Position(line=current_line, column=end_column)))


_KINDS_TO_SPLIT_LIKE_WHITESPACE = (
    TokenKind.COMMENT,
    TokenKind.STRING,
    TokenKind.WHITESPACE,
    TokenKind.ERROR,
)
_KINDS_TO_SPLIT_BY_LENGTH = (
    TokenKind.COMMENT,
    TokenKind.STRING,
    TokenKind.NUMBER,
    TokenKind.IDENTIFIER,
    TokenKind.WHITESPACE,
    TokenKind.ERROR,
)

_UPPERCASE = r'\p{Lu}'
_TITLECASE = r'\p{Lt}'

# Here we abuse the term "lowercase", by using it to refer to anything that
# doesn't cause a camel/Pascal case split. For Python, for example, this
# includes Unicode category Nd ("decimal numbers") and Nl ("number letters").
# We assume that before these regular expressions are applied, any
# characters that don't fall into a legal "other" category have been checked.
_LOWERCASE = r'[^\p{Lu}\p{Lt}]'

# In what follows, L, U, and T will be replaced with _LOWERCASE, _UPPERCASE
# and _TITLECASE later.
_CAMEL_AFTER_SNAKE_EXPRESSIONS = [
    # Beginning lowercase.
    r'^{L}+',
    # A single titlecase followed by 0 or more lowercase.
    r'{T}{L}*',
    # Single uppercase followed by multiple lowercase.
    r'{U}{L}+',
    # Multiple uppercase ending right before a titlecase.
    r'{U}+(?={T})',
    # Multiple uppercase ending right before an uppercase followed by lowercase.
    r'{U}+(?={U}{L})',
    # Multiple uppercase to the end.
    r'{U}+$',
]
_CAMEL_AFTER_SNAKE_EXPRESSION = '|'.join(_CAMEL_AFTER_SNAKE_EXPRESSIONS).format(
    L=_LOWERCASE,
    T=_TITLECASE,
    U=_UPPERCASE)
_CAMEL_RE = regex.compile(_CAMEL_AFTER_SNAKE_EXPRESSION, regex.U)  # pytype: disable=module-attr


class StateType(enum.IntEnum):
  INITIAL_STATE = 0
  UPPERCASE_STATE = 1
  LOWERCASE_STATE = 2
  NUMBER_STATE = 3
  SPECIAL_STATE = 4


def code_to_tokens(code):
  """Convert Python source code to list of tokens.

  Removes all trailing whitespace and then tokenizes the text as if it were
  Python source code. Tokens are 5-tuples as used by the built-in tokenize
  module.

  Args:
    code: string containing python source code

  Returns:
    The code represented as a string of packed tokens separated by spaces.

  Raises:
    tokenize.TokenError: When a multi-line token is incomplete. This is
      generated by `tokenize.generate_tokens`.
    IndentationError: When the source code is incorrectly indented. This is
      generated by `tokenize.generate_tokens`.
  """
  token_tuples = list(tokenize.generate_tokens(
      six.StringIO(code.rstrip()).readline))
  logging.vlog(5, 'Code `%s` was tokenized to token tuples `%s`.', code,
               token_tuples)

  # Now we get rid of an extraneous trailing newline token, if it has been
  # produced. This is a difference in the behavior of generate_tokens between
  # Python 2 and Python 3.
  if six.PY3:
    if len(token_tuples) > 1:
      if token_tuples[-2][0] == python_token.NEWLINE:
        del token_tuples[-2]
        logging.vlog(5, 'Tokenization for `%s` was sanitized. Now token tuples '
                     'are `%s`.', code, token_tuples)
    # Another similar failure mode is if the final tokens are DEDENT, there may
    # be an extraneous newline before them.
    if len(token_tuples) > 2:
      current = len(token_tuples) - 2  # Right before ENDMARKER.
      while current and token_tuples[current][0] == tokenize.DEDENT:
        current -= 1
      if current and token_tuples[current][0] == tokenize.NEWLINE:
        del token_tuples[current]
        logging.vlog(5, 'Tokenization for `%s` was sanitized to remove '
                     'trailing newline after DEDENTs. Now token tuples are '
                     '`%s`.', code, token_tuples)

  return token_tuples


def code_to_tokens_simple_lossless(code):
  r"""Convert python source code to list of tokens.

  This is a simple version using spacing and different classes of characters to
  tokenize a string.

  A sentence will be split at "|" in the following patterns:
    upper | upper lower
    upper | number
    upper | special
    lower | upper
    lower | number
    lower | special
    number | upper
    number | lower
    number | special
    special | upper
    special | lower
    special | number

  In addition to splits caused by the type changes above, the code is also split
  at whitespace. However, a sequence of spaces or tabs will not be split unless
  its length is longer than 20.

  For example: "12345  \n\n678" -> ["12345", "  ", "\n", "\n", "678"]

  We do not split sequences of spaces/tabs to avoid long sequences of single
  " " or "\t" tokens caused by deep indentation.

  This tokenizer uses a finite state machine. The definition of the states is in
  the StateType class.

  Args:
    code: String containing Python source code.

  Returns:
    The code represented as a string of tokens separated by spaces.
    For example, "foo  ,1" -> ["foo", "  ", ",", "1"]
  """
  # normal state transitions that will result in splitting
  normal_transitions = [
      (StateType.UPPERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.UPPERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.LOWERCASE_STATE, StateType.UPPERCASE_STATE),
      (StateType.LOWERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.LOWERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.NUMBER_STATE, StateType.UPPERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.LOWERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.SPECIAL_STATE),
      (StateType.SPECIAL_STATE, StateType.UPPERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.LOWERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.NUMBER_STATE)]
  # output, state
  tokens = []
  state = StateType.INITIAL_STATE
  next_state = None
  memory = []
  for i, inputchar in enumerate(code):
    if inputchar.isupper():
      next_state = StateType.UPPERCASE_STATE
    elif inputchar.islower():
      next_state = StateType.LOWERCASE_STATE
    elif inputchar.isdigit():
      next_state = StateType.NUMBER_STATE
    else:
      next_state = StateType.SPECIAL_STATE

    # splitting cases
    if (state, next_state) in normal_transitions:
      tokens.append(''.join(memory))
      memory = []
    elif (state, next_state) == (StateType.UPPERCASE_STATE,
                                 StateType.LOWERCASE_STATE) and len(memory) > 1:
      tokens.append(''.join(memory[:-1]))
      memory = [memory[-1]]
    elif (state, next_state) == (StateType.SPECIAL_STATE,
                                 StateType.SPECIAL_STATE):
      if inputchar in [' ', '\t'] and inputchar == code[i-1]:
        if len(memory) >= 20:
          tokens.append(''.join(memory))
          memory = []
      elif inputchar.isspace() or code[i-1].isspace():
        tokens.append(''.join(memory))
        memory = []

    # put inputchar into memory, always
    memory.append(inputchar)
    state = next_state
  if memory:
    tokens.append(''.join(memory))
  return tokens


def subtokenize_identifier(identifier):
  """Splits an identifier assuming camel/pascal/snake case conventions.

  This doesn't attempt to classify the identifier as one of snake case/camel/
  pascal, etc. It just applies all possible splits in the order snake case,
  Pascal, camel.

  This doesn't check whether an identifier is a legal identifier for some
  language. It is assumed that the caller has already decided that.

  For Unicode characters in identifiers, we define splitting conventions as
  follows:

  - Snake-case is only defined in terms of the ASCII underscore (U+005F). Other
    characters that may look like an underscore do not introduce a snake-case
    component.
  - For the purpose of Pascal and camel cases, we categorize only the Lu Unicode
    category as uppercase characters, with the exception of the Lt (titlecase)
    character category. Lt characters are treated as a sequence of an uppercase
    character followed by a lowercase character and, as such, may only appear
    in the beginning of a Pascal-case component, but not as an all-uppercase
    component. As an example, if U, L, T are uppercase, lowercase, and titlecase
    characters as defined above (i.e., members of Lu, everything else, or Lt
    categories, respectively), UUUT would be split as UUU and T, ULTL would be
    split as UL and TL, LTL would be split as L and TL, etc.

  Args:
    identifier: A non-empty string, purporting to be an identifier. Assumes its
      validity as an identifier in a given language has already been established
      by the caller.

  Returns:
    A list of substrings of `identifier`. Joining the substrings should return
      the original `identifier` exactly.

  Raises:
    ValueError: if `identifier` is not a legal identifier string.
  """
  snake_splits = identifier.split('_')
  snake_components = []  # type: List[Text]
  current_snake_separator = []  # type: List[Text]
  for snake_split in snake_splits:
    if snake_split:
      snake_components.append(''.join(current_snake_separator))
      current_snake_separator = []
      snake_components.append(snake_split)
    current_snake_separator.append('_')
  # Emit the final separator, but discard the most recent underscore added to
  # it. It should have at least one.
  current_snake_separator.pop()
  if current_snake_separator:
    snake_components.append(''.join(current_snake_separator))

  # Now we want to do camel-case splitting for each non-underscore snake
  # component.
  logging.vlog(_PEDANTIC, 'Split %r into snake case: %r', identifier,
               snake_components)
  all_components = []  # type: List[Text]
  for snake_component in snake_components:
    if '_' in snake_component:
      all_components.append(snake_component)
    else:
      unicodified_snake_component = six.ensure_text(snake_component)
      camel_components = _CAMEL_RE.findall(unicodified_snake_component)
      logging.vlog(_PEDANTIC, 'Split snake component %r into %r components.',
                   unicodified_snake_component, camel_components)
      all_components.extend(camel_components)

  # Finally, we want to combine the underscore components with the component
  # immediately preceding them.
  non_underscore_component = ''
  final_components = []  # type: List[Text]
  for component in all_components:
    if '_' in component:
      # Found an underscore component. Combine it with the previous non-
      # underscore component (if any), emit it, and clear the remembered
      # non-underscore component.
      combined_component = non_underscore_component + component
      final_components.append(combined_component)
      non_underscore_component = ''
    else:
      # This is a non-underscore component.

      if non_underscore_component:
        # We've found two consecutive non-underscore components. Emit the
        # previous one, since it won't be combined with any underscores.
        final_components.append(non_underscore_component)

      # Remember the current non-underscore component, in case we need to
      # combine it with a following underscore.
      non_underscore_component = component
  # We may have collected the final non-underscore component and it wasn't
  # followed by underscores. Just emit it.
  if non_underscore_component:
    final_components.append(non_underscore_component)

  assert (six.ensure_text(
      ''.join(final_components)) == six.ensure_text(identifier)), (
          'Ended up with different identifier when joinining components %r '
          'into combined %r.' % (final_components, identifier))
  return final_components


def sanitize(t, mappings):
  r"""Sanitizes a token to remove "dangerous" characters, like \n and \r."""
  final = t
  for original, sanitized in mappings.items():
    assert len(original) == 1
    final = final.replace(original, sanitized)
  return final


def unsanitize(t, mappings):
  """Unsanitizes a previously sanitized token."""
  final = t
  for original, sanitized in mappings.items():
    assert len(original) == 1
    final = final.replace(sanitized, original)
  return final


def split_long_token(token_string,
                     max_output_token_length):
  """Splits a token losslessly to some maximum length per component.

  A long token is split into multiple tokens. For instance, `'bcd'` with
  `max_output_token_length=2` will become `['bc', 'd']`. No sentinel or other
  split mark is added at this stage.

  A token is assumed to be non-empty.

  Args:
    token_string: The token.
    max_output_token_length: Maximum length of an output token.

  Returns:
    List of split tokens.

  Raises:
    ValueError: if `token` is empty.
  """
  if not token_string:
    raise ValueError('Expected %r to be non-empty' % token_string)

  whole_token_length = len(token_string)
  remainder_length = whole_token_length % max_output_token_length
  even_parts = list(
      map(
          # ...join together...
          ''.join,
          zip(
              # `max_output_token_length` copies of the iterator of
              # whole_token's characters. zip will draw from the same iterator
              # and return `max_output_token_length` tuples of characters from
              # `whole_token`.
              *[iter(token_string)] * max_output_token_length)))
  remainder_part = ([token_string[-remainder_length:]]
                    if remainder_length else [])
  split_token = even_parts + remainder_part
  assert split_token, ('while wrapping >>%s<< into >%r<' %
                       (token_string, split_token))
  assert all([
      len(t) <= max_output_token_length for t in split_token
  ]), ('Got split_token >>>%r<<<, which contains tokens longer than %d.' %
       (split_token, max_output_token_length))
  return split_token


def _agnostic_tokens_to_lists_of_token_lists(
    agnostic_tokens
):
  """Turns each token into a singleton token list, keeping token kinds."""
  return [multi_token_from_token(a) for a in agnostic_tokens]


def _subtokenize_identifiers_heuristically(
    token_lists
):
  """Subtokenizes only identifiers in a list of token lists.

  This assumes that every subtoken list is still a singleton.

  Args:
    token_lists: A list of labelled tokens. Each token is represented as a
      (still) singleton list of subtokens.

  Returns:
    A list of token lists, of which the identifiers are split heuristically.
  """
  with_split_identifiers: List[AbstractMultiToken] = []
  for multi_token in token_lists:
    # spelling_list had better still be a singleton.
    assert len(multi_token.spellings) == 1, (
        'Expected %r to be a singleton, but it is not.' % multi_token)
    if multi_token.kind is TokenKind.IDENTIFIER:
      subtokenized = dataclasses.replace(  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
          multi_token,
          spellings=subtokenize_identifier(multi_token.spellings[0]))
      with_split_identifiers.append(subtokenized)
    else:
      with_split_identifiers.append(multi_token)
  return with_split_identifiers


def _subtokenize_strings_heuristically(
    token_lists
):
  """Splits STRING, COMMENT, WHITESPACE tokens like text.

  Args:
    token_lists: List of subtoken lists, of which only those of kind IDENTIFIER
      are allowed not to be singletons.

  Returns:
    A list of token lists, of which IDENTIFIER, STRING, NUMBER, COMMENT tokens
      are now split heuristically.
  """
  with_heuristically_split_text: List[AbstractMultiToken] = []
  for multi_token in token_lists:
    if multi_token.kind in _KINDS_TO_SPLIT_LIKE_WHITESPACE:
      assert len(multi_token.spellings) == 1, (
          'Expected %r to be a singleton, but it is not.' % multi_token)
      subtokenized = dataclasses.replace(  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
          multi_token,
          spellings=code_to_tokens_simple_lossless(multi_token.spellings[0]))
      with_heuristically_split_text.append(subtokenized)
    else:
      with_heuristically_split_text.append(multi_token)
  return with_heuristically_split_text


def _shorten_subtokens(
    token_lists,
    max_output_token_length,
):
  """Further subtokenizes any subtokens that are too long.

  At this point, we're done with all heuristic splitting. Now split what's left
  by length if need be. We don't do anything about keywords or other
  punctuation.

  Args:
    token_lists: List of subtoken lists, of which only those of kinds
      IDENTIFIER, NUMBER, STRING, COMMENT may have been subtokenized.
    max_output_token_length: The max character length for each subtoken of
      the subtokenizable kinds.

  Returns:
    Subtokenized tokens up to a maximum per-subtoken length.
  """
  shortened_subtokens: List[AbstractMultiToken] = []
  for multi_token in token_lists:
    if multi_token.kind in _KINDS_TO_SPLIT_BY_LENGTH:
      shortened_spelling_list: List[str] = []
      for spelling in multi_token.spellings:
        shortened_spelling_list.extend(
            split_long_token(spelling, max_output_token_length))
      shortened_subtokens.append(
          dataclasses.replace(
              multi_token, spellings=tuple(shortened_spelling_list)))
    else:
      shortened_subtokens.append(multi_token)
  return shortened_subtokens


def split_agnostic_tokens(
    agnostic_tokens,
    max_output_token_length,
):
  """Splits each language-agnostic token according to its kind.

  Args:
    agnostic_tokens: The language-agnostic tokens to subtokenize. These are
      pairs of spelling and generic token kind. No subtokenization has been
      done; the tokens are as the language-specific lexer produced them.
    max_output_token_length: The target maximum output token length.

  Returns:
    A list of subtoken lists, with their associated token kind.
  """
  # Prepare for subtokenization.
  agnostic_token_lists = _agnostic_tokens_to_lists_of_token_lists(
      agnostic_tokens)
  # Perform heuristic subtokenizations.
  with_identifiers_heuristically_split = _subtokenize_identifiers_heuristically(
      agnostic_token_lists)
  with_string_tokens_heuristically_split = _subtokenize_strings_heuristically(
      with_identifiers_heuristically_split)
  # Shorten resulting subtokens by length.
  shortened_subtokens = _shorten_subtokens(
      with_string_tokens_heuristically_split, max_output_token_length)

  return shortened_subtokens


def sanitize_subtoken_lists(
    subtoken_lists,
    sanitization_mapping,
    sentinel):
  """Sanitizes lists of subtoken lists, adding sentinels.

  Args:
    subtoken_lists: A list of multi-tokens. Cannot be empty or contain empty
      sublists.
    sanitization_mapping: A mapping from sensitive characters to replacement
      strings. It is assumed to have been checked by `check_mappings`.
    sentinel: The sentinel character. It is expected to be one of the keys
      in `sanitization_mapping`.

  Returns:
    A list of multi-tokens.

  Raises:
    ValueError: If one of the input sublists is empty, or the entire input
      is empty, or the sentinel is not one of the unsanitary characters.
  """
  if not subtoken_lists:
    raise ValueError('Received empty input %r but expected it to be non '
                     'empty' % subtoken_lists)
  if sentinel not in sanitization_mapping:
    raise ValueError('Sentinel %r should be in the sanitization map %r '
                     'but is not.' % (sentinel, sanitization_mapping))

  sanitized_lists = []
  for multi_token in subtoken_lists:
    spellings = multi_token.spellings
    if not spellings:
      raise ValueError('Received empty multi-token %r but expected no empty '
                       'ones' % multi_token)
    sanitized_spellings = [
        sanitize(t, sanitization_mapping)
        for t in spellings
    ]

    # Add the sentinel to all subtokens except the last one.
    with_sentinel = ([t + sentinel for t in sanitized_spellings[:-1]] +
                     [sanitized_spellings[-1]])

    sanitized_lists.append(
        dataclasses.replace(multi_token, spellings=with_sentinel))  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
  return sanitized_lists


def flatten_subtoken_lists(
    subtoken_lists):
  """Flattens lists of subtoken lists.

  Args:
    subtoken_lists: A list of subtoken lists, one list per initial language
      token. Cannot be empty or contain empty sublits.

  Returns:
    A list of flattened subtokens representing the entire original sequence.

  Raises:
    ValueError: If the input is empty.
  """
  if not subtoken_lists:
    raise ValueError('Received empty input %r but expected it to be non '
                     'empty' % (subtoken_lists,))
  spellings = (t.spellings for t in subtoken_lists)
  subtokens = sum(spellings, [])

  return subtokens


def flatten_and_sanitize_subtoken_lists(
    subtoken_lists,
    sanitization_mapping,
    sentinel):
  """Sanitizes and then flattens lists of subtoken lists, adding sentinels.

  Args:
    subtoken_lists: A list of multi-tokens, one per initial language
      token. Cannot be empty or contain empty sublits.
    sanitization_mapping: A mapping from sensitive characters to replacement
      strings. It is assumed to have been checked by `check_mappings`.
    sentinel: The sentinel character. It is expected to be one of the keys
      in `sanitization_mapping`.

  Returns:
    A list of flattened subtokens representing the entire original sequence.

  Raises:
    ValueError: If one of the input sublists is empty, or the entire input
      is empty, or the sentinel is not one of the unsanitary characters.
  """
  sanitized = sanitize_subtoken_lists(subtoken_lists, sanitization_mapping,
                                      sentinel)
  flattened = flatten_subtoken_lists(sanitized)
  return flattened


def reconstitute_full_unsanitary_tokens(
    split_token_list,
    sanitization_mapping,
    sentinel):
  """Unsplits tokens previously subtokenized and flattened.

  It assumes this is the output of `split_agnostic_tokens`, followed by
  `sanitize_subtoken_lists` and `flatten_subtoken_lists`.

  Split tokens are joined together.  `['bc^', 'd']` will become
  `'bcd'`, where '^' is `SENTINEL` that indicates where joining occurs.

  Args:
    split_token_list: List of split tokens.
    sanitization_mapping: A mapping from sensitive characters to replacement
      strings. It is assumed to have been checked by `check_mappings`.
    sentinel: The sentinel character. It is expected to be one of the keys
      in `sanitization_mapping`.

  Returns:
    Sequence of whole tokens.

  Raises:
    ValueError: if the sentinel character appears in any position other than
      the sentinel position, or if any of the unsanitary characters (as per
      the `sanitization_mapping`) appear anywhere, or if a subtoken is empty,
      or the sentinel is not one of the unsanitary characters.
  """
  if not split_token_list:
    raise ValueError('Received empty input %r but expected it to be non '
                     'empty.' % split_token_list)
  if sentinel not in sanitization_mapping:
    raise ValueError('Sentinel %r should be in the sanitization map %r '
                     'but is not.' % (sentinel, sanitization_mapping))

  whole_token_list = []  # type: List[Text]
  pending_split_tokens = []  # type: List[Text]
  for t in split_token_list:
    if not t:
      raise ValueError('Must have non-empty subtokens, but found %r in %r.' % (
          t, split_token_list))
    if t[-1] == sentinel:
      # Remove sentinel and accumulate until the final one appears.
      pending_split_tokens.append(t[:-1])
    else:
      # It is a final token, so combine everything accumulated into one.
      pending_split_tokens.append(t)
      whole_token = ''.join(pending_split_tokens)
      whole_token_list.append(whole_token)
      pending_split_tokens = []
  # We should have nothing pending.
  if pending_split_tokens:
    raise ValueError('After scanning all subtokens %r, there still is some '
                     'unjoined content: %r' %
                     (split_token_list, pending_split_tokens))

  # At this point we have whole tokens that contain sanitized characters. First
  # we'll see if they are dirty, and then unsanitize them into their original
  # form.
  unsanitary_characters = sanitization_mapping.keys()
  for whole_token in whole_token_list:
    for unsanitary_character in unsanitary_characters:
      if unsanitary_character in whole_token:
        raise ValueError('Reconstructed whole token %r seems to contain a '
                         'character %r that should have been sanitized '
                         'already.' % (whole_token, unsanitary_character))
  # Unsanitize.
  unsanitized_whole_tokens = [
      unsanitize(t, sanitization_mapping) for t in whole_token_list
  ]

  return unsanitized_whole_tokens


def check_mappings(mappings):
  """Checks the correctness of character-to-string sanitization mappings.

  This ensures that all keys are single characters and that no value contains
  any of the keys or other values.

  Args:
    mappings: A mapping from characters to strings.

  Raises:
    ValueError: If a key has length different from 1 or if a key appears in any
      value or if a value is a substring of another value, or if any value is
      empty or non-unique.
  """
  for key in mappings:
    if len(key) != 1:
      raise ValueError('Expecting length-1 strings as keys in mappings, but '
                       'got key %r in mappings %r.' % (key, mappings))

  values = mappings.values()

  if len(values) != len(set(values)):
    raise ValueError('There seem to be some duplicate values in %r, but they '
                     'are expected to be unique.' % mappings)

  if any([not value for value in values]):
    raise ValueError('An empty value found in %r, but no empty values are '
                     'allowed.' % mappings)

  for value in values:
    for other_value in values:
      if value != other_value and value in other_value:
        raise ValueError('Value %r is a substring of %r, but no value may '
                         'be a substring of another.' % (value, other_value))

    for key in mappings:
      if key in value:
        raise ValueError('No key may appear in one of the mapping values, but '
                         'found key %r in value %r, both of which appear in '
                         'the mappings %r.' % (key, value, mappings))


def subtokenize_agnostic_tokens_in_place(
    agnostic_tokens,
    max_output_token_length,
    sanitization_mapping,
    sentinel,
):
  """Subtokenizes language-agnostic tokens, discarding their kind in the end.

  Args:
    agnostic_tokens: The language-agnostic tokens to subtokenize. These are
      pairs of spelling and generic token kind. No subtokenization has been
      done; the tokens are as the language-specific lexer produced them.
    max_output_token_length: The target maximum output token length.
    sanitization_mapping: A mapping from sensitive characters to replacement
      strings. It is assumed to have been checked by `check_mappings`.
    sentinel: The sentinel character. It is expected to be one of the keys
      in `sanitization_mapping`.

  Returns:
    A list of subtoken lists, one per original agnostic token.
  """
  labelled_subtokenized = split_agnostic_tokens(agnostic_tokens,
                                                max_output_token_length)

  subtoken_lists = sanitize_subtoken_lists(labelled_subtokenized,
                                           sanitization_mapping,
                                           sentinel)
  return subtoken_lists
