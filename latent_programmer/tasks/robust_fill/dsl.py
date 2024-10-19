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

"""Defines DSL for the RobustFill domain."""

import abc
import collections
import enum
import functools
import inspect
import re
import string
from typing import TypeVar, List, Dict, Tuple, Any, Optional


ProgramTask = collections.namedtuple('ProgramTask',
                                     ['program', 'inputs', 'outputs'])

# Describes range of possible indices for a character (for SubStr expression).
POSITION = [-100, 100]
# Describes range of possible indices for a regex.
INDEX = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

DELIMITER = '&,.?!@()[]%{}/:;$#"\' '
CHARACTER = string.ascii_letters + string.digits + DELIMITER

BOS = 'BOS'
EOS = 'EOS'


class Type(enum.Enum):
  NUMBER = 1
  WORD = 2
  ALPHANUM = 3
  ALL_CAPS = 4
  PROP_CASE = 5
  LOWER = 6
  DIGIT = 7
  CHAR = 8


class Case(enum.Enum):
  PROPER = 1
  ALL_CAPS = 2
  LOWER = 3


class Boundary(enum.Enum):
  START = 1
  END = 2


def to_python(obj):
  if isinstance(obj, (Type, Case, Boundary)):
    return f'dsl.{obj.__class__.__name__}.{obj.name}'
  elif isinstance(obj, (int, str)):
    return repr(obj)
  else:
    raise ValueError(f'Unhandled object: {obj}')


def to_python_args(*objs):
  return ', '.join(to_python(obj) for obj in objs)


Regex = TypeVar('Regex', Type, str)


def regex_for_type(t):
  """Map types to their regex string."""
  if t == Type.NUMBER:
    return '[0-9]+'
  elif t == Type.WORD:
    return '[A-Za-z]+'
  elif t == Type.ALPHANUM:
    return '[A-Za-z0-9]+'
  elif t == Type.ALL_CAPS:
    return '[A-Z]+'
  elif t == Type.PROP_CASE:
    return '[A-Z][a-z]+'
  elif t == Type.LOWER:
    return '[a-z]+'
  elif t == Type.DIGIT:
    return '[0-9]'
  elif t == Type.CHAR:
    return '[A-Za-z0-9' + ''.join([re.escape(x) for x in DELIMITER]) + ']'
  else:
    raise ValueError('Unsupported type: {}'.format(t))


def match_regex_substr(t, value):
  regex = regex_for_type(t)
  return re.findall(regex, value)


def match_regex_span(r, value):
  if isinstance(r, Type):
    regex = regex_for_type(r)
  else:
    assert (len(r) == 1) and (r in DELIMITER)
    regex = '[' + re.escape(r) + ']'

  return [match.span() for match in re.finditer(regex, value)]


class Base(abc.ABC):
  """Base class for DSL."""

  @abc.abstractmethod
  def __call__(self, value):
    raise NotImplementedError

  @abc.abstractmethod
  def to_string(self):
    raise NotImplementedError

  def __repr__(self):
    return self.to_string()

  @abc.abstractmethod
  def encode(self, token_id_table):
    raise NotImplementedError


class Program(Base):
  pass


class Concat(Program):
  """Concatenation of expressions."""

  def __init__(self, *args):
    self.expressions = args

  def __call__(self, value):
    return ''.join([e(value) for e in self.expressions])

  def to_string(self):
    return ' | '.join([e.to_string() for e in self.expressions])

  def encode(self, token_id_table):
    sub_token_ids = [e.encode(token_id_table) for e in self.expressions]

    return (functools.reduce(lambda a, b: a + b, sub_token_ids)
            + [token_id_table[EOS]])

  def to_python_program(self, name = 'program', version = 1):
    if version == 1:
      code_lines = []
      code_lines.append(f'def {name}(x):')
      code_lines.append('  parts = [')
      for e in self.expressions:
        code_lines.append(f'      {e.to_python_expression()},')
      code_lines.append('  ]')
      code_lines.append("  return ''.join(parts)")
      return '\n'.join(code_lines)
    else:
      raise ValueError(f'Unhandled version: {version}')


class Expression(Base):

  def to_python_expression(self):
    raise NotImplementedError()


class Substring(Expression):
  pass


class Modification(Expression):
  pass


class Compose(Expression):
  """Composition of two modifications or modification and substring."""

  def __init__(self, modification,
               modification_or_substring):
    self.modification = modification
    self.modification_or_substring = modification_or_substring

  def __call__(self, value):
    return self.modification(self.modification_or_substring(value))

  def to_string(self):
    return (self.modification.to_string() + '('
            + self.modification_or_substring.to_string() + ')')

  def encode(self, token_id_table):
    return ([token_id_table[self.__class__]]
            + self.modification.encode(token_id_table)
            + self.modification_or_substring.encode(token_id_table))

  def to_python_expression(self):
    # Every operation's expression has x as the first argument, except ConstStr
    # which is not allowed in Compose.
    return self.modification.to_python_expression().replace(
        '(x', '(' + self.modification_or_substring.to_python_expression())


class ConstStr(Expression):
  """Fixed character."""

  def __init__(self, char):
    self.char = char

  def __call__(self, value):
    return self.char

  def to_string(self):
    return 'Const(' + self.char + ')'

  def encode(self, token_id_table):
    return [token_id_table[self.__class__], token_id_table[self.char]]

  def to_python_expression(self):
    return f'dsl.Const({to_python_args(self.char)})'


class SubStr(Substring):
  """Return substring given indices."""

  def __init__(self, pos1, pos2):
    self.pos1 = pos1
    self.pos2 = pos2

  def __call__(self, value):
    # Positive indices start at 1.
    p1 = self.pos1 - 1 if self.pos1 > 0 else len(value) + self.pos1
    p2 = self.pos2 - 1 if self.pos2 > 0 else len(value) + self.pos2

    if p1 >= p2:  # Handle edge cases.
      return ''
    if p2 == len(value):
      return value[p1:]
    return value[p1:p2 + 1]

  def to_string(self):
    return 'SubStr(' + str(self.pos1) + ', ' + str(self.pos2) + ')'

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.pos1],
        token_id_table[self.pos2],
    ]

  def to_python_expression(self):
    args = to_python_args(self.pos1, self.pos2)
    return f'dsl.SubStr(x, {args})'


class GetSpan(Substring):
  """Return substring given indices of regex matches."""

  def __init__(self, regex1, index1, bound1,
               regex2, index2, bound2):
    self.regex1 = regex1
    self.index1 = index1
    self.bound1 = bound1
    self.regex2 = regex2
    self.index2 = index2
    self.bound2 = bound2

  def _index(self, r, index, bound,
             value):
    """Get index in string of regex match."""
    matches = match_regex_span(r, value)

    # Positive indices start at 1.
    index = index - 1 if index > 0 else len(matches) + index

    if not matches:
      return -1
    if index >= len(matches):  # Handle edge cases.
      return len(matches) - 1
    if index < 0:
      return 0
    span = matches[index]
    return span[0] if bound == Boundary.START else span[1]

  def __call__(self, value):
    p1 = self._index(self.regex1, self.index1, self.bound1, value)
    p2 = self._index(self.regex2, self.index2, self.bound2, value)

    if min(p1, p2) < 0:  # pytype: disable=unsupported-operands
      return ''
    return value[p1:p2]

  def to_string(self):
    return ('GetSpan('
            + ', '.join(map(str, [self.regex1,
                                  self.index1,
                                  self.bound1,
                                  self.regex2,
                                  self.index2,
                                  self.bound2]))
            + ')')

  def encode(self, token_id_table):
    return list(map(lambda x: token_id_table[x],
                    [self.__class__,
                     self.regex1,
                     self.index1,
                     self.bound1,
                     self.regex2,
                     self.index2,
                     self.bound2]))

  def to_python_expression(self):
    args = to_python_args(self.regex1, self.index1, self.bound1,
                          self.regex2, self.index2, self.bound2)
    return f'dsl.GetSpan(x, {args})'


class GetToken(Substring):
  """Get regex match."""

  def __init__(self, regex_type, index):
    self.regex_type = regex_type
    self.index = index

  def __call__(self, value):
    matches = match_regex_substr(self.regex_type, value)

    # Positive indices start at 1.
    index = self.index - 1 if self.index > 0 else len(matches) + self.index
    if not matches:
      return ''
    if index >= len(matches) or index < 0:  # Handle edge cases.
      return ''
    return matches[index]

  def to_string(self):
    return 'GetToken_' + str(self.regex_type) + '_' + str(self.index)

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.regex_type],
        token_id_table[self.index],
    ]

  def to_python_expression(self):
    args = to_python_args(self.regex_type, self.index)
    return f'dsl.GetToken(x, {args})'


class ToCase(Modification):
  """Convert to case."""

  def __init__(self, case):
    self.case = case

  def __call__(self, value):
    if self.case == Case.PROPER:
      return value.capitalize()
    elif self.case == Case.ALL_CAPS:
      return value.upper()
    elif self.case == Case.LOWER:
      return value.lower()
    else:
      raise ValueError('Invalid case: {}'.format(self.case))

  def to_string(self):
    return 'ToCase_' + str(self.case)

  def encode(self, token_id_table):
    return [token_id_table[self.__class__], token_id_table[self.case]]

  def to_python_expression(self):
    args = to_python_args(self.case)
    return f'dsl.ToCase(x, {args})'


class Replace(Modification):
  """Replace delimitors."""

  def __init__(self, delim1, delim2):
    self.delim1 = delim1
    self.delim2 = delim2

  def __call__(self, value):
    return value.replace(self.delim1, self.delim2)

  def to_string(self):
    return 'Replace_' + str(self.delim1) + '_' + str(self.delim2)

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.delim1],
        token_id_table[self.delim2],
    ]

  def to_python_expression(self):
    args = to_python_args(self.delim1, self.delim2)
    return f'dsl.Replace(x, {args})'


class Trim(Modification):
  """Trim whitspace."""

  def __init__(self):
    pass

  def __call__(self, value):
    return value.strip()

  def to_string(self):
    return 'Trim'

  def encode(self, token_id_table):
    return [token_id_table[self.__class__]]

  def to_python_expression(self):
    return 'dsl.Trim(x)'


class GetUpto(Substring):
  """Get substring up to regex match."""

  def __init__(self, regex):
    self.regex = regex

  def __call__(self, value):
    matches = match_regex_span(self.regex, value)

    if not matches:
      return ''
    first = matches[0]
    return value[:first[1]]

  def to_string(self):
    return 'GetUpto_' + str(self.regex)

  def encode(self, token_id_table):
    return [token_id_table[self.__class__], token_id_table[self.regex]]

  def to_python_expression(self):
    args = to_python_args(self.regex)
    return f'dsl.GetUpto(x, {args})'


class GetFrom(Substring):
  """Get substring from regex match."""

  def __init__(self, regex):
    self.regex = regex

  def __call__(self, value):
    matches = match_regex_span(self.regex, value)

    if not matches:
      return ''
    first = matches[0]
    return value[first[1]:]

  def to_string(self):
    return 'GetFrom_' + str(self.regex)

  def encode(self, token_id_table):
    return [token_id_table[self.__class__], token_id_table[self.regex]]

  def to_python_expression(self):
    args = to_python_args(self.regex)
    return f'dsl.GetFrom(x, {args})'


class GetFirst(Modification):
  """Get first occurrences of regex match."""

  def __init__(self, regex_type, index):
    self.regex_type = regex_type
    self.index = index

  def __call__(self, value):
    matches = match_regex_substr(self.regex_type, value)
    if not matches:
      return ''
    if self.index >= len(matches):
      return ''.join(matches)
    return ''.join(matches[:self.index])

  def to_string(self):
    return 'GetFirst_' + str(self.regex_type) + '_' + str(self.index)

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.regex_type],
        token_id_table[self.index],
    ]

  def to_python_expression(self):
    args = to_python_args(self.regex_type, self.index)
    return f'dsl.GetFirst(x, {args})'


class GetAll(Modification):
  """Get all occurrences of regex match."""

  def __init__(self, regex_type):
    self.regex_type = regex_type

  def __call__(self, value):
    return ''.join(match_regex_substr(self.regex_type, value))

  def to_string(self):
    return 'GetAll_' + str(self.regex_type)

  def encode(self, token_id_table):
    return [token_id_table[self.__class__], token_id_table[self.regex_type]]

  def to_python_expression(self):
    args = to_python_args(self.regex_type)
    return f'dsl.GetAll(x, {args})'


# New Functions
# ---------------------------------------------------------------------------


class Substitute(Modification):
  """Replace i-th occurence of regex match with constant."""

  def __init__(self, regex_type, index, char):
    self.regex_type = regex_type
    self.index = index
    self.char = char

  def __call__(self, value):
    matches = match_regex_substr(self.regex_type, value)

    # Positive indices start at 1.
    index = self.index - 1 if self.index > 0 else len(matches) + self.index
    if not matches:
      return value
    if index >= len(matches) or index < 0:  # Handle edge cases.
      return value
    return value.replace(matches[index], self.char, 1)

  def to_string(self):
    return ('Substitute_' + str(self.regex_type) + '_' + str(self.index) + '_'
            + self.char)

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.regex_type],
        token_id_table[self.index],
        token_id_table[self.char],
    ]

  def to_python_expression(self):
    args = to_python_args(self.regex_type, self.index, self.char)
    return f'dsl.Substitute(x, {args})'


class SubstituteAll(Modification):
  """Replace all occurences of regex match with constant."""

  def __init__(self, regex_type, char):
    self.regex_type = regex_type
    self.char = char

  def __call__(self, value):
    matches = match_regex_substr(self.regex_type, value)

    for match in matches:
      value = value.replace(match, self.char, 1)
    return value

  def to_string(self):
    return 'SubstituteAll_' + str(self.regex_type) + '_' + self.char

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.regex_type],
        token_id_table[self.char],
    ]

  def to_python_expression(self):
    args = to_python_args(self.regex_type, self.char)
    return f'dsl.SubstituteAll(x, {args})'


class Remove(Modification):
  """Remove i-th occurence of regex match."""

  def __init__(self, regex_type, index):
    self.regex_type = regex_type
    self.index = index

  def __call__(self, value):
    matches = match_regex_substr(self.regex_type, value)

    # Positive indices start at 1.
    index = self.index - 1 if self.index > 0 else len(matches) + self.index
    if not matches:
      return value
    if index >= len(matches) or index < 0:  # Handle edge cases.
      return value
    return value.replace(matches[index], '', 1)

  def to_string(self):
    return 'Remove_' + str(self.regex_type) + '_' + str(self.index)

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.regex_type],
        token_id_table[self.index],
    ]

  def to_python_expression(self):
    args = to_python_args(self.regex_type, self.index)
    return f'dsl.Remove(x, {args})'


class RemoveAll(Modification):
  """Remove all occurences of regex match."""

  def __init__(self, regex_type):
    self.regex_type = regex_type

  def __call__(self, value):
    matches = match_regex_substr(self.regex_type, value)

    for match in matches:
      value = value.replace(match, '', 1)
    return value

  def to_string(self):
    return 'RemoveAll_' + str(self.regex_type)

  def encode(self, token_id_table):
    return [
        token_id_table[self.__class__],
        token_id_table[self.regex_type],
    ]

  def to_python_expression(self):
    args = to_python_args(self.regex_type)
    return f'dsl.RemoveAll(x, {args})'


def decode_expression(encoding,
                      id_token_table):
  """Decode sequence of token ids to expression (excluding Compose)."""
  cls = id_token_table[encoding[0]]
  return cls(*list(map(lambda x: id_token_table[x], encoding[1:])))


def decode_program(encoding,
                   id_token_table):
  """Decode sequence of token ids into a Concat program."""
  expressions = []

  idx = 0
  while idx < len(encoding) - 1:
    elem = id_token_table[encoding[idx]]
    if elem == Compose:  # Handle Compose separately.
      idx += 1
      modification_elem = id_token_table[encoding[idx]]
      n_args = len(inspect.signature(modification_elem.__init__).parameters)
      modification = decode_expression(encoding[idx:idx+n_args], id_token_table)
      idx += n_args
      modification_or_substring_elem = id_token_table[encoding[idx]]
      n_args = len(
          inspect.signature(modification_or_substring_elem.__init__).parameters)
      modification_or_substring = decode_expression(encoding[idx:idx+n_args],
                                                    id_token_table)
      idx += n_args
      next_e = Compose(modification, modification_or_substring)
    else:
      n_args = len(inspect.signature(elem.__init__).parameters)
      next_e = decode_expression(encoding[idx:idx+n_args], id_token_table)
      idx += n_args
    expressions.append(next_e)

  assert id_token_table[encoding[idx]] == EOS
  return Concat(*expressions)
