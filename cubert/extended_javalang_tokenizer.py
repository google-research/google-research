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

"""Extends javalang.tokenizer to emit comments, end positions, and EOS."""
from typing import Iterable


from javalang import tokenizer as javalang


def tokenize_extended(code,
                      ignore_errors = False
                     ):
  """Same as javalang.tokenize, but using our extended tokenizer.

  Args:
    code: Same as javalang.tokenize.
    ignore_errors: Same as javalang.tokenize.

  Returns:
    The tokens, as javalang.JavaToken objects.
  """
  extended_tokenizer = JavalangTokenizerExtended(code, ignore_errors)
  return extended_tokenizer.tokenize()


class Comment(javalang.JavaToken):
  """A new comment token kind."""

  def __init__(self, is_javadoc, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.is_javadoc = is_javadoc


class Whitespace(javalang.JavaToken):
  """A whitespace token kind."""


class ErrorToken(javalang.JavaToken):
  """An error token for an invalid character."""


class JavalangTokenizerExtended(javalang.JavaTokenizer):
  """Extends the javalang.JavaTokenizer to return additional information.

  * Comments (both block comments and javadoc).
  * Whitespace tokens.
  """

  def consume_whitespace(self):
    """Overrides the superclass to handle final whitespace correctly."""
    match = self.whitespace_consumer.search(self.data, self.i + 1)

    ########################################################################
    # Deviation from javalang:
    #
    # We let even the final whitespace of a file go through the computation of
    # start of line and current line, so that the positioning information for
    # the EOS token is correct.
    if not match:
      i = self.length
    else:
      i = match.start()
    # End of deviation.
    ########################################################################

    start_of_line = self.data.rfind('\n', self.i, i)

    if start_of_line != -1:
      self.start_of_line = start_of_line
      self.current_line += self.data.count('\n', self.i, i)

    self.i = i

  def tokenize(self):
    """Clones the superclass `tokenize` method, but introduces extra tokens.

    Unfortunately the implementation in the superclass isn't modular, so we
    clone the method here. The only changes are in emitting whitespace, rather
    than consuming and discarding it, and emitting comments as separate tokens.

    Beyond the behavior of the superclass `tokenize` method, this extension
    also guarantees that successive tokens abut: the previous one ends where the
    next one begins.

    Yields:
      As per the superclass implementation.
    """
    self.reset()
    self.pre_tokenize()
    while self.i < self.length:
      token_type = None

      c = self.data[self.i]
      c_next = None
      startswith = c

      if self.i + 1 < self.length:
        c_next = self.data[self.i + 1]
        startswith = c + c_next

      if c.isspace():
        ########################################################################
        # Deviation from javalang:
        #
        # We'll be collecting a whitespace span. We record the current start
        # position, because `consume_whitespace` will update the cursors.
        previous_current_line = self.current_line
        previous_i = self.i
        previous_start_of_line = self.start_of_line
        self.consume_whitespace()

        # Now the cursor is advanced. What's between the previous cursor
        # position and the current cursor position is all whitespace. Emit
        # it as a single giant token.
        position = javalang.Position(previous_current_line,
                                     previous_i - previous_start_of_line)
        token = Whitespace(self.data[previous_i:self.i], position, None)
        yield token
        # End of deviation.
        ########################################################################
        continue

      elif startswith in ('//', '/*'):
        ########################################################################
        # Deviation from javalang:
        #
        # We want to emit the comment as a separate token, albeit without
        # modifying the original functionality, which attaches "javadoc"
        # comments to the next token.
        previous_current_line = self.current_line
        previous_i = self.i
        previous_start_of_line = self.start_of_line
        comment = self.read_comment()
        is_javadoc = comment.startswith('/**')
        if is_javadoc:
          self.javadoc = comment

        position = javalang.Position(previous_current_line,
                                     previous_i - previous_start_of_line)
        token = Comment(
            is_javadoc=is_javadoc,
            value=self.data[previous_i:self.i],
            position=position,
            javadoc=None)
        yield token
        # End of deviation.
        ########################################################################
        continue

      elif startswith == '..' and self.try_operator():
        token_type = javalang.Operator

      elif c == '@':
        token_type = javalang.Annotation
        self.j = self.i + 1

      elif c == '.' and c_next and c_next.isdigit():
        token_type = self.read_decimal_float_or_integer()

      elif self.try_separator():
        token_type = javalang.Separator

      elif c in ("'", '"'):
        token_type = javalang.String
        self.read_string()

      elif c in '0123456789':
        token_type = self.read_integer_or_float(c, c_next)

      elif self.is_java_identifier_start(c):
        token_type = self.read_identifier()

      elif self.try_operator():
        token_type = javalang.Operator

      else:
        ########################################################################
        # Deviation from javalang:
        #
        # We emit an error token if there's an unexpected character.
        self.j = self.i + 1
        token_type = ErrorToken
        ########################################################################
        # End of deviation.
        ########################################################################

      position = javalang.Position(self.current_line,
                                   self.i - self.start_of_line)
      token = token_type(self.data[self.i:self.j], position, self.javadoc)
      yield token

      if self.javadoc:
        self.javadoc = None

      self.i = self.j
    ############################################################################
    # Deviation from javalang:
    #
    # We got here because `self.i` reached the end of the file. Emit an end of
    # input.  The javalang tokenizer does not emit `EndOfInput`.
    position = javalang.Position(self.current_line, self.i - self.start_of_line)
    token = javalang.EndOfInput('', position, self.javadoc)
    yield token
    ############################################################################
    # End of deviation.
    ############################################################################
