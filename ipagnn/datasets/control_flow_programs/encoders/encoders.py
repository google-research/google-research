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

"""Encodes ControlFlowPrograms examples as lists of token ids."""

import ast

from absl import logging  # pylint: disable=unused-import

import tensorflow_datasets as tfds
from ipagnn.datasets.control_flow_programs import control_flow_programs_version
from ipagnn.datasets.control_flow_programs.program_generators import constants


def get_program_encoder(program_generator_config):
  """Gets a TextEncoder for the programs that the generator config specifies."""
  if program_generator_config.encoder_name == "simple":
    if control_flow_programs_version.at_least("0.0.42"):
      mod_ops1 = [
          "= %", "> %", ">= %", "< %", "<= %",
      ]
      mod_ops2 = [
          "if > %", "if < %", "while > %", "while < %",
          "if >= %", "if <= %", "while >= %", "while <= %",
      ]
    else:
      mod_ops1 = []
      mod_ops2 = []
    return SimplePythonSourceEncoder(
        base=program_generator_config.base,
        num_digits=program_generator_config.num_digits,
        ops=list(program_generator_config.ops) + [
            "=", ">", ">=", "<", "<=",
        ] + mod_ops1 + [
            "if", "else", "while",
            "if >", "if <", "while >", "while <",
            "if >=", "if <=", "while >=", "while <=",
        ] + mod_ops2 + [
            "pass", "continue", "break",
        ],
        num_variables=10,  # TODO(dbieber): num_variables is hardcoded.
    )
  elif program_generator_config.encoder_name == "text":
    return TextSourceEncoder()
  else:
    raise ValueError("Unexpected encoder_name",
                     program_generator_config.encoder_name)


class SimplePythonSourceEncoder(tfds.deprecated.text.text_encoder.TextEncoder):
  """Encoder for encoding simple Python programs from source.

  Note this specifically handles ControlFlowProgram programs, not arbitrary
  Python programs.

  In the code representation, we'll have statements like "while var > 0:",
  whereas in the trace representation and the CFG representation, we'll see
  statements like (var > 0).

  This is because the CFG's statements don't include control flow primitives,
  such as {if, while, continue, break}, which are instead included through the
  structure of the graph, not its contents.

  In particular, it expects each line of the program source to be one of these:
    - var op value
    - (var > 0)  # Indicates `vBranch = var > 0` was run by the interpreter.
    - while var > 0:
    - if var > {threshold}:
  The variables are required to be of the form v(Number).

  Attributes:
    base: Individual digits range from 0 to (base - 1).
    num_digits: Each integer is encoded as a num_digits digit number.
    ops: The supported ops (strings)
    ops_index: For each op, its index in the self.ops list.
    num_variables: The maximum number of variables supported.
  """

  def __init__(self, base, num_digits, ops, num_variables):
    """Constructs ControlFlowProgramsEncoder."""
    self.base = base
    self.num_digits = num_digits
    self.ops = ops
    self.ops_index = {
        op: index
        for index, op in enumerate(self.ops)
    }
    self.num_variables = num_variables

  def encode(self, python_source):
    token_ids = []
    for line in python_source.split("\n"):
      if line.startswith("(") and line.endswith(")"):
        # This is a condition. It was originally part of a statement like:
        # "while x > 3:" or "if y < 2:".
        # To us, it should symbolize vBranch = `line`, and vBranch will be used
        # immediately in the next line.
        # These lines are inserted in python_interpreter during interpretation
        # of while or if statements.
        original_line = line
        line = line[1:-1]
        line = line.replace("(", "").replace(")", "")
        assert ast.dump(ast.parse(line)) == ast.dump(ast.parse(original_line))
      if not line:  # Skip blank lines.
        continue
      indent = int((len(line) - len(line.lstrip())) / constants.INDENT_SPACES)
      if "while" in line or "if" in line:
        if "%" in line:
          # while v3 % 10 > 0:
          control_op, var, mod_op, unused_mod_operand, cond_op, operand = (
              line.rstrip(":").split())
          operand = int(operand)
          assert control_op in ("while", "if")
          assert cond_op in (">", "<", ">=", "<=")
          op = f"{control_op} {cond_op} {mod_op}"
          statement = (
              [self.indent_token_id(indent),
               self.op_token_id(op),
               self.var_token_id(var)]
              + self.operand_token_ids(operand)
          )

        else:
          # while v3 > 0:
          control_op, var, cond_op, operand = line.rstrip(":").split()
          operand = int(operand)
          assert control_op in ("while", "if")
          assert cond_op in (">", "<", ">=", "<=")
          op = f"{control_op} {cond_op}"
          statement = (
              [self.indent_token_id(indent),
               self.op_token_id(op),
               self.var_token_id(var)]
              + self.operand_token_ids(operand)
          )
      elif "else" in line:
        op = "else"
        statement = (
            [self.indent_token_id(indent),
             self.op_token_id(op),
             self.op_token_id(op)]
            + self.operand_token_ids(0)
        )
      elif "pass" in line:
        op = "pass"
        statement = (
            [self.indent_token_id(indent),
             self.op_token_id(op),
             self.op_token_id(op)]
            + self.operand_token_ids(0)
        )
      elif "continue" in line:
        op = "continue"
        statement = (
            [self.indent_token_id(indent),
             self.op_token_id(op),
             self.op_token_id(op)]
            + self.operand_token_ids(0)
        )
      elif "break" in line:
        op = "break"
        statement = (
            [self.indent_token_id(indent),
             self.op_token_id(op),
             self.op_token_id(op)]
            + self.operand_token_ids(0)
        )
      else:
        # Handles both the `var op operand` case and the `(v1 > 0)` case.
        # The former is a regular statement. The latter is used as a condition
        # in a control flow statement.
        # We cannot distinguish if statements from while statements here.
        if "%" in line:
          # v3 % 10 > 0
          var, mod_op, unused_mod_operand, cond_op, operand = line.split()
          op = f"{cond_op} {mod_op}"
        else:
          # v3 > 0
          var, op, operand = line.split()
        if var == "_":
          # This is a placeholder statement.
          placeholder = self.placeholder_token_id()
          statement = (
              [self.indent_token_id(indent),
               placeholder,
               placeholder]
              + self.operand_token_ids(0)
          )
        else:
          op_token_id = self.op_token_id(op)
          if operand.startswith("v"):
            # If the operand is a variable, pad the var_token_id.
            operand_token_ids = (
                [self.var_token_id(operand)]
                + [self.var_padding_token_id()] * (self.num_digits - 1)
            )
          else:
            # The operand is a number.
            operand_token_ids = self.operand_token_ids(operand)
          statement = [self.indent_token_id(indent),
                       op_token_id,
                       self.var_token_id(var)] + operand_token_ids
      token_ids.extend(statement)
    return token_ids

  def decode(self, token_ids):
    return self.decode_to_python_source(token_ids)

  def decode_to_string(self, token_ids):
    """Produces a human-readable representation of the token list."""
    return self.decode_to_python_source(token_ids)

  def decode_to_python_source(self, token_ids):
    """Returns the Python source represented by the token ids list."""
    return "\n".join(self.decode_to_python_statements(token_ids))

  def decode_to_python_statements(self, token_ids):
    """Returns the Python source represented by the token ids list."""
    statements = []
    token_ids_iter = iter(token_ids)
    for statement_token_ids in zip(*[token_ids_iter]
                                   * self.tokens_per_statement):
      statement = self._decode_to_python_statement(statement_token_ids)
      statements.append(statement)
    return statements

  def _decode_to_python_statement(self, token_ids):
    """Returns the single Python statement represented by the token ids list."""
    indent_token_id, op_token_id, var_token_id = token_ids[:3]
    indent = self.indent_from_token_id(indent_token_id)
    indent_str = constants.INDENT_STRING * int(indent)

    if op_token_id == self.placeholder_token_id():
      return f"{indent_str}_ = 0"

    operand_token_ids = token_ids[3:]
    operand = self.operand_from_token_ids(operand_token_ids)
    var_index = self.var_index_from_token_id(var_token_id)

    op = self.op_from_token_id(op_token_id)
    if op.startswith("while") or op.startswith("if"):
      if "%" in op:
        control_op, cond_op, mod_op = op.split()
        return (f"{indent_str}{control_op} v{var_index} {mod_op} 10 {cond_op} "
                f"{operand}:")
      else:
        control_op, cond_op = op.split()
        return f"{indent_str}{control_op} v{var_index} {cond_op} {operand}:"
    else:
      return f"{indent_str}v{var_index} {op} {operand}"

  def decode_to_tokens(self, token_ids):
    """Returns the Python source represented by the token ids list."""
    all_tokens = []
    token_ids_iter = iter(token_ids)
    for statement_token_ids in zip(*[token_ids_iter]
                                   * self.tokens_per_statement):
      tokens = self._decode_to_tokens(statement_token_ids)
      all_tokens.append(tokens)
    return all_tokens

  def _decode_to_tokens(self, token_ids):
    """Returns the single Python statement represented by the token ids list."""
    indent_token_id, op_token_id, var_token_id = token_ids[:3]
    indent = int(self.indent_from_token_id(indent_token_id))

    if op_token_id == self.placeholder_token_id():
      return [indent, "-", "-", "-"]

    operand_token_ids = token_ids[3:]
    operand = self.operand_from_token_ids(operand_token_ids)
    var_index = self.var_index_from_token_id(var_token_id)

    op = self.op_from_token_id(op_token_id)
    return [indent, op, f"v{var_index}", operand]

  @property
  def tokens_per_statement(self):
    return 3 + self.num_digits

  @property
  def vocab_size(self):
    return self.placeholder_token_id() + 1

  def padding_token_id(self):
    # Padding token id is 0.
    return 0

  def indent_token_id(self, indent):
    # Indent token ids are 1..{max_indent}.
    return indent + 1

  def indent_from_token_id(self, indent_token_id):
    return indent_token_id - 1

  def operand_token_ids(self, operand):
    # Operand token ids are 1..{base}. Note this is not 0..{base-1}.
    nary_list = as_nary_list(int(operand), self.base, self.num_digits)
    return [digit + 1 for digit in nary_list]

  def operand_from_token_ids(self, operand_token_ids):
    nary_list = [operand_token_id - 1 for operand_token_id in operand_token_ids]
    return nary_list_as_number(nary_list, self.base)

  def op_token_id(self, op):
    # Operand token ids are {base+1}..{base+len(ops)}.
    return self.base + self.ops_index[op] + 1

  def op_from_token_id(self, op_token_id):
    op_index = op_token_id - self.base - 1
    return self.ops[op_index]

  def var_token_id(self, var):
    # Var token ids are {base+len(ops)+1}..{base+len(ops)+num_variables}.
    return self.base + len(self.ops) + self.var_index(var) + 1

  def var_index_from_token_id(self, var_token_id):
    return var_token_id - self.base - len(self.ops) - 1

  def var_index(self, var_name):
    return int(var_name.lstrip("v"))

  def var_padding_token_id(self):
    # Operand padding token id is {base+len(ops)+num_variables+1}.
    return self.base + len(self.ops) + self.num_variables + 1

  def placeholder_token_id(self):
    # Placeholder token id is {base+len(ops)+num_variables+2}.
    return self.base + len(self.ops) + self.num_variables + 2

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".vocab"

  def save_to_file(self, filename_prefix):
    filename = self._filename(filename_prefix)
    vocab_list = [str(i) for i in range(self.vocab_size)]
    metadata_dict = dict(base=self.base,
                         num_digits=self.num_digits,
                         ops=self.ops,
                         num_variables=self.num_variables)
    self._write_lines_to_file(filename, vocab_list, metadata_dict=metadata_dict)

  @classmethod
  def load_from_file(cls, filename_prefix):
    filename = cls._filename(filename_prefix)
    unused_vocab_list, metadata_dict = cls._read_lines_from_file(filename)
    encoder = cls(base=metadata_dict["base"],
                  num_digits=metadata_dict["num_digits"],
                  ops=metadata_dict["ops"],
                  num_variables=metadata_dict["num_variables"])
    return encoder


class TextSourceEncoder(tfds.deprecated.text.text_encoder.TextEncoder):
  """Encoder for encoding programs from source.

  Attributes:
    fragment_length: The maximum number of characters to include in a single
      line fragment.
  """

  def __init__(self, fragment_length=20):
    """Initializes the TextSourceEncoder."""
    self.fragment_length = fragment_length

  def encode(self, python_source):
    token_ids = []
    for line in python_source.split("\n"):
      token_ids.extend(self.line_token_ids(line))
    return token_ids

  def line_token_ids(self, line):
    token_ids = []
    remainder = line
    while remainder:
      fragment = remainder[:self.fragment_length]
      remainder = remainder[self.fragment_length:]
      continues = bool(remainder)
      token_ids.extend(self.fragment_token_ids(fragment, continues))
    return token_ids

  def fragment_token_ids(self, fragment, continues):
    return (
        [self.char_token_id(char) for char in fragment]
        + [self.padding_token_id()] * (self.fragment_length - len(fragment))
        + [self.continues_token_id(continues)]
    )

  def fragment_from_token_ids(self, token_ids):
    fragment = [self.char_from_token_id(token_id)
                for token_id in token_ids
                if token_id > 2]  # Skip padding and continues tokens.
    continues = self.continues_from_token_id(token_ids[-1])
    return fragment, continues

  def padding_token_id(self):
    # Padding token id is 0.
    return 0

  def continues_token_id(self, continues):
    # Continues is 1, 2.
    return 1 if continues else 2

  def continues_from_token_id(self, token_id):
    return token_id == 1

  def char_token_id(self, char):
    # Chars are 3..255+3.
    return ord(char) + 3

  def char_from_token_id(self, token_id):
    return chr(token_id - 3)

  def decode(self, token_ids):
    return self.decode_to_python_source(token_ids)

  def decode_to_string(self, token_ids):
    """Produces a human-readable representation of the token list."""
    return self.decode_to_python_source(token_ids)

  def decode_to_python_source(self, token_ids):
    """Returns the Python source represented by the token ids list."""
    return "\n".join(self.decode_to_python_statements(token_ids))

  def decode_to_python_statements(self, token_ids):
    """Returns the Python source represented by the token ids list."""
    statements = []
    token_ids_iter = iter(token_ids)
    current_statement = []
    for fragment_token_ids in zip(*[token_ids_iter]
                                  * self.tokens_per_statement):
      fragment, continues = self.fragment_from_token_ids(fragment_token_ids)
      current_statement.extend(fragment)
      if not continues:
        statements.append("".join(current_statement))
        current_statement = []
    return statements

  @property
  def tokens_per_statement(self):
    # +1 for the continuation bit.
    return self.fragment_length + 1

  @property
  def vocab_size(self):
    return 256 + 3

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".vocab"

  def save_to_file(self, filename_prefix):
    filename = self._filename(filename_prefix)
    vocab_list = [str(i) for i in range(self.vocab_size)]
    metadata_dict = dict(fragment_length=self.fragment_length)
    self._write_lines_to_file(filename, vocab_list, metadata_dict=metadata_dict)

  @classmethod
  def load_from_file(cls, filename_prefix):
    filename = cls._filename(filename_prefix)
    unused_vocab_list, metadata_dict = cls._read_lines_from_file(filename)
    encoder = cls(fragment_length=metadata_dict["fragment_length"])
    return encoder


def as_nary_list(number, base, length):
  result = []
  remainder = number
  for _ in range(length):
    result.append(int(remainder % base))
    remainder = int(remainder / base)
  if remainder:
    raise ValueError("Number could not be converted to n-ary list.",
                     number, base, length)
  return list(reversed(result))


def nary_list_as_number(nary_list, base):
  number = 0
  for digit in nary_list:
    number *= base
    number += digit
  return number


class PassThruEncoder(tfds.deprecated.text.text_encoder.TextEncoder):
  """PassThruEncoder is a "TextEncoder" that encodes ints rather than strings.

  It does this encoding without modifying the integers, so it is really just a
  pass through.
  """

  def __init__(self, vocab_size):
    """Constructs a PassThruEncoder with the given vocab_size."""
    self._vocab_size = vocab_size

  def encode(self, values):
    if not all(0 <= value < self.vocab_size for value in values):
      raise ValueError("All values must be 0 <= value < vocab_size",
                       values, self.vocab_size)
    return values

  def decode(self, token_ids):
    return token_ids

  def decode_to_string(self, token_ids):
    """Produces a human-readable representation of the token list."""
    return ",".join(str(value) for value in self.decode(token_ids))

  @property
  def vocab_size(self):
    return self._vocab_size

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + ".vocab"

  def save_to_file(self, filename_prefix):
    filename = self._filename(filename_prefix)
    vocab_list = [str(i) for i in range(self.vocab_size)]
    metadata_dict = dict(vocab_size=self.vocab_size)
    self._write_lines_to_file(filename, vocab_list, metadata_dict=metadata_dict)

  @classmethod
  def load_from_file(cls, filename_prefix):
    filename = cls._filename(filename_prefix)
    unused_vocab_list, metadata_dict = cls._read_lines_from_file(filename)
    encoder = cls(vocab_size=metadata_dict["vocab_size"])
    return encoder
