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

"""Config definition for the ArithmeticIfRepeats program generator."""

import dataclasses
from typing import Optional, Text, Tuple


DEFAULT_OPS = ("+=", "-=", "*=")


@dataclasses.dataclass
class ArithmeticIfRepeatsConfig:
  """Config for ArithmeticIfRepeats ProgramGenerator.

  Attributes:
    base: The base to represent the integers in.
    length: The number of statements in the generated programs.
    num_digits: The number of digits in the values used by the programs.
    max_repeat_statements: The maximum number of repeat statements allowed in
      a program.
    max_repetitions: The maximum number of repetitions a repeat statement may
      specify.
    repeat_probability: The probability that a given statement is a repeat
      statement, provided a repeat statement is possible at that location.
    max_if_statements: The maximum number of if statements allowed in a program.
    if_probability: The probability that a given statement is an if statement,
      provided an if statement is possible at that location.
    ifelse_probability: The probability that a given statement is an if-else
      statement, provided an if statement is possible at that location.
    max_nesting: The maximum depth of nesting permitted, or None if no limit.
    max_block_size: The maximum number of statements permitted in a block.
    ops: The ops allowed in the generated programs.
    encoder_name: The encoder name to use to encode the generated programs.
    mod: The value (if any) to mod the intermediate values of the program by
      after each step of execution.
    output_mod: The value (if any) to mod the final values of the program by.
  """
  base: int
  length: int
  num_digits: int = 1
  max_repeat_statements: Optional[int] = 2
  max_repetitions: int = 9
  repeat_probability: float = 0.1
  max_if_statements: Optional[int] = 2
  if_probability: float = 0.2
  ifelse_probability: float = 0.2
  max_nesting: Optional[int] = None
  max_block_size: Optional[int] = 9
  ops: Tuple[Text, Ellipsis] = DEFAULT_OPS
  encoder_name: Text = "simple"
  mod: Optional[int] = 10
  output_mod: Optional[int] = None
