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

"""Config definition for the program generator."""

import dataclasses
from typing import Optional, Text, Tuple

DEFAULT_OPS = ("+=", "-=", "*=")


@dataclasses.dataclass
class ArithmeticRepeatsConfig:
  """Config for ArithmeticRepeats program generator.

  Attributes:
    base: The base to represent the integers in.
    length: The number of statements in the generated programs.
    num_digits: The number of digits in the values used by the programs.
    max_repeat_statements: The maximum number of repeat statements allowed in
      a program.
    max_repetitions: The maximum number of repetitions a repeat statement may
      specify.
    max_repeat_block_size: The maximum number of statements permitted in a
      repeat block.
    repeat_probability: The probability that a given statement is a repeat
      statement, provided a repeat statement is possible at that location.
    permit_nested_repeats: Whether or not nested repeat statements are
      permitted.
    ops: The ops allowed in the generated programs.
    encoder_name: The encoder name to use to encode the generated programs.
    mod: The value (if any) to mod the intermediate values of the program by
      after each step of execution.
    output_mod: The value (if any) to mod the final values of the program by.
    start_with_initialization: If true, the program's first line is v0 = _.
  """
  base: int
  length: int
  num_digits: int = 1
  max_repeat_statements: int = 2
  max_repetitions: int = 9
  max_repeat_block_size: int = 9
  repeat_probability: float = 0.1
  permit_nested_repeats: bool = False
  ops: Tuple[Text, Ellipsis] = DEFAULT_OPS
  encoder_name: Text = "simple"
  mod: Optional[int] = 10
  output_mod: Optional[int] = None
  start_with_initialization: bool = False
