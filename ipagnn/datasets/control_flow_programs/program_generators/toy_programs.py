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

"""A simple program generator that randomly selects from a few short programs.
"""

import dataclasses
import random

from typing import Optional, Text, Tuple


DEFAULT_OPS = ("+=", "-=", "*=")


@dataclasses.dataclass
class ToyProgramsConfig(object):
  """The config class for the toy program generator."""
  base: int
  length: int = 3
  programs_index: int = 0
  num_digits: int = 1
  max_value: int = 10
  ops: Tuple[Text, Ellipsis] = DEFAULT_OPS
  encoder_name: Text = "simple"
  mod: Optional[int] = 10
  output_mod: Optional[int] = None


def get_programs(config):
  """A set of toy programs to use in the dataset."""
  if config.programs_index == 0:
    return [
        """
v0 += 2
v0 += 2
""",
        """
v0 += 1
v0 += 1
""",
        """
v0 += 3
v0 += 3
""",
    ]
  elif config.programs_index == 1:
    return [
        """
v0 += 2
v0 += 1
v0 -= 2
""",
        """
v0 += 1
v0 += 2
v0 -= 1
""",
        """
v0 += 1
v0 += 1
v0 += 1
""",
    ]
  elif config.programs_index == 2:
    return [
        """
v0 += 1
""",
        """
v0 += 1
v0 += 2
""",
        """
v0 += 1
v0 += 2
v0 += 3
""",
    ]
  elif config.programs_index == 3:
    return [
        """
v0 = 673
v7 = 7
while v7 > 0:
  v7 -= 1
  v0 -= 9
if v0 % 10 < 5:
  v0 -= 6
  v0 *= 1
  v0 -= 4
v0 += 7
""",
        """
v0 = 159
v7 = 7
while v7 > 0:
  v7 -= 1
  v0 -= 9
if v0 % 10 < 5:
  v0 -= 6
  v0 *= 1
  v0 -= 4
v0 += 7
""",
        """
v0 = 549
v7 = 7
while v7 > 0:
  v7 -= 1
  v0 -= 9
if v0 % 10 < 5:
  v0 -= 6
  v0 *= 1
  v0 -= 4
v0 += 7
""",
    ]
  elif config.programs_index == 4:
    # Figure programs
    return [
        """
v0 = 849
if v0 % 10 < 5:
  v0 -= 3
else:
  v0 -= 4
  v0 += 2
  v0 *= 9
if v0 % 10 >= 4:
  if v0 % 10 < 6:
    v0 *= 8
""",
        """
v0 = 323
if v0 % 10 < 5:
  v0 -= 3
else:
  v0 -= 4
  v0 += 2
  v0 *= 9
if v0 % 10 >= 4:
  if v0 % 10 < 6:
    v0 *= 8
""",
    ]
  elif config.programs_index == 5:
    # Figure programs 2
    return [
        """
v0 = 23
v1 = 6
while v1 > 0:
  v1 -= 1
  if v0 % 10 <= 3:
    v0 += 4
    v0 *= 6
  v0 -= 1
""",
        """
v0 = 890
v1 = 6
while v1 > 0:
  v1 -= 1
  if v0 % 10 <= 3:
    v0 += 4
    v0 *= 6
  v0 -= 1
""",
    ]
  elif config.programs_index == 6:
    return [
        """
v0 = 847
v0 *= 2
v0 *= 7
v0 *= 8
v9 = 7
while v9 > 0:
  v9 -= 1
  v0 += 8
v0 += 9
v0 *= 9
""",
        """
v0 = 183
v0 *= 2
v0 *= 7
v0 *= 8
v9 = 7
while v9 > 0:
  v9 -= 1
  v0 += 8
v0 += 9
v0 *= 9
""",
        """
v0 = 214
v0 *= 2
v0 *= 7
v0 *= 8
v9 = 7
while v9 > 0:
  v9 -= 1
  v0 += 8
v0 += 9
v0 *= 9
""",
        """
v0 = 251
v0 *= 2
v0 *= 7
v0 *= 8
v9 = 7
while v9 > 0:
  v9 -= 1
  v0 += 8
v0 += 9
v0 *= 9
""",
    ]
  elif config.programs_index == 7:
    return [
        """
v0 = 670
v5 = 8
while v5 > 0:
  v5 -= 1
  v0 -= 4
v4 = 3
while v4 > 0:
  v4 -= 1
  v0 *= 6
v0 *= 4
""",
        """
v0 = 671
v5 = 8
while v5 > 0:
  v5 -= 1
  v0 -= 4
v4 = 3
while v4 > 0:
  v4 -= 1
  v0 *= 6
v0 *= 4
""",
        """
v0 = 672
v5 = 8
while v5 > 0:
  v5 -= 1
  v0 -= 4
v4 = 3
while v4 > 0:
  v4 -= 1
  v0 *= 6
v0 *= 4
""",
    ]
  elif config.programs_index == 8:
    return [
        """
v0 = 147
v0 -= 7
v7 = 4
while v7 > 0:
  v7 -= 1
  v0 += 8
  v0 *= 2
  if v0 % 10 <= 8:
    v0 -= 1
v0 -= 3
""",
        """
v0 = 149
v0 -= 7
v7 = 4
while v7 > 0:
  v7 -= 1
  v0 += 8
  v0 *= 2
  if v0 % 10 <= 8:
    v0 -= 1
v0 -= 3
""",
        """
v0 = 219
v0 -= 7
v7 = 4
while v7 > 0:
  v7 -= 1
  v0 += 8
  v0 *= 2
  if v0 % 10 <= 8:
    v0 -= 1
v0 -= 3
""",
        """
v0 = 239
v0 -= 7
v7 = 4
while v7 > 0:
  v7 -= 1
  v0 += 8
  v0 *= 2
  if v0 % 10 <= 8:
    v0 -= 1
v0 -= 3
""",
        """
v0 = 251
v0 -= 7
v7 = 4
while v7 > 0:
  v7 -= 1
  v0 += 8
  v0 *= 2
  if v0 % 10 <= 8:
    v0 -= 1
v0 -= 3
""",
    ]
  elif config.programs_index == 9:
    return [
        """
v0 = 261
if v0 % 10 > 0:
  v9 = 5
  while v9 > 0:
    v9 -= 1
    if v0 % 10 > 8:
      v0 *= 9
      v0 += 4
    else:
      v0 += 1
""",
        """
v0 = 262
if v0 % 10 > 0:
  v9 = 5
  while v9 > 0:
    v9 -= 1
    if v0 % 10 > 8:
      v0 *= 9
      v0 += 4
    else:
      v0 += 1
""",
        """
v0 = 268
if v0 % 10 > 0:
  v9 = 5
  while v9 > 0:
    v9 -= 1
    if v0 % 10 > 8:
      v0 *= 9
      v0 += 4
    else:
      v0 += 1
""",
        """
v0 = 569
if v0 % 10 > 0:
  v9 = 5
  while v9 > 0:
    v9 -= 1
    if v0 % 10 > 8:
      v0 *= 9
      v0 += 4
    else:
      v0 += 1
""",
        """
v0 = 549
if v0 % 10 > 0:
  v9 = 5
  while v9 > 0:
    v9 -= 1
    if v0 % 10 > 8:
      v0 *= 9
      v0 += 4
    else:
      v0 += 1
""",
        """
v0 = 349
if v0 % 10 > 0:
  v9 = 5
  while v9 > 0:
    v9 -= 1
    if v0 % 10 > 8:
      v0 *= 9
      v0 += 4
    else:
      v0 += 1
""",
    ]
  elif config.programs_index == 10:
    return [
        """
v0 = 36
if v0 % 10 >= 7:
  v0 *= 3
  if v0 % 10 > 3:
    v0 *= 4
    v5 = 3
    while v5 > 0:
      v5 -= 1
      break
v0 *= 2
""",
        """
v0 = 589
if v0 % 10 >= 8:
  v0 *= 4
else:
  if v0 % 10 < 0:
    v0 *= 1
  else:
    if v0 % 10 >= 6:
      if v0 % 10 < 3:
        v0 += 9
""",
        """
v0 = 528
v0 *= 1
v0 += 9
v0 += 3
if v0 % 10 < 8:
  if v0 % 10 < 3:
    if v0 % 10 < 0:
      v0 -= 7
    v0 -= 9
""",
        """
v0 = 117
if v0 % 10 <= 6:
  v0 -= 9
  v0 += 7
else:
  v1 = 2
  while v1 > 0:
    v1 -= 1
    v0 -= 6
v0 *= 1
""",
    ]
  elif config.programs_index == 11:
    return [
        ("""
v0 = 36
if v0 % 10 >= 7:
  v0 *= 3
  if v0 % 10 > 3:
    v0 *= 4
    v5 = 3
    while v5 > 0:
      v5 -= 1
      break
v0 *= 2
""", 2),
        ("""
v0 = 589
if v0 % 10 >= 8:
  v0 *= 4
else:
  if v0 % 10 < 0:
    v0 *= 1
  else:
    if v0 % 10 >= 6:
      if v0 % 10 < 3:
        v0 += 9
""", 9),
        ("""
v0 = 528
v0 *= 1
v0 += 9
v0 += 3
if v0 % 10 < 8:
  if v0 % 10 < 3:
    if v0 % 10 < 0:
      v0 -= 7
    v0 -= 9
""", 1),
        ("""
v0 = 117
if v0 % 10 <= 6:
  v0 -= 9
  v0 += 7
else:
  v1 = 2
  while v1 > 0:
    v1 -= 1
    v0 -= 6
v0 *= 1
""", 8),
    ]
  elif config.programs_index == 12:
    return [
        """
v0 = 305
v0 *= 3
if v0 % 10 > 0:
  v4 = 7
  while v4 > 0:
    v4 -= 1
    v0 += 5
  if v0 % 10 >= 4:
    v5 = 5
    while v5 > 0:
      v5 -= 1
      v4 = 4
      while v4 > 0:
        v4 -= 1
        v0 -= 1
  if v0 % 10 < 4:
    v0 += 5
  else:
    v0 -= 6
v0 += 7
""",
    ]


def generate_python_source(length, config):
  """Generates Python code according to the config."""
  del length  # Unused.
  programs = get_programs(config)
  program = random.choice(programs)
  return program.strip()


def generate_python_source_and_partial_python_source(length, config):
  """Generates Python code according to the config."""
  del length  # Unused.
  programs = get_programs(config)
  program, placeholder_index = random.choice(programs)
  program = program.strip()
  lines = program.split("\n")
  line_to_replace = lines[placeholder_index]
  indent = int(
      (len(line_to_replace) - len(line_to_replace.lstrip())))
  new_line = indent * " " + "_ = 0"
  lines[placeholder_index] = new_line
  return program, "\n".join(lines)
