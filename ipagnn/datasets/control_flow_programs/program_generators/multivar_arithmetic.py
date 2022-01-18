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

"""Generates straight-line programs performing arithmetic on multiple variables.
"""

import dataclasses
import random

from typing import Optional, Text, Tuple

DEFAULT_OPS = ("+=", "-=", "*=")


@dataclasses.dataclass
class MultivarArithmeticConfig(object):
  """The config class for the multivar arithmetic generator."""

  base: int
  length: int
  num_digits: int = 1
  variables: int = 2
  constant_probability: float = 0.75
  max_value: Optional[int] = None
  ops: Tuple[Text, Ellipsis] = DEFAULT_OPS
  encoder_name: Text = "simple"
  mod: Optional[int] = None
  output_mod: Optional[int] = None


def generate_python_source(length, config):
  """Generates Python code according to the config."""
  max_value = config.max_value or (config.base ** config.num_digits - 1)
  # Initialize v1..vN to 0.
  statements = [
      "v{} = 0".format(i)
      for i in range(1, config.variables)
  ]
  used_variables = {0}
  for _ in range(length):
    # choose variable to modify
    var = random.randint(0, config.variables - 1)
    # choose operation
    op = random.choice(config.ops)
    # choose constant or existing variable
    use_constant = random.random() < config.constant_probability
    if use_constant:
      value = random.randint(0, max_value)
    else:
      # Choose a random variable.
      value = "v{}".format(random.choice(list(used_variables)))
    used_variables.add(var)
    statement = "v{var} {op} {value}".format(
        var=var,
        op=op,
        value=value,
    )
    statements.append(statement)
  return "\n".join(statements)
