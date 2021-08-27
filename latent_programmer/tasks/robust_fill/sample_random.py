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

"""Generates random string transformation programs in RobustFill DSL.

Changed to support entire RobustFill DSL instead of just SubStr and ConstStr.
"""


import random
import string
from typing import Dict, List, Optional, Tuple

from latent_programmer.tasks.robust_fill import dsl


# Maximum number of characters in a sampled substring token.
MAX_TOKEN_LENGTH = 5


def sample_type_str(t):
  """Map types to their regex string."""
  if t == dsl.Type.NUMBER or t == dsl.Type.DIGIT:
    return get_number()
  elif t == dsl.Type.WORD:
    return get_word()
  elif t == dsl.Type.ALPHANUM or t == dsl.Type.CHAR:
    return get_alphanumeric()
  elif t == dsl.Type.ALL_CAPS:
    return get_caps()
  elif t == dsl.Type.PROP_CASE:
    return get_proper_case()
  elif t == dsl.Type.LOWER:
    return get_lower()
  else:
    raise ValueError('Unsupported type: {}'.format(t))


def get_lower():
  l = random.randint(1, MAX_TOKEN_LENGTH)
  constant = random.choice(dsl.DELIMITER)
  return constant + ''.join(random.choice(string.ascii_lowercase)
                            for _ in range(l))


def get_number():
  l = random.randint(1, MAX_TOKEN_LENGTH)
  constant = random.choice(dsl.DELIMITER)
  return constant + ''.join(random.choice(string.digits) for _ in range(l))


def get_caps():
  l = random.randint(1, MAX_TOKEN_LENGTH)
  constant = random.choice(dsl.DELIMITER)
  return constant + ''.join(random.choice(string.ascii_uppercase)
                            for _ in range(l))


def get_word():
  l = random.randint(1, MAX_TOKEN_LENGTH)
  constant = random.choice(dsl.DELIMITER)
  return constant + ''.join(random.choice(string.ascii_letters)
                            for _ in range(l))


def get_alphanumeric():
  l = random.randint(1, MAX_TOKEN_LENGTH)
  constant = random.choice(dsl.DELIMITER)
  return constant + ''.join(random.choice(string.ascii_letters + string.digits)
                            for _ in range(l))


def get_proper_case():
  l = random.randint(1, MAX_TOKEN_LENGTH)
  constant = random.choice(dsl.DELIMITER)
  capital = random.choice(string.ascii_uppercase)
  return constant + capital + ''.join(random.choice(string.ascii_lowercase)
                                      for _ in range(l))


def sample_inputs(num_examples,
                  max_input_tokens,
                  max_k,
                  max_input_length
                 ):
  """Returns `numExamples` inputs satisfying the provided constraints.

  Args:
    num_examples: Number of inputs to generate.
    max_input_tokens: Maximum number of unique tokens in the inputs. A token is
        either a constant string, or a sample from a regular expression.
    max_k: Maximum number of times a generated token can be repeated.
    max_input_length: Maximum length of inputs to generate.
  Returns:
    inputs: List of `numExamples` strings representing the inputs.
    constTokDict: Dictionary mapping constant indices to occurrence counts.
    regexTokDict: Dictionary mapping regex indices to occurrence counts.
  """
  while True:
    delimiter_dict = {}
    type_dict = {}

    n_toks = random.randint(3, max_input_tokens)

    input_lists = []
    for _ in range(num_examples):
      input_lists.append([])

    n_tok = 0
    input_length = 0
    while n_tok < n_toks:
      n_tok += 1
      is_delimiter = random.randint(1, 3)
      if is_delimiter == 1:
        delimiter = random.choice(dsl.DELIMITER)
        k = random.randint(1, max_k)
        delimiter_dict[delimiter] = delimiter_dict.get(delimiter, 0) + k
        for _ in range(k):
          for j in range(num_examples):
            input_lists[j].append(delimiter)
          input_length += 1
      else:
        type_ = random.choice(list(dsl.Type))
        k = random.randint(1, max_k)
        type_dict[type_] = type_dict.get(type_, 0) + k
        for _ in range(k):
          max_len = 0
          for j in range(num_examples):
            s = sample_type_str(type_)
            input_lists[j].append(s)
            delimiter_dict[s[0]] = delimiter_dict.get(s[0], 0) + 1
            max_len = max(max_len, len(s))
          input_length += max_len

      # Early stopping if the sampled input strings are close to maximum length.
      if input_length >= max_input_length - MAX_TOKEN_LENGTH * max_k // 2:
        break

    inputs = []
    for i in range(num_examples):
      random.shuffle(input_lists[i])
      inputs.append(''.join(input_lists[i]))

    # Everything is a character.
    type_dict[dsl.Type.CHAR] = max(len(input_value) for input_value in inputs)

    # Rejection step on input lengths.
    if max(len(input_value) for input_value in inputs) > max_input_length:
      continue
    return inputs, delimiter_dict, type_dict


def random_task(max_expressions,
                max_k,
                max_input_tokens,
                max_input_length,
                max_output_length,
                num_examples,
                min_expressions = 1,
                n_expressions = None,
               ):
  """Returns a sampled program and IO examples satisfying the given constraints.

  Args:
    max_expressions: Maximum number of concatenated expressions in the program.
    max_k: Maximum number of times a generated token can be repeated.
    max_input_tokens: Maximum number of unique tokens in the inputs. A token is
        either a constant string, or a sample from a regular expression.
    max_input_length: Maximum length of inputs to generate.
    max_output_length: Maximum length of outputs to generate.
    num_examples: Number of input-output examples to generate.
    min_expressions: Minimum number of concatenated expressions in the program.
    n_expressions: Fixed number of concatenated expressions (if provided)
  Returns:
    Input strings, output strings, and a program expression.
  """

  # Sample inputs.
  inputs, delimiter_dict, type_dict = sample_inputs(
      num_examples, max_input_tokens, max_k, max_input_length)

  # Sample program.
  if not n_expressions:
    n_expressions = random.randint(min_expressions, max_expressions)
  while True:
    program = dsl.Concat(
        *[random_expression(inputs, delimiter_dict, type_dict)
          for _ in range(n_expressions)])

    outputs = [program(inp) for inp in inputs]
    # Rejection step on output lengths.
    if ((max(len(out) for out in outputs) <= max_output_length) and
        (min(len(out) for out in outputs) > 0)):
      return dsl.ProgramTask(program, inputs, outputs)


def random_expression(inputs, delimiter_dict, type_dict):
  sampler = random.choice([
      random_substring,
      random_nesting,
      random_compose,
      random_const_str,
  ])
  return sampler(inputs, delimiter_dict, type_dict)


def random_substring(inputs, delimiter_dict, type_dict):
  sampler = random.choice([
      random_sub_str,
      random_get_span,
  ])
  return sampler(inputs, delimiter_dict, type_dict)


def random_nesting(inputs, delimiter_dict, type_dict):
  """Samples random Nesting."""
  sampler = random.choice([
      random_get_token,
      random_to_case,
      random_replace,
      random_trim,
      random_get_upto,
      random_get_from,
      random_get_first,
      random_get_all,
  ])
  return sampler(inputs, delimiter_dict, type_dict)


def random_compose(inputs, delimiter_dict, type_dict):
  """Samples random Compose expression."""
  nesting_or_substring = random.choice([
      random_nesting,
      random_substring,
  ])
  while True:
    expr = dsl.Compose(random_nesting(inputs, delimiter_dict, type_dict),
                       nesting_or_substring(inputs, delimiter_dict, type_dict))
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr


def random_const_str(inputs, delimiter_dict, type_dict):
  del inputs, delimiter_dict, type_dict
  char = random.choice(dsl.CHARACTER)
  return dsl.ConstStr(char)


def random_sub_str(inputs, delimiter_dict, type_dict):
  """Samples random SubStr expression."""
  del delimiter_dict, type_dict
  while True:
    # Make sure indices are in range.
    min_input_length = min(len(input_value) for input_value in inputs)
    positions = [-min_input_length + 1, min_input_length]
    pos1 = random.randint(*positions)
    if pos1 > 0:
      pos2 = random.randint(*[pos1, positions[1]])
    else:
      pos2 = random.randint(*[pos1, 0])

    expr = dsl.SubStr(pos1, pos2)
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr


def random_type(type_dict):
  return random.choice(list(type_dict.keys()))


def random_delimiter(delimiter_dict):
  return random.choice(list(delimiter_dict.keys()))


def random_boundary():
  return random.choice(list(dsl.Boundary))


def random_get_span(inputs, delimiter_dict, type_dict):
  """Samples random GetSpan expression."""
  while True:
    is_delimiters = [random.randint(1, 2), random.randint(1, 2)]
    if delimiter_dict and is_delimiters[0] == 1:
      r1 = random_delimiter(delimiter_dict)
      indices = [i for i in dsl.INDEX if abs(i) <= delimiter_dict[r1]]
      i1 = random.choice(indices)
    else:
      r1 = random_type(type_dict)
      indices = [i for i in dsl.INDEX if abs(i) <= type_dict[r1]]
      i1 = random.choice(indices)
    if delimiter_dict and is_delimiters[1] == 1:
      r2 = random_delimiter(delimiter_dict)
      indices = [i for i in dsl.INDEX if abs(i) <= delimiter_dict[r2]]
      i2 = random.choice(indices)
    else:
      r2 = random_type(type_dict)
      indices = [i for i in dsl.INDEX if abs(i) <= type_dict[r2]]
      i2 = random.choice(indices)

    expr = dsl.GetSpan(
        r1, i1, random_boundary(), r2, i2, random_boundary())
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr


def random_get_token(inputs, delimiter_dict, type_dict):
  """Samples random GetToken expression."""
  del delimiter_dict
  while True:
    t = random_type(type_dict)
    indices = [i for i in dsl.INDEX if abs(i) <= type_dict[t]]
    i = random.choice(indices)

    expr = dsl.GetToken(t, i)
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr


def random_to_case(inputs, delimiter_dict, type_dict):
  del inputs, delimiter_dict, type_dict
  case = random.choice(list(dsl.Case))
  return dsl.ToCase(case)


def random_replace(inputs, delimiter_dict, type_dict):
  del inputs, type_dict
  return dsl.Replace(random_delimiter(delimiter_dict),
                     random.choice(dsl.DELIMITER))


def random_trim(inputs, delimiter_dict, type_dict):
  del inputs, delimiter_dict, type_dict
  return dsl.Trim()


def random_get_upto(inputs, delimiter_dict, type_dict):
  """Samples random GetUpto expression."""
  while True:
    is_delimiter = random.randint(1, 2)
    if delimiter_dict and is_delimiter == 1:
      r = random_delimiter(delimiter_dict)
    else:
      r = random_type(type_dict)

    expr = dsl.GetUpto(r)
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr


def random_get_from(inputs, delimiter_dict, type_dict):
  """Samples random GetFrom expression."""
  while True:
    is_delimiter = random.randint(1, 2)
    if delimiter_dict and is_delimiter == 1:
      r = random_delimiter(delimiter_dict)
    else:
      r = random_type(type_dict)

    expr = dsl.GetFrom(r)
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr


def random_get_first(inputs, delimiter_dict, type_dict):
  """Samples random GetFirst expression."""
  del delimiter_dict
  while True:
    t = random_type(type_dict)
    indices = [i for i in dsl.INDEX if abs(i) <= type_dict[t]]
    i = random.choice(indices)

    expr = dsl.GetFirst(t, i)
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr


def random_get_all(inputs, delimiter_dict, type_dict):
  """Samples random GetAll expression."""
  del delimiter_dict
  while True:
    t = random_type(type_dict)

    expr = dsl.GetAll(t)
    # Make sure outputs are non-empty.
    try:
      if min(len(expr(input_value)) for input_value in inputs) == 0:
        continue
    except:  # pylint: disable=[bare-except]
      continue
    return expr
