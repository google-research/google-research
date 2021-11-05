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


import collections
import random
import string
from typing import Dict, List, Optional, Tuple

from absl import logging

from latent_programmer.tasks.robust_fill import dsl


# Maximum number of characters in a sampled substring token.
MAX_TOKEN_LENGTH = 4


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
  l = random.randint(1, MAX_TOKEN_LENGTH - 1)
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
    delimiter_dict = collections.defaultdict(int)
    type_dict = collections.defaultdict(int)

    n_toks = random.randint(3, max_input_tokens)

    input_lists = []
    for _ in range(num_examples):
      input_lists.append([])

    for i in range(n_toks):
      is_delimiter = random.randint(1, 3)
      if is_delimiter == 1:
        delimiter = random.choice(dsl.DELIMITER)
        k = random.randint(1, max_k)
        delimiter_dict[delimiter] += k
        for _ in range(k):
          for j in range(num_examples):
            input_lists[j].append(delimiter)
      else:
        type_ = random.choice(list(dsl.Type))
        k = random.randint(1, max_k)
        type_dict[type_] += k
        for _ in range(k):
          for j in range(num_examples):
            s = sample_type_str(type_)
            input_lists[j].append(s)
            delimiter_dict[s[0]] += 1

      input_length = max(sum(len(s) for s in inner_list)
                         for inner_list in input_lists)
      if input_length >= max_input_length:
        break

    inputs = []
    for i in range(num_examples):
      random.shuffle(input_lists[i])
      inputs.append(''.join(input_lists[i])[:max_input_length])

    # Everything is a character.
    type_dict[dsl.Type.CHAR] = max(len(input_value) for input_value in inputs)

    # Inputs should have appropriate lengths.
    if not all(0 < len(input_str) <= max_input_length for input_str in inputs):
      raise ValueError('Some input has a bad length')

    return inputs, delimiter_dict, type_dict


def random_task(max_expressions,
                max_k,
                max_input_tokens,
                max_input_length,
                num_examples,
                min_expressions = 1,
                n_expressions = None,
                sampler_pool=None,
                valid_num_expressions_fn=None,
                keep_fn=None):
  """Returns a sampled program and IO examples satisfying the given constraints.

  Args:
    max_expressions: Maximum number of concatenated expressions in the program.
    max_k: Maximum number of times a generated token can be repeated.
    max_input_tokens: Maximum number of unique tokens in the inputs. A token is
        either a constant string, or a sample from a regular expression.
    max_input_length: Maximum length of inputs to generate.
    num_examples: Number of input-output examples to generate.
    min_expressions: Minimum number of concatenated expressions in the program.
    n_expressions: Fixed number of concatenated expressions (if provided)
    sampler_pool: Pool of expression to sampled from (if None, all expressions
        are allowed).
    valid_num_expressions_fn: A function that returns True if the number of
        expressions is ok, or False if it should be rejected and re-sampled.
    keep_fn: A function that returns True if the Concat should be kept, or False
        if it should be rejected and re-sampled.
  Returns:
    Input strings, output strings, and a program expression.
  """
  max_output_length = max_input_length * max_expressions

  # Sample inputs.
  inputs, delimiter_dict, type_dict = sample_inputs(
      num_examples, max_input_tokens, max_k, max_input_length)

  # Sample program.
  if not n_expressions:
    while True:
      n_expressions = random.randint(min_expressions, max_expressions)
      if (valid_num_expressions_fn is None
          or valid_num_expressions_fn(n_expressions)):
        break

  while True:
    program = dsl.Concat(
        *[random_expression(inputs, delimiter_dict, type_dict,
                            sampler_pool=sampler_pool)
          for _ in range(n_expressions)])

    outputs = [program(inp) for inp in inputs]
    # Assert output lengths are ok.
    if not all(0 < len(out) <= max_output_length for out in outputs):
      logging.error('Output length not ok')
      logging.error('program: %s', program)
      logging.error('inputs: %s', inputs)
      logging.error('outputs: %s', outputs)
      raise ValueError('Output lengths not ok')

    # Rejection step.
    if keep_fn is not None and not keep_fn(program):
      continue

    return dsl.ProgramTask(program, inputs, outputs)


def random_task_switch_concept_order(
    max_expressions,
    max_k,
    max_input_tokens,
    max_input_length,
    num_examples,
    is_train,
    min_expressions = 1):
  """Returns a sampled program and IO examples satisfying the given constraints.

  Args:
    max_expressions: Maximum number of concatenated expressions in the program.
    max_k: Maximum number of times a generated token can be repeated.
    max_input_tokens: Maximum number of unique tokens in the inputs. A token is
        either a constant string, or a sample from a regular expression.
    max_input_length: Maximum length of inputs to generate.
    num_examples: Number of input-output examples to generate.
    is_train: Whether to generate a task for train or test / finetune.
    min_expressions: Minimum number of concatenated expressions in the program.
  Returns:
    Input strings, output strings, and a program expression.
  """
  max_output_length = max_input_length * max_expressions

  # Sample inputs.
  inputs, delimiter_dict, type_dict = sample_inputs(
      num_examples, max_input_tokens, max_k, max_input_length)

  # Sample program.
  assert min_expressions >= 2
  n_expressions = random.randint(min_expressions, max_expressions)
  n_first_half_expressions = n_expressions // 2
  n_second_half_expressions = n_expressions - n_first_half_expressions
  first_half_sampler_pool = (
      ALL_SUBSTRING if is_train else SAMPLER_POOL_MODIFY_OR_CONST)
  second_half_sampler_pool = (
      SAMPLER_POOL_MODIFY_OR_CONST if is_train else ALL_SUBSTRING)

  expression_list = [
      random_expression(inputs, delimiter_dict, type_dict,
                        sampler_pool=first_half_sampler_pool)
      for _ in range(n_first_half_expressions)
  ] + [
      random_expression(inputs, delimiter_dict, type_dict,
                        sampler_pool=second_half_sampler_pool)
      for _ in range(n_second_half_expressions)
  ]
  program = dsl.Concat(*expression_list)

  outputs = [program(inp) for inp in inputs]
  # Assert output lengths are ok.
  assert all(0 < len(out) <= max_output_length for out in outputs)

  return dsl.ProgramTask(program, inputs, outputs)


def random_expression(inputs, delimiter_dict, type_dict, sampler_pool=None):
  """Samples random expression."""
  if sampler_pool is None:
    sampler_pool = SAMPLER_POOL_ALL
  while True:
    # Sampler pool lists may contain other lists.
    sampler = sampler_pool
    while isinstance(sampler, list):
      sampler = random.choice(sampler)
    expr = sampler(inputs, delimiter_dict, type_dict)
    # Some samplers may return None if it's impossible to create a valid
    # expression, e.g., one that doesn't produce empty outputs.
    if expr is not None:
      return expr


def _is_output_empty(expr, inputs):
  try:
    return min(len(expr(input_value)) for input_value in inputs) == 0
  except:  # pylint: disable=[bare-except]
    return True


def random_compose_modification(inputs, delimiter_dict, type_dict):
  """Samples random Compose expression using only modify ops."""
  while True:
    expr = dsl.Compose(
        random_expression(inputs, delimiter_dict, type_dict,
                          sampler_pool=ALL_MODIFICATION),
        random_expression(inputs, delimiter_dict, type_dict,
                          sampler_pool=ALL_MODIFICATION))
    if not _is_output_empty(expr, inputs):
      return expr


def random_compose_substring(inputs, delimiter_dict, type_dict):
  """Samples random Compose expression."""
  while True:
    expr = dsl.Compose(
        random_expression(inputs, delimiter_dict, type_dict,
                          sampler_pool=ALL_MODIFICATION),
        random_expression(inputs, delimiter_dict, type_dict,
                          sampler_pool=ALL_SUBSTRING))
    if not _is_output_empty(expr, inputs):
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
    if not _is_output_empty(expr, inputs):
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
    if not _is_output_empty(expr, inputs):
      return expr


def random_get_token(inputs, delimiter_dict, type_dict):
  """Samples random GetToken expression."""
  del delimiter_dict
  while True:
    t = random_type(type_dict)
    indices = [i for i in dsl.INDEX if abs(i) <= type_dict[t]]
    i = random.choice(indices)

    expr = dsl.GetToken(t, i)
    if not _is_output_empty(expr, inputs):
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
  del delimiter_dict, type_dict
  expr = dsl.Trim()
  if not _is_output_empty(expr, inputs):
    return expr
  return None


def random_get_upto(inputs, delimiter_dict, type_dict):
  """Samples random GetUpto expression."""
  while True:
    is_delimiter = random.randint(1, 2)
    if delimiter_dict and is_delimiter == 1:
      r = random_delimiter(delimiter_dict)
    else:
      r = random_type(type_dict)

    expr = dsl.GetUpto(r)
    if not _is_output_empty(expr, inputs):
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
    if not _is_output_empty(expr, inputs):
      return expr


def random_get_first(inputs, delimiter_dict, type_dict):
  """Samples random GetFirst expression."""
  del delimiter_dict
  while True:
    t = random_type(type_dict)
    indices = [i for i in dsl.INDEX if abs(i) <= type_dict[t]]
    i = random.choice(indices)

    expr = dsl.GetFirst(t, i)
    if not _is_output_empty(expr, inputs):
      return expr


def random_get_all(inputs, delimiter_dict, type_dict):
  """Samples random GetAll expression."""
  del delimiter_dict
  while True:
    t = random_type(type_dict)

    expr = dsl.GetAll(t)
    if not _is_output_empty(expr, inputs):
      return expr


def random_substitute(inputs, delimiter_dict, type_dict):
  """Samples random Substitute expression."""
  del inputs, delimiter_dict
  t = random_type(type_dict)
  indices = [i for i in dsl.INDEX if abs(i) <= type_dict[t]]
  i = random.choice(indices)
  char = random.choice(dsl.CHARACTER)
  return dsl.Substitute(t, i, char)


def random_substitute_all(inputs, delimiter_dict, type_dict):
  """Samples random SubstituteAll expression."""
  del inputs, delimiter_dict
  t = random_type(type_dict)
  char = random.choice(dsl.CHARACTER)
  return dsl.SubstituteAll(t, char)


def random_remove(inputs, delimiter_dict, type_dict):
  """Samples random Remove expression."""
  del delimiter_dict
  types = list(type_dict.keys())
  random.shuffle(types)
  for t in types:  # Try all types in a random order.
    indices = [i for i in dsl.INDEX if abs(i) <= type_dict[t]]
    i = random.choice(indices)
    expr = dsl.Remove(t, i)
    if not _is_output_empty(expr, inputs):
      return expr

  return None  # No type worked.


def random_remove_all(inputs, delimiter_dict, type_dict):
  """Samples random RemoveAll expression."""
  del delimiter_dict
  types = list(type_dict.keys())
  random.shuffle(types)
  for t in types:  # Try all types in a random order.
    expr = dsl.RemoveAll(t)
    if not _is_output_empty(expr, inputs):
      return expr

  return None  # No type worked.


ALL_SUBSTRING = [
    random_sub_str,
    random_get_span,
    random_get_token,
    random_get_upto,
    random_get_from,
]

ALL_MODIFICATION = [
    random_to_case,
    random_replace,
    random_trim,
    random_get_first,
    random_get_all,
    random_substitute,
    random_substitute_all,
    random_remove,
    random_remove_all,
]


SAMPLER_POOL_ALL = [
    ALL_SUBSTRING,
    ALL_MODIFICATION,
    [random_compose_modification, random_compose_substring],
    random_const_str,
]

SAMPLER_POOL_NO_COMPOSE = [
    ALL_SUBSTRING,
    ALL_MODIFICATION,
    random_const_str,
]

SAMPLER_POOL_NO_COMPOSE_SUBSTRING = [
    ALL_SUBSTRING,
    ALL_MODIFICATION,
    random_compose_modification,
    random_const_str,
]

SAMPLER_POOL_ONLY_COMPOSE = [random_compose_modification,
                             random_compose_substring]

SAMPLER_POOL_MODIFY_OR_CONST = ALL_MODIFICATION + [random_const_str]
