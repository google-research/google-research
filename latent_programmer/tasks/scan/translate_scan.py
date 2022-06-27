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

"""Translates SCAN inputs to outputs."""

import copy
import itertools

from typing import Iterator, List, Tuple, Union

from latent_programmer.tasks.scan import scan_vocab


# An object to represent a [[u]] variable.
U_VAR = '<U_VAR>'  # A string helps simplify type inference.
# A [[u]] variable must be one of these single input tokens.
U_VAR_SET = frozenset(['walk', 'look', 'run', 'jump'])
# Output tokens for turns.
LEFT_TURN = 'I_TURN_LEFT'
RIGHT_TURN = 'I_TURN_RIGHT'
# A mapping from input tuples to output tuples.
U_MAP = {
    (U_VAR,): (U_VAR,),
    ('turn', 'left'): (LEFT_TURN,),
    ('turn', 'right'): (RIGHT_TURN,),
    (U_VAR, 'left'): (LEFT_TURN, U_VAR),
    (U_VAR, 'right'): (RIGHT_TURN, U_VAR),
    ('turn', 'opposite', 'left'): (LEFT_TURN,) * 2,
    ('turn', 'opposite', 'right'): (RIGHT_TURN,) * 2,
    (U_VAR, 'opposite', 'left'): (LEFT_TURN, LEFT_TURN, U_VAR),
    (U_VAR, 'opposite', 'right'): (RIGHT_TURN, RIGHT_TURN, U_VAR),
    ('turn', 'around', 'left'): (LEFT_TURN,) * 4,
    ('turn', 'around', 'right'): (RIGHT_TURN,) * 4,
    (U_VAR, 'around', 'left'): (LEFT_TURN, U_VAR) * 4,
    (U_VAR, 'around', 'right'): (RIGHT_TURN, U_VAR) * 4,
}
# Input tokens for operators on [[x]] variables (i.e., sequences of tokens).
X_OPERATOR_SET = frozenset(['twice', 'thrice', 'and', 'after'])


def translate_u(u_var_token):
  return 'I_' + u_var_token.upper()


def is_output_token(token):
  return token.startswith('I_') or token == scan_vocab.SEP


def get_window_iterator(
    seq,
    window_size):
  zip_args = [seq[i:] for i in range(window_size)]
  return itertools.zip_longest(*zip_args)


def add_output_token(seq, token):
  add_output_tokens(seq, [token])


def add_output_tokens(seq,
                      tokens):
  if seq and isinstance(seq[-1], list):
    seq[-1].extend(tokens)
  else:
    seq.append(list(tokens))  # Make a copy!


def add_unknown(seq,
                unknown):
  if isinstance(unknown, list):
    add_output_tokens(seq, unknown)
  elif is_output_token(unknown):
    add_output_token(seq, unknown)
  else:
    seq.append(unknown)


def translate(input_tokens, add_separators):
  """Translates input tokens to output tokens."""
  # Input validation.
  for token in input_tokens:
    if token not in scan_vocab.INPUT_VOCAB:
      raise ValueError('Token {} not in input vocab'.format(token))

  # Copy the input to avoid modifying it.
  seq = copy.deepcopy(input_tokens)

  # Handle all rules except for those involving sequences of output tokens.
  new_seq = []  # type: List[Union[str, List[str]]]
  window_iterator = get_window_iterator(seq, 3)
  for x, y, z in window_iterator:
    triple = (U_VAR if x in U_VAR_SET else x, y, z)
    matched = False

    for i in range(3):
      tuple_to_match = triple[:3-i]  # Try progressively shorter tuples.
      if tuple_to_match in U_MAP:
        tuple_result = [translate_u(x) if e is U_VAR else e
                        for e in U_MAP[tuple_to_match]]
        new_seq.extend(tuple_result)
        for _ in range(len(tuple_to_match) - 1):
          next(window_iterator)  # Skip the rest of this processed tuple.
        matched = True
        break

    if matched:
      continue

    # These are the only terms unhandled for now.
    if x not in X_OPERATOR_SET:
      raise ValueError('Unhandled (x, y, z) = ({}, {}, {})'.format(x, y, z))
    new_seq.append(x)

  seq = new_seq

  # Group together all consecutive output tokens.
  new_seq = []
  for token in seq:
    if is_output_token(token):
      add_output_token(new_seq, token)
    else:
      if token not in X_OPERATOR_SET:
        raise ValueError('Token {} should be handled by now'.format(token))
      new_seq.append(token)
  seq = new_seq

  # Handle 'twice' and 'thrice'.
  new_seq = []
  window_iterator = get_window_iterator(seq, 2)
  for x, y in window_iterator:
    if y == 'twice':
      add_output_tokens(new_seq, x)
      add_output_tokens(new_seq, x)
      next(window_iterator)
    elif y == 'thrice':
      add_output_tokens(new_seq, x)
      add_output_tokens(new_seq, x)
      add_output_tokens(new_seq, x)
      next(window_iterator)
    else:
      add_unknown(new_seq, x)
  seq = new_seq

  # Handle 'and'.
  new_seq = []
  for x in seq:
    if x == 'and':
      if add_separators:
        add_output_token(new_seq, scan_vocab.SEP)
    else:
      add_unknown(new_seq, x)
  seq = new_seq

  # Handle 'after'.
  new_seq = []
  for x in seq[::-1]:
    if x == 'after':
      if add_separators:
        add_output_token(new_seq, scan_vocab.SEP)
    else:
      add_unknown(new_seq, x)
  seq = new_seq

  # At this point, seq is a list containing a single list of all output tokens.
  # Flatten this list.
  if len(seq) != 1:
    raise ValueError('Expected length 1: {}'.format(seq))
  seq = seq[0]

  # Output validation.
  for token in seq:
    if token not in scan_vocab.OUTPUT_VOCAB:
      raise ValueError('Token {} not in output vocab'.format(token))

  return seq
