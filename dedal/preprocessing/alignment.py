# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utilities to manipulate and represent ground-truth alignments as strings."""

import re
from typing import List, Tuple


def alignment_from_gapped_sequences(
    gapped_sequence_x,
    gapped_sequence_y,
):
  """Extracts positions of match states in gapped sequences `x` and `y`."""
  matches = []
  ptr_x, ptr_y = 1, 1  # One-based indexing.
  for c_x, c_y in zip(gapped_sequence_x, gapped_sequence_y):
    if c_x.isupper() and c_y.isupper():
      matches.append((ptr_x, ptr_y))
    ptr_x += c_x.isalpha()
    ptr_y += c_y.isalpha()

  # Retrieves (one-based) starting position of the alignment.
  ali_start_x = matches[0][0] if matches else 0
  ali_start_y = matches[0][1] if matches else 0
  # Normalizes the indices of matching positions relative to the start of the
  # alignment.
  matches = [(x - ali_start_x, y - ali_start_y) for x, y in matches]

  return matches, (ali_start_x, ali_start_y)


def states_from_matches(
    matches,
    bos = 'S',
):
  """Generates (uncompressed) alignment `states` string from `matches`."""
  states = []
  for match, next_match in zip(matches[:-1], matches[1:]):
    states.append('M')
    num_gap_in_x = next_match[1] - match[1] - 1
    while num_gap_in_x:
      states.append('X')
      num_gap_in_x -= 1
    num_gap_in_y = next_match[0] - match[0] - 1
    while num_gap_in_y:
      states.append('Y')
      num_gap_in_y -= 1
  if matches:  # Adds last match of alignment, if non-empty.
    states.append('M')
  return ''.join([bos] + states)


def compress_states(states):
  """Run-length encoding compression of alignment `states` string."""
  chunks = []
  last_state = states[0]
  num_occurrences = 1
  for state in states[1:]:
    if state != last_state:
      chunks.append(f'{num_occurrences}{last_state}')
      last_state = state
      num_occurrences = 0
    num_occurrences += 1
  # Adds chunk corresponding to last stretch of identical chars.
  return ''.join(chunks + [f'{num_occurrences}{last_state}'])


def pid_from_matches(
    sequence_x,
    sequence_y,
    matches,
    ali_start_x = 1,
    ali_start_y = 1,
):
  """Computes the PID of sequences `x` and `y` from their alignment."""
  num_identical = 0
  alignment_length = 0
  for match, next_match in zip(matches[:-1], matches[1:]):
    c_x = sequence_x[ali_start_x + match[0] - 1]
    c_y = sequence_y[ali_start_y + match[1] - 1]
    num_identical += c_x == c_y
    alignment_length += 1
    alignment_length += next_match[0] - match[0] - 1
    alignment_length += next_match[1] - match[1] - 1
  if matches:  # Processes last match of alignment, if non-empty.
    c_x = sequence_x[ali_start_x + matches[-1][0] - 1]
    c_y = sequence_y[ali_start_y + matches[-1][1] - 1]
    num_identical += c_x == c_y
    alignment_length += 1
  return num_identical / max(1, alignment_length)


def gapped_sequence_from_cigar(
    sequence,
    cigar,
    seq_start = 1,
    gap = '.',
    is_reference = True,
):
  """Adds gaps to `sequence` according to input `cigar` string."""
  insert = 'D' if is_reference else 'I'
  delete = 'I' if is_reference else 'D'

  start_pos = seq_start - 1  # One-based indexing.
  chunk_len = None

  output_chunks = []
  for field in re.split(r'([DIM])', cigar):
    if field.isnumeric():
      chunk_len = int(field)
    elif field:  # skips empty strings.
      chunk_len = chunk_len or 1
      if field == delete:
        output_chunks.append(chunk_len * gap)
      elif field in ('M', insert):
        end_pos = start_pos + chunk_len
        output_chunks.append(sequence[start_pos:end_pos])
        start_pos = end_pos
      else:
        raise ValueError(f'CIGAR string {cigar} is invalid.')
      chunk_len = None

  return ''.join(output_chunks)
