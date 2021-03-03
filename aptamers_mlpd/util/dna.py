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

# Lint as: python3
"""Utilites for working with DNA data.
"""


class Error(Exception):
  pass


# Base ordering must be kept in sync with learning/custom_ops.cc
DNA_BASES = u"ATGC"

COMPLEMENT = {u"A": u"T", u"C": u"G", u"G": u"C", u"T": u"A"}

VALID_BASES_SET = frozenset(list(COMPLEMENT.keys()))


def has_invalid_bases(sequence):
  """Determines whether a sequence has invalid bases.

  Args:
    sequence: string DNA sequence
  Returns:
    True if the sequence contains letters other than A, C, G, T
  """
  return any(base not in VALID_BASES_SET for base in sequence)


def count_invalid_bases(sequence):
  """Returns the number of Ns and non-ACGTN letters present in the sequence.

  Ns, which are valid off the sequencer but not valid DNA, are counted
  separately from other letters.

  Args:
     sequence: the string DNA sequence to check
  Returns:
    A tuple of (int, int) where the first int is the number of Ns and the
    second int is the number of other non-ACGT letters.
  """
  count_n = 0
  count_other = 0

  for base in sequence:
    if base not in COMPLEMENT:
      if base == "N":
        count_n += 1
      else:
        count_other += 1

  return (count_n, count_other)


def reverse_complement(sequence):
  """Reverse complements a DNA sequence.

  Args:
    sequence: The string sequence to reverse complement.
  Returns:
    The string of the reverse complemented DNA, e.g. the reverse complement
      of the string ATACCG would be CGGTAT.
  Raises:
    Error: if the sequence contains letters other than A/T/C/G.
  """
  try:
    return "".join(COMPLEMENT[base] for base in reversed(sequence))
  except KeyError:
    raise Error("Illegal base in '%s', cannot reverse complement" % sequence)
