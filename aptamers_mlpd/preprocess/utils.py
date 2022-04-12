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

"""Utilites for working with Aptitude data, including basic FASTQ functions."""

import collections

# Google internal
import gfile


from ..util import dna

# Classification of read pairs.
READ_TOO_LONG = "too_long"
READ_TOO_SHORT = "too_short"
LOW_QUALITY = "low_quality"
BAD_READ_PAIR = "bad_pair"
OKAY = "okay"

BASES = [u"A", u"C", u"G", u"T"]

SMALL_THRESHOLD = 100000


class Error(Exception):
  pass


class Read(
    collections.namedtuple("Read",
                           ["title", "title_aux", "sequence", "quality"])):
  """NamedTuple to hold the information in one FASTQ sequence record.

  A Read is one record from a FASTQ file and represents one read off a
  sequencer. All the values are strings. The record contains:

  (1) the title of the read (unique identifier),
  (2) any auxiliary information from the title line (e.g., barcode info)
  (3) the DNA sequence as recorded by the sequencer, and
  (4) the per-base quality of the sequencing.

  The per-base quality is a string of the same length as the DNA sequence,
  which can be converted to an integer based on the ascii + an offset.
  The numbers are generally between 0 and 40 and the exact encoding and
  offset depends on the version of the sequencer.
  See https://en.wikipedia.org/wiki/FASTQ_format
  """
  pass


FASTQ_HIGHEST_QUALITY = chr(40 + 64)  # h


def write_fastq(forward_path, reverse_path, fastq_reads):
  """Write forward and reverse reads into a pair of FASTQ files.

  Args:
    forward_path: path to which to save forward reads.
    reverse_path: path to which to save reverse reads.
    fastq_reads: iterable of Read objects.
  """
  with gfile.Open(forward_path, "w") as forward:
    with gfile.Open(reverse_path, "w") as reverse:
      for read in fastq_reads:
        comp_sequence = dna.reverse_complement(read.sequence)
        forward.write("@%s\n%s\n+\n%s\n" %
                      (read.title, read.sequence, read.quality))
        reverse.write("@%s\n%s\n+\n%s\n" %
                      (read.title, comp_sequence, read.quality[::-1]))
