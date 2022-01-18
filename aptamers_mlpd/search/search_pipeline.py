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

# Lint as: python3
"""Functions for pipeline functions for picking aptamers using trained models.
"""

import os

# Google internal
import gfile


# Set a seed so we can re-run the selection and get the same results.
RANDOM_SEED = 12152007


class Error(Exception):
  pass


def seq_to_array_seq(seq, array_len=60):
  """Adjust a sequence to make it ready for the microarray.

  Sequences are provided 5' to 3' but printed with the 3' end attached. If
  the sequence is less than the maximum allowed on the array (60 for our
  standard Agilent microarrays), we buffer the sequence out to 60 bases
  using T's to raise it up off the plate.

  Args:
    seq: String sequence to put on the microarray.
    array_len: Maximum sequence length to print on the microarray.
  Raises:
    Error: If seq is longer than the array_len.
  Returns:
    String sequence ready for putting in an Agilent txt file for array ordering.
  """
  if len(seq) > array_len:
    raise Error('Sequence is too long for the array. Max of %d but %s is %d.' %
                (array_len, seq, len(seq)))
  return '%s%s' % (seq, 'T' * (array_len - len(seq)))


def collapse_and_write(choice_protos,
                       output_base,
                       array_name,
                       array_prefix,
                       copies_per_seq=10):
  """Writes choice_protos out and a collapsed file of sequences for an array.

  Args:
    choice_protos: List of Choice protos to save.
    output_base: String base directory to save the files in.
    array_name: String name of the file to save the sequences. The file will
      be in the right format to be uploaded to the Agilent website.
    array_prefix: The string to append to the front of every spot name,
      used to keep probe ids unique on the Agilent website.
    copies_per_seq: Integer number of spots to put on the array for each
      sequence.
  Returns:
    Set of unique sequences in the choice_protos.
  """
  count_proto = 0
  count_seq = 0
  count_spot = 0

  # There will be duplicate sequences in choice_protos. Write each seq once.
  previous_seqs = set()
  with gfile.GFile(os.path.join(output_base, array_name), 'w') as f:
    for p in choice_protos:
      count_proto += 1
      seq = p.aptamer_sequence
      if seq in previous_seqs:
        continue
      previous_seqs.add(seq)
      count_seq += 1
      for i in range(copies_per_seq):
        count_spot += 1
        probe_id = '%s_%s_%d_of_%d' % (array_prefix, seq, i + 1, copies_per_seq)
        f.write(
            ('%s\t%s\n' % (probe_id, seq_to_array_seq(seq))).encode('utf-8'))
  print(('There are %d protos with %d unique sequences, yielding %d array '
         'spots' % (count_proto, count_seq, count_spot)))

  return previous_seqs


def write_oligos(sequences, output_base, oligo_pool_name, fwd_primer,
                 rev_primer):
  """Writes sequences out to a file format for ordering an oligo pool.

  Args:
    sequences: An iteratable collection of string sequences.
    output_base: String base directory to save the files in.
    oligo_pool_name: String name of the file to save the sequences to send
      to CustomArray to order an oligo pool. These sequences do not have
      duplicates printed and have forward and reverse primers.
    fwd_primer: String to be appended to the front of the sequence for the
      oligo pool sequences
    rev_primer: String to be appended to the back of the sequence for the
      oligo pool sequences. Final oligo pool sequence for an aptamer sequence
      'seq' will be '%s%s%s' % (fwd_primer, seq, rev_primer).
  """

  # For CustomArray, we just want a list of sequences but include primers
  count_oligo = 0
  with gfile.GFile(os.path.join(output_base, oligo_pool_name), 'w') as f:
    for seq in sequences:
      count_oligo += 1
      f.write(('%s%s%s\n' % (fwd_primer, seq, rev_primer)).encode('utf-8'))
  print('Wrote %d oligos to a txt file.' % (count_oligo))
