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

r"""Creates a CSV file with context sequences for Pfam-A seed entries."""

import collections
import re
import time
from typing import List, Mapping, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from dedal import vocabulary
from dedal.data import specs
from dedal.preprocessing import buffered_line_reader


flags.DEFINE_multi_string(
    'input_file', None,
    'Path to input dat file, as provided by UniprotKB.')
flags.DEFINE_string(
    'data_dir', None,
    '<data_dir>/iid_ood_clans contains output of create_splits.py.')
flags.DEFINE_string(
    'output_file', None,
    'Path to output csv file.')
flags.mark_flag_as_required('input_file')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('output_file')
FLAGS = flags.FLAGS


# Preprocessed Pfam-A seed 34.0 config.
TASK = 'iid_ood_clans'
KEYS = ('id', 'start', 'end')
SPLITS = ('train', 'iid_validation', 'ood_validation', 'iid_test', 'ood_test')

# buffered_line_reader config.
SEP = b'//\n'
BUFFER_SIZE = 1024

# IDentification lines are formatted as
#   ID   EntryName Status; SequenceLength AA.
# where EntryName consists of two alphanumeric strings joined by an underscore
# char, Status can be Reviewed or Unreviewed and SequenceLength is a string of
# base 10 digits.
ID_RE = 'ID'
ENTRY_NAME_RE = r'([0-9a-zA-Z]+_[0-9a-zA-Z]+)'
STATUS_RE = r'[Reviewed|Unreviewed]+;'
SEQLEN_RE = r'([\d]+) AA.'
ID_REGEX = re.compile(r'[\s]+'.join(
    [ID_RE, ENTRY_NAME_RE, STATUS_RE, SEQLEN_RE]))

# Field names for header in output csv file.
OUTPUT_FIELDS = ['pfam_id', 'uniprot_starts', 'full_sequence']

# Type aliases
Pfam = Tuple[Sequence[int], Sequence[int], Sequence[str]]
IndicesFromPfamId = Mapping[str, Sequence[int]]
Uniprot = Tuple[Sequence[str], Sequence[str]]


def find_all(s, t):
  """Returns one-based start position of all occurrences of t in s."""
  offset = 0
  starts = []
  start = s.find(t, offset)
  while start != -1:
    starts.append(start + 1)  # Uses one-based indexing, as Pfam does.
    offset = start + 1
    start = s.find(t, offset)
  return starts


def load_pfam():
  """Loads all Pfam-A seed data from TFRecords.

  Returns:
    A tuple `(pfam, indices_from_pfam_id)` such that:
      + `pfam` is itself a tuple `((pfam_starts, pfam_ends, pfam_sequences)`
        of lists having the values of the fields 'start', 'end' and 'sequence'
        for all Pfam-A seed 34.0 TFRecords.
      + `indices_from_pfam_id` is a Python dict mapping UniprotKB entry names
        (field 'id' in the TFRecords) to the indices of examples having that ID.
        Note that IDs are *not* in general unique, since a protein sequence
        might contribute multiple, distinct Pfam entries. The 'start' and 'end'
        fields need to be considered to achieve uniqueness.
  """
  data = collections.defaultdict(list)
  ds_loader = specs.make_pfam34_loader(
      root_dir=FLAGS.data_dir, sub_dir='', extra_keys=KEYS, task=TASK)
  for split in SPLITS:
    logging.info('Loading %s Pfam-A seed 34.0 split...', split)
    split_data = collections.defaultdict(list)
    for ex in ds_loader.load(split).prefetch(tf.data.AUTOTUNE):
      split_data['id'].append(ex['id'].numpy().decode('utf-8'))
      split_data['start'].append(ex['start'].numpy())
      split_data['end'].append(ex['end'].numpy())
      split_data['seq'].append(
          vocabulary.alternative.decode(ex['sequence'].numpy()))
    for k, v in split_data.items():
      data[k].append(v)
  result = {k: sum(v, []) for k, v in data.items()}

  indices_from_pfam_id = collections.defaultdict(list)
  for i, pfam_id in enumerate(result['id']):
    indices_from_pfam_id[pfam_id].append(i)
  logging.info('Found %d unique IDs among %d Pfam-A seed sequences',
               len(indices_from_pfam_id), len(result['id']))
  return (result['start'], result['end'], result['seq']), indices_from_pfam_id


def parse_uniprotkb(indices_from_pfam_id):
  """Parses UniprotKB dat file, keeping entries present in Pfam-A seed 34.0.

  Args:
    indices_from_pfam_id: a map from UniprotKB entry names to indices of Pfam-A
      seed examples, as provided by `load_pfam`.

  Returns:
    A tuple `(uniprot_ids, uniprot_sequences)` such that:
      + `uniprot_ids` is a list of UniprotKB entry names.
      + `uniprot_sequences` is a list of UniprotKB protein sequences, matching
        the order of `uniprot_ids`.
    UniprotKB entries not represented in Pfam-A seed 34.0 are silently
    discarded.
  """
  # Parses Uniprot dat file, keeping IDs and sequences of entries present in
  # Pfam-A seed.
  uniprot_ids, uniprot_sequences = [], []
  for input_file in FLAGS.input_file:
    with tf.io.gfile.GFile(input_file, 'rb') as f:
      line_reader = buffered_line_reader.BufferedLineReader(
          f, sep=SEP, buffer_size=BUFFER_SIZE)
      for entry in line_reader:
        id_line, entry = entry.split('\n', 1)
        g = ID_REGEX.match(id_line)
        # Skips malformed / incomplete entries.
        if g is not None:
          uniprot_id, seq_len = g.group(1), int(g.group(2))
          # Parses sequence data lines iff the entry is part of Pfam-A seed.
          if uniprot_id in indices_from_pfam_id:
            seq_entry = entry.split('SQ   SEQUENCE', 1)[-1]
            seq_entry = seq_entry.split('\n', 1)[-1]
            uniprot_sequence = ''.join([line.strip().replace(' ', '')
                                        for line in seq_entry.split('\n')])
            if len(uniprot_sequence) != seq_len:
              raise ValueError(
                  f'Length for entry {uniprot_id} ({len(uniprot_sequence)}) '
                  f'does not match ID line ({seq_len})!')
            uniprot_ids.append(uniprot_id)
            uniprot_sequences.append(uniprot_sequence)
  logging.info(
      'Found %d matching entries in %s (%d unique).',
      len(uniprot_ids), ', '.join(FLAGS.input_file), len(set(uniprot_ids)))
  return uniprot_ids, uniprot_sequences


def write_output(pfam,
                 indices_from_pfam_id,
                 uniprot):
  """Writes UniprotKB context sequences to output CSV file.

  The output file will contain one row for each Pfam-A seed entry. Each row will
  have three fields:
    + 'pfam_id' is a unique identifier of the form <entry name>/<start>-<end>,
      where `entry name` refers to the UniprotKB context sequence and `start`
      and `end` refer to the position of the Pfam-A seed (sub)sequence in the
      UniprotKB context sequence. Note, however, that these do not always
      coincide with those computed empirically (see next field).
    + 'uniprot_starts' is a semicolon-delimited list of integers. It describes
      the empirical start positions of the Pfam-A seed (sub)sequence in the
      UniprotKB context sequence. It can be empty if the (sub)sequence was not
      found in its context (sequence mismatches) or contain more than one
      element if the (sub)sequence occurs more than once (repeats).
    + 'full_sequence' contains the UniprotKB context sequence. Note that several
      Pfam-A seed entries might share the same context sequence.

  Args:
    pfam: 'start', 'end' and 'sequence' fields of TFRecords, as returned by
      'load_pfam'.
    indices_from_pfam_id: a map from UniprotKB entry names to indices of Pfam-A
      seed examples, as returned by `load_pfam`.
    uniprot: UniprotKB entry names and sequences, as returned by
      `parse_uniprotkb`.
  """
  pfam_starts, pfam_ends, pfam_sequences = pfam
  uniprot_ids, uniprot_sequences = uniprot

  logging.info('Writing output file %s...', FLAGS.output_file)

  n_pfam_entries_found = 0
  n_sequence_mismatches = 0
  n_repeats = 0
  n_start_mismatches = 0
  with tf.io.gfile.GFile(FLAGS.output_file, 'w') as f:
    f.write(','.join(OUTPUT_FIELDS) + '\n')
    for uniprot_id, uniprot_sequence in zip(uniprot_ids, uniprot_sequences):
      for idx in indices_from_pfam_id[uniprot_id]:
        pfam_start, pfam_end = pfam_starts[idx], pfam_ends[idx]
        pfam_sequence = pfam_sequences[idx]

        uniprot_starts = find_all(uniprot_sequence, pfam_sequence)

        n_pfam_entries_found += 1
        if uniprot_starts:
          n_repeats += len(uniprot_starts) > 1
          n_start_mismatches += pfam_start not in uniprot_starts
        else:
          n_sequence_mismatches += 1

        pfam_id = f'{uniprot_id}/{pfam_start}-{pfam_end}'
        uniprot_starts = ';'.join([str(i) for i in uniprot_starts])
        fields = [pfam_id, uniprot_starts, uniprot_sequence]
        f.write(','.join(fields) + '\n')

  logging.info('Finished writing %d entries to output file.',
               n_pfam_entries_found)

  logging.info('%d / %d Pfam-A seed entries have mismatching sequences.',
               n_sequence_mismatches, n_pfam_entries_found)
  logging.info('%d / %d Pfam-A seed entries have repeats.',
               n_repeats, n_pfam_entries_found)
  logging.info('%d / %d Pfam-A seed entries have mismatching starts.',
               n_start_mismatches, n_pfam_entries_found)


def main(_):
  start = time.time()
  pfam, indices_from_pfam_id = load_pfam()
  uniprot = parse_uniprotkb(indices_from_pfam_id)
  write_output(pfam, indices_from_pfam_id, uniprot)
  runtime = time.time() - start
  logging.info('Total time elapsed: %.3f seconds.', runtime)


if __name__ == '__main__':
  app.run(main)
