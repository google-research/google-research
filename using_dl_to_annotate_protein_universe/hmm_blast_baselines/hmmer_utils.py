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

r"""Utilities for working with the HMMER software package.

The HMMER suite of programs provides utilities to build HMM profiles from
alignments of protein sequences, and evaluate the predicted class membership for
sets of unaligned protein sequences. The HMMER suite can be installed by running
apt-get install hmmer

The HMMER manual can be found at
http://eddylab.org/software/hmmer/Userguide.pdf

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
from Bio.SeqIO import FastaIO
import pandas as pd
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import util

NO_SEQUENCE_MATCH_SCORE_SENTINEL = 0.
NO_SEQUENCE_MATCH_DOMAIN_EVALUE_SENTINEL = -1.
NO_SEQUENCE_MATCH_FAMILY_NAME_SENTINEL = 'PF00000.0'
NO_SEQUENCE_MATCH_SEQUENCE_NAME_SENTINEL = 'no_sequence/0-0'
DATAFRAME_SCORE_NAME_KEY = 'score'
DATAFRAME_DOMAIN_EVALUE_NAME_KEY = 'domain_evalue'

HMMER_OUTPUT_CSV_COLUMN_HEADERS = util.PREDICTION_FILE_COLUMN_NAMES + [
    DATAFRAME_DOMAIN_EVALUE_NAME_KEY
]


def get_sequence_name_from(seq_name_and_family):
  """Get sequence name from concatenated sequence name and family string.

  Args:
    seq_name_and_family: string. Of the form `sequence_name`_`family_accession`,
      like OLF1_CHICK/41-290_PF00001.20. Output would be OLF1_CHICK/41-290.

  Returns:
    string. Sequence name.
  """
  return '_'.join(seq_name_and_family.split('_')[0:2])


def get_family_name_from(seq_name_and_family):
  """Get family accession from concatenated sequence name and family string.

  Args:
    seq_name_and_family: string. Of the form `sequence_name`_`family_accession`,
      like OLF1_CHICK/41-290_PF00001.20. Assumes the family does not have an
      underscore.

  Returns:
    string. PFam family accession.
  """
  return seq_name_and_family.split('_')[-1]


def get_name_from_seq_filename(seq_filename):
  """Get sequence name from the name of an individual sequence fasta filename.

  Args:
    seq_filename: string. Of the form 'sequence_name'.fasta, like
      OLF1_CHICK/41-290.fasta.

  Returns:
    string. Sequence name.
  """
  return seq_filename.split('.')[0]


def get_family_from(hmm_filepath):
  """Get family name from the name of an individual family hmm filename.

  Args:
    hmm_filepath: string. Of the form '~/family_name'.hmm, like
      hmm_files/PF00001.21.hmm.

  Returns:
    string. Family name.
  """
  hmm_filename = hmm_filepath.split('/')[-1]
  return '.'.join(hmm_filename.split('.')[0:2])


class HMMEROutput(
    collections.namedtuple('HMMEROutput', HMMER_OUTPUT_CSV_COLUMN_HEADERS)):
  """Parsed tblout output from HMMER.

  Args:
    sequence_name: str. the query sequence.
    true_label: str. the family that the sequence belongs to.
    predicted_label: str. the predicted family.
    score: float. the score of this match.
  """
  __slots__ = ()

  def format_as_csv(self):
    """Convert HMMEROutput to csv."""
    return ','.join([
        self.sequence_name,
        self.true_label,
        self.predicted_label,
        str(self.score),
        str(self.domain_evalue),
    ])


def parse_hmmer_output(hmmer_output, query_identifier):
  """Return HMMEROutput from the text output of a hmmer binary.

  Args:
    hmmer_output: string. The output of running a hmmer binary.
    query_identifier: string. Identity of the query sequence or profile family.

  Returns:
    list of HMMEROutputs. If none, returns a 'no result' HMMEROutput; this will
    be populated differently for phmmer and hmmer, because their use cases
    differ. If hmmer_output is mal-formed, returns [].
  """
  outputs = []
  all_lines = hmmer_output.split('\n')
  hmmer_output_lines = [line for line in all_lines if line]
  # Remove blank lines
  for i, line in enumerate(hmmer_output_lines):
    if ('--- full sequence ----' in line) and (
        '# Program:' in hmmer_output_lines[i + 4]):
      # In this case, there was no match above the inclusion threshold, so
      # we say there isn't a prediction at all.
      sequence = NO_SEQUENCE_MATCH_SEQUENCE_NAME_SENTINEL
      true_family = NO_SEQUENCE_MATCH_FAMILY_NAME_SENTINEL
      predicted_family = query_identifier
      score = NO_SEQUENCE_MATCH_SCORE_SENTINEL
      domain_evalue = NO_SEQUENCE_MATCH_DOMAIN_EVALUE_SENTINEL
      outputs.append(
          HMMEROutput(
              sequence_name=sequence,
              true_label=true_family,
              predicted_label=predicted_family,
              score=score,
              domain_evalue=domain_evalue,
          ))
      return outputs
    else:
      # There is some output, find it.
      if '#' in line:
        # This is a comment line, not an output line.
        pass
      else:
        # This is an output line. The sequence name is found in the 1st field,
        # formatted as: MT4_CANLF/1-62_PF00131.19
        seq_name_and_family = line.split()[0]
        sequence = get_sequence_name_from(seq_name_and_family)
        true_family = get_family_name_from(seq_name_and_family)
        domain_evalue = float(line.split()[4])
        score = float(line.split()[5])
        predicted_family = line.split()[2]
        outputs.append(
            HMMEROutput(
                sequence_name=sequence,
                true_label=true_family,
                predicted_label=predicted_family,
                score=score,
                domain_evalue=domain_evalue))
  return outputs


def _get_sentinel_phmmer_output(query_identifier):
  return HMMEROutput(
      sequence_name=get_sequence_name_from(query_identifier),
      true_label=get_family_name_from(query_identifier),
      predicted_label=NO_SEQUENCE_MATCH_FAMILY_NAME_SENTINEL,
      score=NO_SEQUENCE_MATCH_SCORE_SENTINEL,
      domain_evalue=NO_SEQUENCE_MATCH_DOMAIN_EVALUE_SENTINEL,
  )


def _report_hmmer_outputs_for_set_difference(seen_identifiers, all_identifiers):
  """Return HMMEROutputs for identifiers that were not seen in phmmer output.

  Args:
    seen_identifiers: iterable of string. All the query identifiers seen in the
      output of phmmer.
    all_identifiers: iterable of string. All fasta entry identifiers passed into
      phmmer. These should be formatted seqName_actualFamily.

  Returns:
    List of HMMEROutput. Sentinel values for query identifiers in
    all_identifiers that are not in seen_identifiers.
  """
  outputs = []
  for identifier in all_identifiers:
    if identifier not in seen_identifiers:
      outputs.append(_get_sentinel_phmmer_output(identifier))
  return outputs


def _phmmer_output_line_to_hmmer_output(line):
  """Convert line of output of phmmer to a HMMEROutput.

  Args:
    line: Line from running phmmer --tblout. See section "Tabular output
      formats" of
      http://eddylab.org/software/hmmer/Userguide.pdf.

  Returns:
    HMMEROutput.
  """
  # This is an output line.
  # The query sequence name is found in the 3rd field, formatted as:
  # MT4_CANLF/1-62_PF00131.19
  query_seq_name_and_family = line.split()[2]
  query_sequence = get_sequence_name_from(query_seq_name_and_family)
  true_family = get_family_name_from(query_seq_name_and_family)

  domain_evalue = float(line.split()[4])
  score = float(line.split()[5])
  # The matching sequence name is found in the 1st field, formatted as:
  # MT4_CANLF/1-62_PF00131.19
  matching_seq_name_and_family = line.split()[0]
  predicted_family = get_family_name_from(matching_seq_name_and_family)

  return HMMEROutput(
      sequence_name=query_sequence,
      true_label=true_family,
      predicted_label=predicted_family,
      score=score,
      domain_evalue=domain_evalue,
  )


def parse_phmmer_output(hmmer_output, query_identifiers):
  """Return list of HMMEROutput from stdout of phmmer.

  Args:
    hmmer_output: stdout of phmmer --tblout.
    query_identifiers: all fasta entry identifiers passed into phmmer. These
      should be formatted seqName_actualFamily.

  Returns:
    list of HMMEROutput. Entries in query_identifiers that were not in the
      hmmer_output (those for which there was not a match) are reported with
      sentinel values (see _get_sentinel_phmmer_output).
  """
  query_identifiers_seen = set()

  outputs = []
  all_lines = hmmer_output.split('\n')
  for line in all_lines:
    is_output_line = (
        line and  # Not a blank line.
        # Lines beginning with '#' are comment lines, not output lines.
        '#' not in line)

    if is_output_line:
      hmmer_output = _phmmer_output_line_to_hmmer_output(line)

      query_identifiers_seen.add(hmmer_output.sequence_name + '_' +
                                 hmmer_output.true_label)

      outputs.append(hmmer_output)

  return outputs + _report_hmmer_outputs_for_set_difference(
      seen_identifiers=query_identifiers_seen,
      all_identifiers=query_identifiers)


def yield_top_el_by_score_for_each_sequence_name(hmmer_predictions):
  """Return the predictions with the top full sequence scores."""
  logging.info('Picking top prediction by score in hmmer predictions.')

  grouped_duplicates = hmmer_predictions.groupby(
      util.DATAFRAME_SEQUENCE_NAME_KEY)
  for _, group in tqdm(grouped_duplicates, position=0):
    max_el = group.loc[group[DATAFRAME_SCORE_NAME_KEY].idxmax()]
    yield pd.DataFrame(
        data={
            util.PREDICTED_LABEL_KEY: [max_el.predicted_label],
            util.TRUE_LABEL_KEY: [max_el.true_label],
            util.DATAFRAME_SEQUENCE_NAME_KEY: [max_el.sequence_name],
            DATAFRAME_SCORE_NAME_KEY: [max_el.score],
            DATAFRAME_DOMAIN_EVALUE_NAME_KEY: [max_el.domain_evalue]
        })


def sequences_with_no_prediction(hmmer_predictions, all_sequence_names):
  """Returns sequence names that are not in hmmer_predictions.

  Args:
    hmmer_predictions: pd.DataFrame with the column
      util.DATAFRAME_SEQUENCE_NAME_KEY.
    all_sequence_names: iterable of string.

  Returns:
    set of string.
  """
  sequence_names_with_predictions = hmmer_predictions.sequence_name.values
  all_sequence_names_without_family = set(
      get_sequence_name_from(s) for s in all_sequence_names)
  return set(all_sequence_names_without_family) - set(
      sequence_names_with_predictions)


def all_sequence_names_from_fasta_file(input_fasta_file_name):
  """Returns all sequence names from a fasta file.

  Args:
    input_fasta_file_name: string.

  Returns:
    list of string.
  """
  with tf.io.gfile.GFileText(input_fasta_file_name) as input_file:
    return [
        get_sequence_name_from(protein_name_incl_family)
        for protein_name_incl_family, _ in FastaIO.SimpleFastaParser(input_file)
    ]


def filter_fasta_file_by_sequence_name(input_fasta_file_name,
                                       acceptable_sequence_names):
  """Yield only entries from a fasta file that are in acceptable_sequence_names.

  Args:
    input_fasta_file_name: string. This file should contain fasta entries that
      are formatted seqName_actualFamily, as above.
    acceptable_sequence_names: iterable of string. This set just seqName (no
      actualFamily, as with `input_fasta_file_name`).

  Yields:
    strings, each of which is an entry for a fasta file.
  """
  acceptable_sequence_names = set(acceptable_sequence_names)
  with tf.io.gfile.GFileText(input_fasta_file_name) as input_file:
    for protein_name, sequence in FastaIO.SimpleFastaParser(input_file):
      if get_sequence_name_from(protein_name) in acceptable_sequence_names:
        yield '>' + protein_name + '\n' + sequence + '\n'


def parse_domtblout(domtblout_text):
  """Parses hmmer output from flag --domtblout.

  Args:
    domtblout_text: output from hmmer. See test.

  Returns:
    pd.DataFrame with columns
    full_sequence_name (str) - e.g. pfamseq accession without start and end
    sequence_name (str) - full_sequence_name with start and end
    sequence_start (int) - 1-index based (not 0-index)
    sequence_end (int) - 1-index based (not 0-index)
    predicted_label (str) - name of hmm profile, e.g. PF00001
    domain_evalue_score (float) - independent evalue for the domain
      (not the whole sequence!)
    domain_bit_score (float) - bit score for the domain
      (not the whole sequence!)
  """
  domtblout_rows = collections.defaultdict(list)
  for line in domtblout_text.split('\n'):
    if line.startswith('#') or not line:
      continue
    split = line.split()

    # Parse columns.
    full_sequence_name = split[0]
    sequence_start = int(split[17])
    sequence_end = int(split[18])
    sequence_name = f'{full_sequence_name}/{sequence_start}-{sequence_end}'

    hmm_label = split[4]

    domain_evalue_score = float(split[12])
    domain_bit_score = float(split[13])

    # Assign columns values.
    domtblout_rows['full_sequence_name'].append(full_sequence_name)
    domtblout_rows['sequence_name'].append(sequence_name)
    domtblout_rows['sequence_start'].append(sequence_start)
    domtblout_rows['sequence_end'].append(sequence_end)

    domtblout_rows['predicted_label'].append(hmm_label)

    domtblout_rows['domain_evalue_score'].append(domain_evalue_score)
    domtblout_rows['domain_bit_score'].append(domain_bit_score)
  domtblout_df = pd.DataFrame(domtblout_rows)
  return domtblout_df
