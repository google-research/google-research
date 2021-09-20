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

r"""Parse predicted vs actual family membership from the HMM model output.

The HMMER suite of programs provides utilities to build HMM profiles from
alignments of protein sequences, and evaluate the predicted class membership for
sets of unaligned protein sequences. HMMER can be installed by running
apt-get install hmmer

The HMMER manual can be found at
http://eddylab.org/software/hmmer/Userguide.pdf

Given a set of PFAM hmmsearch output files built from a set of unaligned test
sequences, this script extracts output statistics from these text files.
Specifically it records all sequences that score below the reporting threshold
which by default is set to an E-value <= 10. If no such sequences were found,
then a sentinel value is written.

The output files analyzed in this script are produced with a command like"
```
hmmsearch --tblout pfam_output/PF00131.19.txt pfam_hmm/PF00131.19.hmm
testseqs.fasta
```

A sample HMMER output file can be found in hmmer_test.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import tempfile

from absl import logging
import hmmer_utils
import pandas as pd
import parallel
import tensorflow.compat.v1 as tf
from tqdm import tqdm


# Manually verified 12 threads performs better than anything smaller.
# 12 is the number of cores on most engineering desktops.
_NUM_HMMER_PROCESSES_TO_RUN_IN_PARALLEL = 12


def run_hmmbuild_for_seqalign(seqalign_file_path, profile_file_path):
  """Write output of hmmsearch binary of align all queries with all sequences.

  This runs with the command
  hmmbuild hmmfile msafile

  Args:
    seqalign_file_path: string. Path to aligned family fasta file (msafile).
    profile_file_path: string. Filename of hmm profile to be made by hmmbuild.
  """
  subprocess.check_output(
      ['hmmbuild', '--amino', profile_file_path, seqalign_file_path])


def run_hmmbuild_write_profiles(train_align_dir, hmm_dir):
  """Run hmmbuild over all train alignments and write profiles.

  Args:
    train_align_dir: string. Directory of aligned sequence files for training.
    hmm_dir: string. Path to write .hmm files created by HMMbuild.
  """
  if not tf.io.gfile.IsDirectory(hmm_dir):
    logging.warn('Making directory %s', hmm_dir)
    tf.io.gfile.MakeDirs(hmm_dir)

  list_of_args_to_function = []
  for seqalign_file in os.listdir(train_align_dir):
    family_name = hmmer_utils.get_name_from_seq_filename(seqalign_file)
    train_seqalign_file_path = os.path.join(train_align_dir, seqalign_file)
    profile_file_path = os.path.join(hmm_dir, family_name) + '.hmm'
    list_of_args_to_function.append(
        dict(
            seqalign_file_path=train_seqalign_file_path,
            profile_file_path=profile_file_path))

  logging.info('Building hmms for %d families.', len(list_of_args_to_function))
  parallel.RunInParallel(
      run_hmmbuild_for_seqalign,
      list_of_args_to_function,
      _NUM_HMMER_PROCESSES_TO_RUN_IN_PARALLEL,
      cancel_futures=True)


def run_hmmsearch_for_profile(profile_file_path, test_sequence_file,
                              request_all_match_output):
  """Run hmmsearch binary of all test sequences against an hmm profile.

  Args:
    profile_file_path: string. Filename of .hmm profile made by hmmbuild.
    test_sequence_file: string. Fasta file containing all test sequences.
    request_all_match_output: boolean. If True, run hmmsearch with --max
      to turn off all filtering.

  Returns:
    string. Output of running the binary hmmsearch.
  """
  command = [
      'hmmsearch',
      '--tblout',
      '/dev/stdout',
      '-o',
      '/dev/null',
  ]
  if request_all_match_output:
    command.append('--max')

  command.extend([profile_file_path, test_sequence_file])

  return subprocess.check_output(command)


def write_hmmsearch_outputs_for_one_profile(hmm_profile, test_sequence_file,
                                            request_all_match_output):
  """Returns HMMEROutput from hmmsearch for the hmm against the test file.

  Args:
    hmm_profile: string. Filename of file created by hmmsearch for a set of test
      sequences.
    test_sequence_file: string. Path to fasta file of unaligned test sequences.
    request_all_match_output: boolean. If True, run hmmsearch with --max
      to turn off all filtering.

  Returns:
    list of HMMEROutputs.
  """
  output = run_hmmsearch_for_profile(
      hmm_profile,
      test_sequence_file,
      request_all_match_output=request_all_match_output)
  hmm_profile_family = hmmer_utils.get_family_from(hmm_profile)
  hmmer_output = hmmer_utils.parse_hmmer_output(output, hmm_profile_family)

  return hmmer_output


def write_hmmsearch_outputs_for_all_profiles(
    hmm_dir, test_sequence_file, parsed_output, request_all_match_output):
  """Run hmmsearch over testseqs for each profile in hmm_dir, write to csv file.

  The csv content is:
  sequence_name, predicted_label, true_label, score

  Where sequence_name is the uniprot identifier, including domain indices,
  and true and predicted label are pfam family accession ids.

  Args:
    hmm_dir: string. Path to .hmm files created by HMMbuild.
    test_sequence_file: string. Path to fasta file of unaligned test sequences.
    parsed_output: string. csv file to which to write parsed HMMsearch outputs.
    request_all_match_output: boolean. If True, run hmmsearch with --max
      to turn off all filtering.
  """
  input_hmmer_files = [
      os.path.join(hmm_dir, hmm_output_file)
      for hmm_output_file in os.listdir(hmm_dir)
  ]

  list_of_kwargs_to_function = [
      dict(
          hmm_profile=hmm_profile,
          test_sequence_file=test_sequence_file,
          request_all_match_output=request_all_match_output)
      for hmm_profile in input_hmmer_files
  ]

  logging.info('Running hmmsearch for %d families.', len(input_hmmer_files))
  hmmsearch_results = parallel.RunInParallel(
      write_hmmsearch_outputs_for_one_profile,
      list_of_kwargs_to_function,
      _NUM_HMMER_PROCESSES_TO_RUN_IN_PARALLEL,
      cancel_futures=True)

  logging.info('Writing results to file %s', parsed_output)
  with tf.io.gfile.GFile(parsed_output, 'w') as parsed_output_first_pass_file:
    parsed_output_first_pass_file.write(
        ','.join(hmmer_utils.HMMER_OUTPUT_CSV_COLUMN_HEADERS) + '\n')

    for search_result in tqdm(hmmsearch_results):
      to_write = '\n'.join([str(x.format_as_csv()) for x in search_result])
      parsed_output_first_pass_file.write(to_write + '\n')


def make_second_pass(hmm_dir, parsed_output_first_pass,
                     parsed_output_second_pass, test_sequence_file):
  """Runs hmmsearch with higher specificity on seqs that have no predictions.

  - Computes set of sequences for which the first pass did not produce any
    predictions.
  - Reruns hmmsearch with --max argument to get even more predictions (slower,
    but more correct).

  Args:
    hmm_dir: string. Path to .hmm files created by HMMbuild.
    parsed_output_first_pass: string. csv file where the first pass of
      lower-specificity hmmsearch results have been written.
    parsed_output_second_pass: string. csv file to which to write parsed
      hmmsearch outputs (of those sequences missed by the first pass).
    test_sequence_file: string. Path to fasta file of unaligned test sequences.
  """
  with tf.io.gfile.GFile(parsed_output_first_pass, 'r') as pred_file:
    first_pass_predictions = pd.read_csv(pred_file)

  sequences_with_no_prediction = hmmer_utils.sequences_with_no_prediction(
      hmmer_predictions=first_pass_predictions,
      all_sequence_names=hmmer_utils.all_sequence_names_from_fasta_file(
          test_sequence_file))

  if not sequences_with_no_prediction:
    logging.info('Second pass not needed: all sequences had predictions in the '
                 'first pass.')
    return

  to_run_in_second_pass = hmmer_utils.filter_fasta_file_by_sequence_name(
      test_sequence_file,
      acceptable_sequence_names=sequences_with_no_prediction)

  (_, file_input_to_second_pass) = tempfile.mkstemp(
      '.fasta', 'file_input_to_second_pass')
  logging.info(
      'Using %s as temp file for writing input sequences for second pass.',
      file_input_to_second_pass)

  with tf.io.gfile.GFile(file_input_to_second_pass, 'w') as second_pass_file:
    for el in to_run_in_second_pass:
      second_pass_file.write(el)

  logging.info(
      'Making a second set of predictions for %d sequences using '
      'hmmsearch with higher sensitivity.', len(sequences_with_no_prediction))
  write_hmmsearch_outputs_for_all_profiles(
      hmm_dir,
      file_input_to_second_pass,
      parsed_output_second_pass,
      request_all_match_output=True)


def postprocess_first_and_second_passes(
    parsed_output_first_pass, parsed_output_second_pass, final_output):
  """Join and deduplicate first and second passes, and write to final_output.

  Also removes any hmmer_utils.NO_SEQUENCE_MATCH_SEQUENCE_NAME_SENTINEL entries.

  Args:
    parsed_output_first_pass: string. csv file to which to write the first pass
      of lower-specificity hmmsearch results will be written.
    parsed_output_second_pass: string. csv file to which to write parsed
      hmmsearch outputs (of those sequences missed by the first pass).
    final_output: string. csv file to which to write the final, postprocessed
      results.
  """
  logging.info('Processing first and second passes into final predictions.')
  with tf.io.gfile.GFile(parsed_output_first_pass, 'r') as pred_file:
    first_pass_predictions = pd.read_csv(pred_file)
  with tf.io.gfile.GFile(parsed_output_second_pass, 'r') as pred_file:
    second_pass_predictions = pd.read_csv(pred_file)

  all_predictions = pd.concat([first_pass_predictions, second_pass_predictions],
                              ignore_index=True)
  all_predictions = all_predictions[all_predictions.sequence_name != hmmer_utils
                                    .NO_SEQUENCE_MATCH_SEQUENCE_NAME_SENTINEL]
  top_pred_dfs_for_each_seq = (
      hmmer_utils.yield_top_el_by_score_for_each_sequence_name(all_predictions))

  top_pred_for_each_seq = pd.concat(
      top_pred_dfs_for_each_seq, ignore_index=True)

  with tf.io.gfile.GFile(final_output, 'w') as parsed_output_file:
    top_pred_for_each_seq.to_csv(parsed_output_file, index=False)


def run_hmmer_on_data(train_align_dir, hmm_dir, test_sequence_file,
                      parsed_output):
  """Run hmmer analysis on a train/test split of the protein data.

  Args:
    train_align_dir: string. Directory of aligned sequence files for training.
    hmm_dir: string. Path to .hmm profiles created by HMMbuild.
    test_sequence_file: string. Path to fasta file of unaligned test sequences.
    parsed_output: string. csv file to which to write the final, postprocessed
      results.
  """
  _, parsed_output_first_pass = tempfile.mkstemp('.csv',
                                                 'parsed_output_first_pass')
  _, parsed_output_second_pass = tempfile.mkstemp('.csv',
                                                  'parsed_output_second_pass')

  run_hmmbuild_write_profiles(train_align_dir, hmm_dir)
  write_hmmsearch_outputs_for_all_profiles(
      hmm_dir,
      test_sequence_file,
      parsed_output_first_pass,
      request_all_match_output=False)

  make_second_pass(
      hmm_dir=hmm_dir,
      parsed_output_first_pass=parsed_output_first_pass,
      parsed_output_second_pass=parsed_output_second_pass,
      test_sequence_file=test_sequence_file)

  postprocess_first_and_second_passes(parsed_output_first_pass,
                                      parsed_output_second_pass, parsed_output)
