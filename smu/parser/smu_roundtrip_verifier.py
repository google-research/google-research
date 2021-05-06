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
"""Verifies that we can correctly roundtrip SMU7 data files through protos.

smu_parser_lib and smu_writer_lib allows us to read and write the .dat file
format. This scripts read in multiple .dat files and verifies that we correctly
regenerate them. Any mismatches are written to new files.
"""

import collections
import enum

from absl import app
from absl import flags
from absl import logging

from tensorflow.io import gfile
from smu.parser import smu_parser_lib
from smu.parser import smu_writer_lib


flags.DEFINE_string(
    'input_glob', None,
    'Glob of .dat files to read. '
    'These files are expected to be in the SMU file format provided by Uni Basel.'
)
flags.DEFINE_string(
    'output_stem', None,
    'Filestem to be used for writing entries that differ. '
    'Files with _original.dat and _regenerated.dat suffixes will be created'
)
flags.DEFINE_enum('stage', 'stage2', ['stage1', 'stage2'],
                  'Whether to expect files in stage1 or stage2 format')

flags.mark_flag_as_required('input_glob')
flags.mark_flag_as_required('output_stem')

FLAGS = flags.FLAGS


class Outcome(enum.Enum):
  SUCCESS = 1
  MISMATCH = 2
  PARSE_ERROR_KNOWN = 3
  PARSE_ERROR_UNKNOWN = 4


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  smu_writer = smu_writer_lib.SmuWriter(annotate=False)

  # output_files maps from Outcome to the a pair of file handle
  output_files = {}
  output_files[Outcome.SUCCESS] = (
      gfile.GFile(FLAGS.output_stem + '_success_original.dat', 'w'),
      gfile.GFile(FLAGS.output_stem + '_success_regen.dat', 'w'))
  output_files[Outcome.MISMATCH] = (
      gfile.GFile(FLAGS.output_stem + '_mismatch_original.dat', 'w'),
      gfile.GFile(FLAGS.output_stem + '_mismatch_regen.dat', 'w'))
  output_files[Outcome.PARSE_ERROR_KNOWN] = (
      gfile.GFile(FLAGS.output_stem + '_parse_error_known_original.dat', 'w'),
      gfile.GFile(FLAGS.output_stem + '_parse_error_known_regen.dat', 'w'))
  output_files[Outcome.PARSE_ERROR_UNKNOWN] = (
      gfile.GFile(FLAGS.output_stem + '_parse_error_unknown_original.dat', 'w'),
      gfile.GFile(FLAGS.output_stem + '_parse_error_unknown_regen.dat', 'w'))

  file_count = 0
  conformer_count = 0
  outcome_counts = collections.Counter()

  for filepath in gfile.Glob(FLAGS.input_glob):
    logging.info('Processing file %s', filepath)
    file_count += 1
    smu_parser = smu_parser_lib.SmuParser(filepath)
    if FLAGS.stage == 'stage1':
      process_fn = smu_parser.process_stage1
    else:
      process_fn = smu_parser.process_stage2
    for conformer, orig_contents_list in process_fn():
      conformer_count += 1

      outcome = None

      if isinstance(conformer, Exception):
        if isinstance(conformer, smu_parser_lib.SmuKnownError):
          outcome = Outcome.PARSE_ERROR_KNOWN
        else:
          outcome = Outcome.PARSE_ERROR_UNKNOWN
        regen_contents = '{}\n{}: {} {}\n'.format(smu_parser_lib.SEPARATOR_LINE,
                                                  conformer.conformer_id,
                                                  type(conformer).__name__,
                                                  str(conformer))
      else:
        if FLAGS.stage == 'stage1':
          regen_contents = smu_writer.process_stage1_proto(conformer)
        else:
          regen_contents = smu_writer.process_stage2_proto(conformer)
        try:
          smu_writer_lib.check_dat_formats_match(orig_contents_list,
                                                 regen_contents.splitlines())
          outcome = Outcome.SUCCESS
        except smu_writer_lib.DatFormatMismatchError as e:
          outcome = Outcome.MISMATCH
          print(e)

      outcome_counts[outcome] += 1
      output_files[outcome][0].write('\n'.join(orig_contents_list) + '\n')
      output_files[outcome][1].write(regen_contents)

  for file_orig, file_regen in output_files.values():
    file_orig.close()
    file_regen.close()

  def outcome_status(outcome):
    if conformer_count:
      percent = outcome_counts[outcome] / conformer_count * 100
    else:
      percent = float('nan')
    return '%5.1f%% %7d %s \n' % (percent, outcome_counts[outcome],
                                  str(outcome))

  status_str = ('COMPLETE: Read %d files, %d conformers\n' %
                (file_count, conformer_count) +
                outcome_status(Outcome.SUCCESS) +
                outcome_status(Outcome.PARSE_ERROR_KNOWN) +
                outcome_status(Outcome.MISMATCH) +
                outcome_status(Outcome.PARSE_ERROR_UNKNOWN))

  logging.info(status_str)
  print(status_str)


if __name__ == '__main__':
  app.run(main)
