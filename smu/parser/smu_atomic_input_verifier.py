# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Verifies that we can correctly generate atomic2 input files."""

import os

from absl import app
from absl import flags
from absl import logging

from tensorflow.io import gfile
from smu.parser import smu_parser_lib
from smu.parser import smu_writer_lib

flags.DEFINE_string(
    'input_glob', None, 'Glob of .dat files to read. '
    'These files are expected to be in the SMU file format provided by Uni Basel.'
)
flags.DEFINE_string(
    'atomic2_input_dir', None,
    'Directory containing .inp files named like x07_c2n2f3h3.253852.001.inp  '
    'These are the desired outputs')
flags.DEFINE_string('output_dir', None,
                    'If given, given to write files with mismatches')

flags.mark_flag_as_required('input_glob')
flags.mark_flag_as_required('atomic2_input_dir')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  atomic2_writer = smu_writer_lib.Atomic2InputWriter()

  file_count = 0
  molecule_count = 0
  mismatches = 0

  for filepath in gfile.glob(FLAGS.input_glob):
    logging.info('Processing file %s', filepath)
    file_count += 1
    smu_parser = smu_parser_lib.SmuParser(filepath)
    for molecule, _ in smu_parser.process_stage2():
      molecule_count += 1

      actual_contents = atomic2_writer.process(molecule, 0)

      expected_fn = atomic2_writer.get_filename_for_atomic2_input(
          molecule, None)
      with gfile.GFile(os.path.join(FLAGS.atomic2_input_dir,
                                    expected_fn)) as expected_f:
        expected_contents = expected_f.readlines()

      try:
        smu_writer_lib.check_dat_formats_match(expected_contents,
                                               actual_contents.splitlines())
      except smu_writer_lib.DatFormatMismatchError as e:
        mismatches += 1
        print(e)
        if FLAGS.output_dir:
          with gfile.GFile(
              os.path.join(
                  FLAGS.output_dir,
                  atomic2_writer.get_filename_for_atomic2_input(
                      molecule, topo_idx=None)), 'w') as f:
            f.write(actual_contents)

  status_str = ('COMPLETE: Read %d files, %d molecules, %d mismatches\n' %
                (file_count, molecule_count, mismatches))

  logging.info(status_str)
  print(status_str)


if __name__ == '__main__':
  app.run(main)
