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
r"""A command line tool to parse a SMU file and convert it to a protocol buffer.

This tool is intended to do a one-to-one conversion from a SMU file to the
corresponding protocol buffer.

Example:
./smu_parser \
  --alsologtostderr \
  --input_file=<path to Basel .dat file> \
  --output_file=<path to protobuf output>
"""

from absl import app
from absl import flags
from absl import logging

from tensorflow.io import gfile

from google.protobuf import text_format

from smu import dataset_pb2
from smu.parser import smu_parser_lib

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file. This file is expected to be in the SMU file format provided by Uni Basel.'
)
flags.DEFINE_string(
    'output_file', None, 'Path to the output file. ' +
    'This file will be a protocol buffer in text format. ' +
    'If empty, outputs to stdout.')

flags.mark_flag_as_required('input_file')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  number_of_parse_errors = 0
  multiple_conformers = dataset_pb2.MultipleConformers()

  parser = smu_parser_lib.SmuParser(FLAGS.input_file)
  for e, orig_contents in parser.process_stage2():
    if isinstance(e, Exception):
      number_of_parse_errors += 1
      print('Parse error for:\n{}\n{}'.format(
          orig_contents[1], e))
    else:
      multiple_conformers.conformers.append(e)

  if FLAGS.output_file:
    logging.info('Writing protobuf to file %s.', FLAGS.output_file)
    with gfile.GFile(FLAGS.output_file, 'w') as f:
      f.write(
          '# proto-file: third_party/google_research/google_research/smu/dataset.proto\n'
      )
      f.write('# proto-message: MultipleConformers\n')
      f.write(text_format.MessageToString(multiple_conformers))
  else:
    print(
        '# proto-file: third_party/google_research/google_research/smu/dataset.proto'
    )
    print('# proto-message: MultipleConformers')
    print(text_format.MessageToString(multiple_conformers), end='')

  return number_of_parse_errors

if __name__ == '__main__':
  app.run(main)
