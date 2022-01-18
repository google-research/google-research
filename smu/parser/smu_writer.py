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
r"""A command line tool to write a protocol buffer to a file in SMU file format.

This tool is intended to faithfully reproduce the Basel University SMU file
format.

Example:
./smu_writer \
  --alsologtostderr \
  --input_file=<path to protobuf file> \
  --output_file=<path to Basel .dat output file>
"""

from absl import app
from absl import flags
from absl import logging
from tensorflow.io import gfile

from google.protobuf import text_format
from smu import dataset_pb2
from smu.parser.smu_writer_lib import SmuWriter

flags.DEFINE_string('input_file', None,
                    'Path to the input file in SMU protobuf text format.')
flags.DEFINE_string(
    'output_file', None, 'Path to the output file. ' +
    'This file will be a protocol buffer in text format.' +
    'If empty, outputs to stdout.')
flags.DEFINE_bool(
    'annotate', False,
    'Whether to generate annotations in the output file with proto field names')

flags.mark_flag_as_required('input_file')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  smu_proto = dataset_pb2.MultipleConformers()
  with gfile.GFile(FLAGS.input_file) as f:
    raw_proto = f.read()
  text_format.Parse(raw_proto, smu_proto)
  smu_writer = SmuWriter(FLAGS.annotate)
  contents = ''.join(
      smu_writer.process_stage2_proto(conformer)
      for conformer in smu_proto.conformers
  )
  if FLAGS.output_file:
    logging.info('Writing smu7 conformers to .dat file %s.', FLAGS.output_file)
    with open(FLAGS.output_file, 'w') as f:
      f.write(contents)
  else:
    print(contents, end='')

if __name__ == '__main__':
  app.run(main)
