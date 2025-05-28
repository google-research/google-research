# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Beam token_ids reidentification binary."""

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options

from re_identification_risk import reidentification


# Define command-line flags for the input file, output file and other
# parameters.
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_file_pattern', '', 'File pattern of tfrecordio file(s) to read.'
)
flags.DEFINE_string(
    'weight_path', '', 'File pattern of weights for asymmetric hamming attack.'
)
flags.DEFINE_string('output_dir', '', 'Output directory.')
flags.DEFINE_integer('query_size', 10240, 'The number of sampled queries.')
flags.DEFINE_integer(
    'direct_num_workers', 0, 'Number of workers for DirectRunner.'
)


# Create a reidentification pipeline and run it.
def main(unused_argv):
  pipeline = reidentification.reidentification(
      FLAGS.input_file_pattern,
      FLAGS.output_dir,
      FLAGS.query_size,
      FLAGS.weight_path,
  )

  options = pipeline_options.PipelineOptions(
      runner='DirectRunner',
      direct_running_mode='multi_processing',
      direct_num_workers=FLAGS.direct_num_workers,
  )
  root_pipeline = beam.Pipeline(options=options)
  pipeline(root_pipeline)
  root_pipeline.run().wait_until_finish()


if __name__ == '__main__':
  app.run(main)
