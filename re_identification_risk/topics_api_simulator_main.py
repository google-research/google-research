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

"""Topics API Simulator Binary."""

from collections.abc import Sequence

from absl import app
from absl import flags

import apache_beam as beam
from apache_beam.options import pipeline_options
import tensorflow as tf

from re_identification_risk.topics_api_simulator import SimulateTopicsAPIFn

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_file_pattern',
    None,
    help=(
        "File pattern for the TFRecordIO files containing users' topic"
        ' profiles.'
    ),
)
flags.DEFINE_string(
    'output_file_pattern',
    None,
    help=(
        'Output file pattern for the TFRecordIO files containing the topics'
        ' simulator output.'
    ),
)
flags.DEFINE_string(
    'topics_taxonomy_file_pattern',
    None,
    help=(
        'File pattern for the text files containing the topic taxonomy. This is'
        ' a text file where each line is a unique integer topic ID. Every topic'
        ' in the taxonomy must appear exactly once.'
    ),
)
flags.DEFINE_integer(
    'num_epochs', None, help='The number of epochs in the simulation.'
)
flags.DEFINE_integer(
    'top_k', 5, help='The number of top-topics per user and epoch.'
)
flags.DEFINE_float(
    'prob_random_choice',
    0.05,
    help=(
        'The Topics API parameter controlling how often users report random'
        ' topics.'
    ),
)
flags.DEFINE_integer(
    'direct_num_workers', 0, help='Number of workers for DirectRunner.'
)


def main(unused_argv):

  def pipeline(root):
    # Create the simulator with parameters.
    simulator_fn = SimulateTopicsAPIFn(
        num_epochs=FLAGS.num_epochs,
        top_k=FLAGS.top_k,
        prob_random_choice=FLAGS.prob_random_choice,
    )
    # Load the topics taxonomy as a PCollection of ints.
    all_topics = (
        root
        | 'Load Topics Taxonomy'
        >> beam.io.ReadFromText(FLAGS.topics_taxonomy_file_pattern)
        | 'Parse Lines as Integers' >> beam.Map(int)
    )
    # Load and parse the tf.train.Examples encoding topic profiles.
    topic_profiles = root | 'Load Topic Profiles' >> beam.io.ReadFromTFRecord(
        FLAGS.input_file_pattern, coder=beam.coders.ProtoCoder(tf.train.Example)
    )
    # Apply the simulator to those topic profiles, with the collection of all
    # topics as a side-input.
    simulator_output = topic_profiles | 'Simulate Topics API' >> beam.ParDo(
        simulator_fn, beam.pvalue.AsList(all_topics)
    )
    # Write the simulator output as a TFRecordIO file.
    _ = simulator_output | 'Write Simulator Output' >> beam.io.WriteToTFRecord(
        FLAGS.output_file_pattern,
        coder=beam.coders.ProtoCoder(tf.train.Example),
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
