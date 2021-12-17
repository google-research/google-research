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
# pylint:disable=line-too-long
r"""Beam job to try a bunch of hparams.

"""
# pylint:enable=line-too-long

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam


from non_semantic_speech_benchmark.eval_embedding.sklearn import train_and_eval_sklearn as utils

flags.DEFINE_string('train_glob', None, 'Glob for train data.')
flags.DEFINE_string('eval_glob', None, 'Glob for eval data.')
flags.DEFINE_string('test_glob', None, 'Glob for test dir.')
flags.DEFINE_string('output_file', None, 'Output filename.')
flags.DEFINE_list('embedding_list', None, 'Python list of embedding names.')
flags.DEFINE_string('label_name', None, 'Name of label to use.')
flags.DEFINE_list('label_list', None, 'Python list of possible label values.')
flags.DEFINE_string('speaker_id_name', None, '`None`, or speaker ID field.')
flags.DEFINE_string('save_model_dir', None, 'If not `None`, save sklearn '
                    'models in this directory.')
flags.DEFINE_string(
    'save_predictions_dir', None, 'If not `None`, write numpy '
    'array of predictions on train, eval, and test into this '
    'directory.')
flags.DEFINE_enum('eval_metric', 'accuracy', [
    'accuracy', 'balanced_accuracy', 'equal_error_rate',
    'unweighted_average_recall', 'auc', 'dprime'
], 'Which metric to compute and report.')
flags.DEFINE_string(
    'comma_escape_char', '?',
    'Sometimes we want commas to appear in `embedding_modules`, '
    '`embedding_names`, or `module_output_key`. However, commas get split out '
    'in Googles Python `DEFINE_list`. We compromise by introducing a special '
    'character, which we replace with commas.')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Validate flags and setup directories.
  utils.validate_flags(FLAGS.train_glob, FLAGS.eval_glob, FLAGS.test_glob,
                       FLAGS.output_file)

  # Generate experiment parameters based on flags.
  exp_params = utils.experiment_params(
      FLAGS.embedding_list,
      FLAGS.speaker_id_name,
      FLAGS.label_name,
      FLAGS.label_list,
      FLAGS.train_glob,
      FLAGS.eval_glob,
      FLAGS.test_glob,
      FLAGS.save_model_dir,
      FLAGS.save_predictions_dir,
      FLAGS.eval_metric,
  )

  # Make and run beam pipeline.
  beam_options = None

  logging.info('Starting to create flume pipeline...')
  with beam.Pipeline(beam_options) as root:
    _ = (
        root
        | 'MakeCollection' >> beam.Create(exp_params)
        |
        'CalcScores' >> beam.Map(lambda d: (d, utils.train_and_get_score(**d)))
        | 'FormatText' >> beam.Map(utils.format_text_line)
        | 'Reshuffle' >> beam.Reshuffle()
        | 'WriteOutput' >> beam.io.WriteToText(FLAGS.output_file, num_shards=1))


if __name__ == '__main__':
  flags.mark_flags_as_required(['train_glob', 'eval_glob', 'output_file',
                                'embedding_list', 'label_name', 'label_list'])
  app.run(main)
