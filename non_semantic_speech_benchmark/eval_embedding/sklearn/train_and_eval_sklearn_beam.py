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

import itertools
import os
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam


from non_semantic_speech_benchmark import file_utils

from non_semantic_speech_benchmark.eval_embedding.sklearn import models
from non_semantic_speech_benchmark.eval_embedding.sklearn import train_and_eval_sklearn

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
flags.DEFINE_enum('eval_metric', 'accuracy',
                  ['accuracy', 'balanced_accuracy', 'equal_error_rate',
                   'unweighted_average_recall'],
                  'Which metric to compute and report.')
flags.DEFINE_bool('fast_write', True,
                  'Writes to multiple shards if `True`, otherwise just one.')

FLAGS = flags.FLAGS


def format_text_line(k_v):
  """Convert params and score to human-readable format."""
  p, (eval_score, test_score) = k_v
  cur_elem = ', '.join([
      f'Eval score: {eval_score}',
      f'Test score: {test_score}',
      f'Embed: {p["embedding_name"]}',
      f'Label: {p["label_name"]}',
      f'Model: {p["model_name"]}',
      f'L2 normalization: {p["l2_normalization"]}',
      f'Speaker normalization: {p["speaker_id_name"] is not None}',
      '\n'
  ])
  logging.info('Finished formatting: %s', cur_elem)
  return cur_elem


def main(unused_argv):
  assert file_utils.Glob(FLAGS.train_glob), FLAGS.train_glob
  assert file_utils.Glob(FLAGS.eval_glob), FLAGS.eval_glob
  assert file_utils.Glob(FLAGS.test_glob), FLAGS.test_glob

  # Create output directory if it doesn't already exist.
  outdir = os.path.dirname(FLAGS.output_file)
  file_utils.MaybeMakeDirs(outdir)

  # Enumerate the configurations we want to run.
  exp_params = []
  model_names = models.get_sklearn_models().keys()
  for elem in itertools.product(*[FLAGS.embedding_list, model_names]):
    def _params_dict(
        l2_normalization, speaker_id_name=FLAGS.speaker_id_name, elem=elem):
      return {
          'embedding_name': elem[0],
          'model_name': elem[1],
          'label_name': FLAGS.label_name,
          'label_list': FLAGS.label_list,
          'train_glob': FLAGS.train_glob,
          'eval_glob': FLAGS.eval_glob,
          'test_glob': FLAGS.test_glob,
          'l2_normalization': l2_normalization,
          'speaker_id_name': speaker_id_name,
          'save_model_dir': FLAGS.save_model_dir,
          'eval_metric': FLAGS.eval_metric,
      }
    exp_params.append(_params_dict(l2_normalization=True))
    exp_params.append(_params_dict(l2_normalization=False))
    if FLAGS.speaker_id_name is not None:
      exp_params.append(
          _params_dict(l2_normalization=True, speaker_id_name=None))
      exp_params.append(
          _params_dict(l2_normalization=False, speaker_id_name=None))

  # Make and run beam pipeline.
  beam_options = None

  logging.info('Starting to create flume pipeline...')
  with beam.Pipeline(beam_options) as root:
    score = (root
             | 'MakeCollection' >> beam.Create(exp_params)
             | 'CalcScores' >> beam.Map(
                 lambda d: (d, train_and_eval_sklearn.train_and_get_score(**d)))
             | 'FormatText' >> beam.Map(format_text_line)
            )
    if not FLAGS.fast_write:
      score = score | 'Reshuffle' >> beam.Reshuffle()
    _ = score | 'WriteOutput' >> beam.io.WriteToText(
        FLAGS.output_file, num_shards=0 if FLAGS.fast_write else 1)


if __name__ == '__main__':
  flags.mark_flags_as_required(['train_glob', 'eval_glob', 'output_file',
                                'embedding_list', 'label_name', 'label_list'])
  app.run(main)
