# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Uses Python Beam to compute the multivariate Gaussian."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import app
from absl import flags

import create_embeddings_beam

flags.DEFINE_string('input_files', None,
                    'File containing a list of all input audio files.')
flags.DEFINE_string(
    'tfrecord_input', None, 'Path to a tfrecord file. The audio clips should be'
    'wraped tf.examples as float featues using the feature'
    'key specified by --feature_key.')
flags.DEFINE_string(
    'feature_key', 'audio/reference/raw_audio',
    'Tf.example feature that contains the samples that are '
    'to be processed.')
flags.DEFINE_string('embeddings', None, 'The embeddings output file path.')
flags.DEFINE_string('stats', None, 'The stats output file path.')
flags.DEFINE_string('model_ckpt', 'data/vggish_model.ckpt',
                    'The model checkpoint that should be loaded.')
flags.DEFINE_integer('model_embedding_dim', 128,
                     'The model dimension of the models emedding layer.')
flags.DEFINE_integer('model_step_size', 8000,
                     'Number of samples between each extraced windown.')

flags.mark_flags_as_mutual_exclusive(['input_files', 'tfrecord_input'],
                                     required=True)
FLAGS = flags.FLAGS


ModelConfig = collections.namedtuple(
    'ModelConfig', 'model_ckpt embedding_dim step_size')


def main(unused_argv):
  if not FLAGS.embeddings and not FLAGS.stats:
    raise ValueError('No output provided. Please specify at least one of '
                     '"--embeddings" or "--stats".')
  pipeline = create_embeddings_beam.create_pipeline(
      tfrecord_input=FLAGS.tfrecord_input,
      files_input_list=FLAGS.input_files,
      feature_key=FLAGS.feature_key,
      embedding_model=ModelConfig(
          model_ckpt=FLAGS.model_ckpt,
          embedding_dim=FLAGS.model_embedding_dim,
          step_size=FLAGS.model_step_size),
      embeddings_output=FLAGS.embeddings,
      stats_output=FLAGS.stats)
  result = pipeline.run()
  result.wait_until_finish()


if __name__ == '__main__':
  app.run(main)
