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

"""Preprocessing for NarrativeQA data."""
import os

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.options import pipeline_options
import nltk

from readtwice.models.narrative_qa import preprocess_lib



# IO
flags.DEFINE_string('input_qaps', None, 'qaps.csv input file.')
flags.DEFINE_string('input_documents', None, 'documents.csv input file.')
flags.DEFINE_string('data_split', None, 'Data set split: train, test or valid')
flags.DEFINE_string('stories_dir', None, 'Stories input.')
flags.DEFINE_string('summaries_path', None, 'Summaries input.')

flags.DEFINE_string('output_prefix', None, 'Prefix for the output files.')
flags.DEFINE_integer('output_num_shards', 32, 'Number of output shards.')
flags.DEFINE_bool('generate_summaries', False,
                  'Whether the generated examples should contain summaries.')
flags.DEFINE_bool('generate_answers', False,
                  'Whether the generated examples should contain answers.')
flags.DEFINE_float('min_rouge_l_oracle_score', 0.35,
                   'Min threshold for accepting oracle answer.')
# Tokenization
flags.DEFINE_string('spm_model_path', None, 'Path to sentence piece model.')
flags.DEFINE_string('nltk_data_path', None, '')

# Document settings.
flags.DEFINE_integer('num_blocks_per_example', 128,
                     'Number of blocks per a single tf.train.Example.')
flags.DEFINE_integer('block_overlap_length', 128,
                     'Between block overlap length.')
flags.DEFINE_integer('block_length', 512, 'Length of a single block')
flags.DEFINE_integer('max_num_annotations', 32,
                     'Maximum number of annotations per block')
flags.DEFINE_integer('padding_token_id', 0, 'ID of the padding token.')
flags.DEFINE_integer('cls_token_id', 2, 'ID of [CLS] token.')

# By default, we're using token ID 4, which corresponds to <sep_0> here
flags.DEFINE_integer(
    'sep_token_id', 3,
    'ID of token to separate question from the rest of the document.')

flags.DEFINE_integer(
    'direct_num_workers', 0,
    'Number of workers to use for the Beam DirectRunner. '
    'Increasing this should speed up example generation, '
    'but DirectRunner appears to run out of memory quickly '
    'when using more workers. 0 is automatically using all available workers.')

FLAGS = flags.FLAGS


def main(unused_argv):
  pipeline = preprocess_lib.get_pipeline(
      input_qaps=FLAGS.input_qaps,
      input_documents=FLAGS.input_documents,
      data_split=FLAGS.data_split,
      stories_dir=FLAGS.stories_dir,
      summaries_path=FLAGS.summaries_path,
      spm_model_path=FLAGS.spm_model_path,
      num_blocks_per_example=FLAGS.num_blocks_per_example,
      block_overlap_length=FLAGS.block_overlap_length,
      block_length=FLAGS.block_length,
      max_num_annotations_per_block=FLAGS.max_num_annotations,
      padding_token_id=FLAGS.padding_token_id,
      cls_token_id=FLAGS.cls_token_id,
      sep_token_id=FLAGS.sep_token_id,
      generate_answers=FLAGS.generate_answers,
      generate_summaries=FLAGS.generate_summaries,
      min_rouge_l_oracle_score=FLAGS.min_rouge_l_oracle_score,
      nltk_data_path=FLAGS.nltk_data_path,
      output_prefix=FLAGS.output_prefix,
      output_num_shards=FLAGS.output_num_shards)
  logging.info('Initializing NLTK data path from %s.', FLAGS.nltk_data_path)
  nltk.data.path.append(FLAGS.nltk_data_path)
  nltk.download('punkt', download_dir=FLAGS.nltk_data_path)
  nltk.download('averaged_perceptron_tagger', download_dir=FLAGS.nltk_data_path)
  nltk.download('maxent_ne_chunker', download_dir=FLAGS.nltk_data_path)

  # run the pipeline:
  options = pipeline_options.PipelineOptions(
      runner='DirectRunner',
      direct_running_mode='multi_processing',
      direct_num_workers=FLAGS.direct_num_workers)
  p = beam.Pipeline(options=options)
  pipeline(p)
  p.run().wait_until_finish()


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'input_qaps', 'input_documents', 'stories_dir', 'data_split',
      'output_prefix', 'nltk_data_path'
  ])
  app.run(main)
