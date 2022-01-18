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

"""Compute realized predictions for a dataset."""

from absl import app
from absl import flags
from absl import logging
from official.common import distribute_utils
from official.nlp.bert import configs
import tensorflow as tf

from felix import felix_flags  # pylint: disable=unused-import
from felix import predict
from felix import utils

FLAGS = flags.FLAGS


def batch_generator():
  """Produces batches for felix to predict."""
  source_batch = []
  target_batch = []
  for sources, target in utils.yield_sources_and_targets(
      FLAGS.predict_input_file, FLAGS.input_format):

    source_batch.append(
        FLAGS.special_glue_string_for_joining_sources.join(sources))
    target_batch.append(target)
    if len(source_batch) == FLAGS.predict_batch_size:
      yield source_batch, target_batch
      source_batch = []
      target_batch = []

  if source_batch:
    yield source_batch, target_batch


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.use_open_vocab:
    raise ValueError('Currently only use_open_vocab=True is supported')

  label_map = utils.read_label_map(FLAGS.label_map_file)
  bert_config_tagging = configs.BertConfig.from_json_file(
      FLAGS.bert_config_tagging)
  bert_config_insertion = configs.BertConfig.from_json_file(
      FLAGS.bert_config_insertion)
  if FLAGS.tpu is not None:
    cluster_resolver = distribute_utils.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    with strategy.scope():
      predictor = predict.FelixPredictor(
          bert_config_tagging=bert_config_tagging,
          bert_config_insertion=bert_config_insertion,
          model_tagging_filepath=FLAGS.model_tagging_filepath,
          model_insertion_filepath=FLAGS.model_insertion_filepath,
          vocab_file=FLAGS.vocab_file,
          label_map=label_map,
          sequence_length=FLAGS.max_seq_length,
          max_predictions=FLAGS.max_predictions_per_seq,
          do_lowercase=FLAGS.do_lower_case,
          use_open_vocab=FLAGS.use_open_vocab,
          is_pointing=FLAGS.use_pointing,
          insert_after_token=FLAGS.insert_after_token,
          special_glue_string_for_joining_sources=FLAGS
          .special_glue_string_for_joining_sources)
  else:
    predictor = predict.FelixPredictor(
        bert_config_tagging=bert_config_tagging,
        bert_config_insertion=bert_config_insertion,
        model_tagging_filepath=FLAGS.model_tagging_filepath,
        model_insertion_filepath=FLAGS.model_insertion_filepath,
        vocab_file=FLAGS.vocab_file,
        label_map_file=FLAGS.label_map_file,
        sequence_length=FLAGS.max_seq_length,
        max_predictions=FLAGS.max_predictions_per_seq,
        do_lowercase=FLAGS.do_lower_case,
        use_open_vocab=FLAGS.use_open_vocab,
        is_pointing=FLAGS.use_pointing,
        insert_after_token=FLAGS.insert_after_token,
        special_glue_string_for_joining_sources=FLAGS
        .special_glue_string_for_joining_sources)

  source_batch = []
  target_batch = []
  num_predicted = 0
  with tf.io.gfile.GFile(FLAGS.predict_output_file, 'w') as writer:
    for source_batch, target_batch in batch_generator():
      predicted_tags, predicted_inserts = predictor.predict_end_to_end_batch(
          source_batch)
      num_predicted += len(source_batch)
      logging.log_every_n(logging.INFO, f'{num_predicted} predicted.', 200)
      for source_input, target_output, predicted_tag, predicted_insert in zip(
          source_batch, target_batch, predicted_tags, predicted_inserts):
        writer.write(f'{source_input}\t{predicted_tag}\t{predicted_insert}\t'
                     f'{target_output}\n')


if __name__ == '__main__':
  flags.mark_flag_as_required('predict_input_file')
  flags.mark_flag_as_required('input_format')
  flags.mark_flag_as_required('predict_output_file')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('vocab_file')
  app.run(main)
