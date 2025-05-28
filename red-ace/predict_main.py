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

"""Computes predictions for a JSON dataset."""

import json

from absl import app
from absl import flags
from absl import logging

import predict
import redace_config as configs
import redace_flags  # pylint: disable=unused-import
import tokenization
import utils

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  predictor = predict.RedAcePredictor(
      redace_config=configs.RedAceConfig(),
      model_filepath=FLAGS.model_dir,
      sequence_length=FLAGS.max_seq_length,
      batch_size=FLAGS.predict_batch_size,
  )

  num_predicted = 0
  results = []
  for (
      source_batch,
      confidence_scores_batch,
      _,
      utterance_id_batch,
  ) in utils.batch_generator(
      FLAGS.predict_input_file,
      FLAGS.predict_batch_size,
  ):
    (
        _,
        prediction_information,
    ) = predictor.predict_end_to_end_batch(source_batch,
                                           confidence_scores_batch)
    num_predicted += len(source_batch)
    logging.log_every_n(logging.INFO, f'{num_predicted} predicted.', 10)
    for source, prediction_output, utterance_id, in zip(
        source_batch,
        prediction_information,
        utterance_id_batch,
    ):
      untokenized_words = tokenization.untokenize(
          source, prediction_output.input_tokens, prediction_output.tags)
      results.append({
          'id':
              utterance_id,
          'asr': [[word, 0 if tag == 'KEEP' else 1]
                  for word, tag in untokenized_words],
      })

  with open(FLAGS.predict_output_file, 'w') as f:
    json.dump(results, f)


if __name__ == '__main__':
  flags.mark_flag_as_required('predict_input_file')
  flags.mark_flag_as_required('predict_output_file')
  flags.mark_flag_as_required('vocab_file')
  app.run(main)
