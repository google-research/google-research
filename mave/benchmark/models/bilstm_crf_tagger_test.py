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

import unittest

import tensorflow as tf

from mave.benchmark.models import bilstm_crf_tagger
from official.nlp import optimization


class BilstmCrfTaggerTest(unittest.TestCase):

  def test_bilstm_crf_sequence_tagger_call(self):
    inputs = {
        'input_word_ids': tf.constant([[1, 2, 3], [2, 3, 4]]),
        'input_mask': tf.constant([[1, 1, 0], [1, 0, 0]]),
    }
    tagger = bilstm_crf_tagger.BiLSTMCRFSequenceTagger(
        seq_length=3,
        vocab_size=5,
        word_embeddding_size=10,
        lstm_units=10,
        recurrent_dropout=0.4,
        use_attention_layer=True,
        use_attention_scale=True,
        attention_dropout=0.0,
        num_tags=2,
    )

    outputs = tf.function(tagger)(inputs)

    self.assertEqual(outputs.shape, [2, 3])

  def test_bilstm_crf_sequence_tagger_train_step(self):
    inputs = {
        'input_word_ids': tf.constant([[1, 2, 3], [2, 3, 4]]),
        'input_mask': tf.constant([[1, 1, 0], [1, 0, 0]]),
    }
    labels = tf.constant([[1, 1, 0], [1, 0, 0]])
    tagger = bilstm_crf_tagger.BiLSTMCRFSequenceTagger(
        seq_length=3,
        vocab_size=5,
        word_embeddding_size=10,
        lstm_units=10,
        recurrent_dropout=0.4,
        use_attention_layer=True,
        use_attention_scale=True,
        attention_dropout=0.0,
        num_tags=2,
    )
    optimizer = optimization.create_optimizer(1e-3, 2, 1, 0.0, 'adamw')
    tagger.compile(optimizer=optimizer)

    metrics = tf.function(tagger.train_step)((inputs, labels))

    self.assertEqual(set(metrics), {'loss'})
    self.assertEqual(metrics['loss'].shape, [])


if __name__ == '__main__':
  unittest.main()
