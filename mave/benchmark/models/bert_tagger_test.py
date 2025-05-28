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

from mave.benchmark.models import bert_tagger
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs


class BertTaggerTest(unittest.TestCase):

  def test_bert_tagger_call(self):
    inputs = {
        'input_word_ids': tf.constant([[1, 2, 3], [2, 3, 4]]),
        'input_mask': tf.constant([[1, 1, 0], [1, 0, 0]]),
        'input_type_ids': tf.constant([[1, 2, 0], [1, 0, 0]]),
    }
    bert_config = bert_configs.BertConfig(vocab_size=5)
    bert_encoder = bert_models.get_transformer_encoder(bert_config)
    tagger = bert_tagger.BertSequenceTagger(network=bert_encoder)

    outputs = tf.function(tagger)(inputs)

    self.assertEqual(outputs.shape, [2, 3, 1])

  def test_bert_tagger_train_step(self):
    inputs = {
        'input_word_ids': tf.constant([[1, 2, 3], [2, 3, 4]]),
        'input_mask': tf.constant([[1, 1, 0], [1, 0, 0]]),
        'input_type_ids': tf.constant([[1, 2, 0], [1, 0, 0]]),
    }
    labels = tf.constant([[1, 1, 0], [1, 0, 0]])
    bert_config = bert_configs.BertConfig(vocab_size=5)
    bert_encoder = bert_models.get_transformer_encoder(bert_config)
    tagger = bert_tagger.BertSequenceTagger(network=bert_encoder)
    optimizer = optimization.create_optimizer(1e-3, 2, 1, 0.0, 'adamw')
    tagger.compile(optimizer=optimizer, loss='binary_crossentropy')

    metrics = tf.function(tagger.train_step)((inputs, labels))

    self.assertEqual(set(metrics), {'loss'})
    self.assertEqual(metrics['loss'].shape, [])


if __name__ == '__main__':
  unittest.main()
