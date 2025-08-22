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

from etcmodel.models import modeling
from mave.benchmark.models import etc_tagger
from official.nlp import optimization


class EtcTaggerTest(unittest.TestCase):

  def test_etc_tagger_call(self):
    inputs = {
        'global_token_ids': tf.constant([[1, 2], [2, 3]]),
        'global_breakpoints': tf.constant([[1, 0], [1, 0]]),
        'global_token_type_ids': tf.constant([[1, 2], [1, 0]]),
        'long_token_ids': tf.constant([[1, 2, 3], [2, 3, 4]]),
        'long_breakpoints': tf.constant([[0, 0, 1], [0, 1, 0]]),
        'long_token_type_ids': tf.constant([[1, 2, 0], [1, 0, 0]]),
        'long_paragraph_ids': tf.constant([[0, 1, 0], [0, 0, 0]]),
    }
    etc_config = modeling.EtcConfig(vocab_size=5)
    tagger = etc_tagger.EtcSequenceTagger(etc_config)

    outputs = tf.function(tagger)(inputs)

    self.assertEqual(set(outputs), {'global', 'long'})
    self.assertEqual(outputs['global'].shape, [2, 2, 1])
    self.assertEqual(outputs['long'].shape, [2, 3, 1])

  def test_etc_tagger_train_step(self):
    inputs = {
        'global_token_ids': tf.constant([[1, 2], [2, 3]]),
        'global_breakpoints': tf.constant([[1, 0], [1, 0]]),
        'global_token_type_ids': tf.constant([[1, 2], [1, 0]]),
        'long_token_ids': tf.constant([[1, 2, 3], [2, 3, 4]]),
        'long_breakpoints': tf.constant([[0, 0, 1], [0, 1, 0]]),
        'long_token_type_ids': tf.constant([[1, 2, 0], [1, 0, 0]]),
        'long_paragraph_ids': tf.constant([[0, 1, 0], [0, 0, 0]]),
    }
    labels = {
        'global': tf.constant([[1, 1], [1, 0]]),
        'long': tf.constant([[1, 1, 0], [1, 0, 0]]),
    }
    etc_config = modeling.EtcConfig(vocab_size=5)
    tagger = etc_tagger.EtcSequenceTagger(etc_config)
    optimizer = optimization.create_optimizer(1e-3, 2, 1, 0.0, 'adamw')
    tagger.compile(optimizer=optimizer, loss='binary_crossentropy')

    metrics = tf.function(tagger.train_step)((inputs, labels))

    self.assertEqual(set(metrics), {'loss', 'global_loss', 'long_loss'})
    self.assertEqual(metrics['loss'].shape, [])
    self.assertEqual(metrics['global_loss'].shape, [])
    self.assertEqual(metrics['long_loss'].shape, [])

if __name__ == '__main__':
  unittest.main()
