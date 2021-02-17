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

"""Tests for felix.felix_models."""
from absl.testing import absltest
from absl.testing import parameterized
from official.nlp.bert import configs as bert_configs
from official.nlp.modeling import networks
import tensorflow as tf

from felix import felix_models


class FelixModelsTest(parameterized.TestCase):

  def setUp(self):
    super(FelixModelsTest, self).setUp()
    self._bert_test_config = bert_configs.BertConfig(
        attention_probs_dropout_prob=0.0,
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        hidden_size=16,
        initializer_range=0.02,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        type_vocab_size=2,
        vocab_size=30522)
    self._bert_test_config.num_classes = 20
    self._bert_test_config.query_size = 23
    self._bert_test_config.query_transformer = True

  @parameterized.named_parameters(('insertion', True), ('tagging', False))
  def test_pretrain_model(self, use_insertion=True):
    if use_insertion:
      model, encoder = felix_models.get_insertion_model(
          self._bert_test_config, seq_length=5, max_predictions_per_seq=2)
    else:
      model, encoder = felix_models.get_tagging_model(
          self._bert_test_config, seq_length=5, use_pointing=True)
    self.assertIsInstance(model, tf.keras.Model)
    self.assertIsInstance(encoder, networks.BertEncoder)

    # model has one scalar output: loss value.
    self.assertEqual(model.output.shape.as_list(), [])

    # Expect two output from encoder: sequence and classification output.
    self.assertIsInstance(encoder.output, list)
    self.assertLen(encoder.output, 2)
    # shape should be [batch size, hidden_size]
    self.assertEqual(encoder.output[1].shape.as_list(), [None, 16])


if __name__ == '__main__':
  absltest.main()
