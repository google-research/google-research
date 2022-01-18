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

import json
import random

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from official.nlp.bert import configs

from felix import predict

FLAGS = flags.FLAGS


def _convert_to_one_hot(labels, vocab_size):
  one_hots = []
  for label in labels:
    one_hot = [0] * vocab_size
    one_hot[label] = 1
    one_hots.append(one_hot)
  return np.array(one_hots)


class DummyPredictorTagging():

  def __init__(self, pred, raw_points=None):
    """Initializer for a dummy predictor.

    Args:
      pred: The predicted tag.
      raw_points: Logits for the pointer network (X,X) matrix. If None, then
        this value won't be returned when calling the predictor.
    """
    self._pred = pred
    self._raw_points = raw_points

  def __call__(self, example=None, training=None):
    del example, training
    if self._raw_points is not None:
      return np.array([self._pred]), np.array([self._raw_points])
    else:
      return np.array([self._pred])


class DummyPredictorInsertion:

  def __init__(self, prediction):
    """Initializer for a dummy predictor.

    Args:
      prediction: Predicted tokens
    """
    self._prediction = prediction

  def __call__(self, example=None, training=None):
    del example, training
    # Prepend and append IDs for the begin and the end token.
    return self._prediction


class PredictUtilsTest(parameterized.TestCase):

  def setUp(self):
    super(PredictUtilsTest, self).setUp()

    self._vocab_tokens = [
        'NOTHING', '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]', '[unused1]',
        '[unused2]', 'a', 'b', 'c', 'd'
    ]
    vocab_file = self.create_tempfile('vocab.txt')
    vocab_file.write_text(''.join([x + '\n' for x in self._vocab_tokens]))
    self._vocab_file = vocab_file.full_path
    self._vocab_to_id = {}
    for i, vocab_token in enumerate(self._vocab_tokens):
      self._vocab_to_id[vocab_token] = i
    self._label_map = {'PAD': 0, 'KEEP': 1, 'DELETE': 2, 'KEEP|1': 3}
    self._label_map_file = self.create_tempfile('label_map.json')
    self._label_map_file.write_text(json.dumps(self._label_map))
    self._label_map_path = self._label_map_file.full_path
    self._max_sequence_length = 30
    self._max_predictions = 20
    self._bert_test_tagging_config = configs.BertConfig(
        attention_probs_dropout_prob=0.0,
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        hidden_size=16,
        initializer_range=0.02,
        intermediate_size=32,
        max_position_embeddings=40,
        num_attention_heads=1,
        num_hidden_layers=1,
        type_vocab_size=2,
        vocab_size=len(self._vocab_tokens))
    self._bert_test_tagging_config.num_classes = len(self._label_map)
    self._bert_test_tagging_config.query_size = 23
    self._bert_test_tagging_config.pointing = False
    self._bert_test_tagging_config.query_transformer = False

  def test_predict_end_to_end_batch_random(self):
    """Test the model predictions end-2-end with randomly initialized models."""

    batch_size = 11
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        vocab_file=self._vocab_file,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=True,
        do_lowercase=True,
        use_open_vocab=True)
    source_batch = []

    for i in range(batch_size):
      source_batch.append(' '.join(
          random.choices(self._vocab_tokens[8:], k=i + 1)))
    # Uses a randomly initialized tagging model.
    predictions_tagging, predictions_insertion = \
        felix_predictor.predict_end_to_end_batch(source_batch)
    self.assertLen(predictions_tagging, batch_size)
    self.assertLen(predictions_insertion, batch_size)

  @parameterized.parameters(
      # Straightforward.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'insertions': ['NOTHING'],
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ]),
      },
      # With deletion.
      {
          'pred': [1, 2, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] b c [SEP]',
          'insertions': ['NOTHING'],
          'gold_with_deletions':
              '[CLS] [unused1] a [unused2] b c [SEP]',
          'raw_points':
              np.array([
                  [0, 0, 10, 0, 0, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ]),
      },
            # With deletion and insertion.
      {
          'pred': [3, 2, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] d b c [SEP]',
          'insertions': ['d'],
          'gold_with_deletions':
              '[CLS] [MASK] [unused1] a [unused2] b c [SEP]',
          'raw_points':
              np.array([
                  [0, 0, 10, 0, 0, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ]),
      },
      )
  def test_predict_end_to_end_batch_fake(self, pred, raw_points, sources, gold,
                                         gold_with_deletions, insertions):
    """Test end-to-end with fake tensorflow models."""
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        vocab_file=self._vocab_file,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=True,
        do_lowercase=True,
        use_open_vocab=True)
    tagging_model = DummyPredictorTagging(
        _convert_to_one_hot(pred, len(self._label_map)), raw_points)
    insertions = [
        _convert_to_one_hot([self._vocab_to_id[token] for token in insertions],
                            len(self._vocab_tokens))
    ]
    insertion_model = DummyPredictorInsertion(insertions)
    felix_predictor._tagging_model = tagging_model
    felix_predictor._insertion_model = insertion_model
    taggings_outputs, insertion_outputs = (
        felix_predictor.predict_end_to_end_batch(sources))
    self.assertEqual(taggings_outputs[0], gold_with_deletions)
    self.assertEqual(insertion_outputs[0], gold)

  def test_convert_source_sentences_into_tagging_batch(self):
    batch_size = 11
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=True,
        do_lowercase=True,
        vocab_file=self._vocab_file,
        use_open_vocab=True)
    source_batch = []
    for i in range(batch_size):
      # Produce random sentences from the vocab (excluding special tokens).
      source_batch.append(' '.join(
          random.choices(self._vocab_tokens[7:], k=i + 1)))
    batch_dictionaries, batch_list = (
        felix_predictor._convert_source_sentences_into_batch(
            source_batch, is_insertion=False))
    # Each input should be of the size (batch_size, max_sequence_length).
    for value in batch_list.values():
      self.assertEqual(value.shape, (batch_size, self._max_sequence_length))

    self.assertLen(batch_dictionaries, batch_size)
    for batch_item in batch_dictionaries:
      for value in batch_item.values():
        self.assertLen(value, self._max_sequence_length)

  def test_convert_source_sentences_into_insertion_batch(self):
    batch_size = 11
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        vocab_file=self._vocab_file,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        max_predictions=self._max_predictions,
        is_pointing=True,
        do_lowercase=True,
        use_open_vocab=True)
    source_batch = []
    for i in range(batch_size):
      # Produce random sentences from the vocab (excluding special tokens).
      source_batch.append(
          ' '.join(random.choices(self._vocab_tokens[7:], k=i + 1)))

    batch_dictionaries, batch_list = (
        felix_predictor._convert_source_sentences_into_batch(
            source_batch, is_insertion=True))
    # Each input should be of the size (batch_size, max_sequence_length).
    self.assertEqual(batch_list['input_word_ids'].shape,
                     (batch_size, self._max_sequence_length))
    self.assertEqual(batch_list['input_mask'].shape,
                     (batch_size, self._max_sequence_length))
    self.assertEqual(batch_list['input_type_ids'].shape,
                     (batch_size, self._max_sequence_length))

    self.assertEqual(batch_list['masked_lm_positions'].shape,
                     (batch_size, self._max_predictions))
    self.assertEqual(batch_list['masked_lm_ids'].shape,
                     (batch_size, self._max_predictions))
    self.assertEqual(batch_list['masked_lm_weights'].shape,
                     (batch_size, self._max_predictions))

    self.assertLen(batch_dictionaries, batch_size)
    for batch_item in batch_dictionaries:
      self.assertLen(batch_item['input_word_ids'], self._max_sequence_length)
      self.assertLen(batch_item['input_mask'], self._max_sequence_length)
      self.assertLen(batch_item['input_type_ids'], self._max_sequence_length)
      self.assertLen(batch_item['masked_lm_positions'], self._max_predictions)
      self.assertLen(batch_item['masked_lm_ids'], self._max_predictions)
      self.assertLen(batch_item['masked_lm_weights'], self._max_predictions)

  def test_predict_tagging_batch(self):

    batch_size = 11
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=True,
        vocab_file=self._vocab_file,
        do_lowercase=True,
        use_open_vocab=True)
    source_batch = []
    for i in range(batch_size):
      source_batch.append(' '.join(
          random.choices(self._vocab_tokens[7:], k=i + 1)))
    # Uses a randomly initialized tagging model.
    predictions = felix_predictor._predict_batch(
        felix_predictor._convert_source_sentences_into_batch(
            source_batch, is_insertion=False)[1],
        is_insertion=False)
    self.assertLen(predictions, batch_size)

    for tag_logits, pointing_logits in predictions:
      self.assertLen(tag_logits, self._max_sequence_length)
      self.assertEqual(pointing_logits.shape,
                       (self._max_sequence_length, self._max_sequence_length))

  def test_predict_insertion_batch(self):

    batch_size = 11
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        vocab_file=self._vocab_file,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=True,
        do_lowercase=True,
        use_open_vocab=True)
    source_batch = []
    for i in range(batch_size):
      source_batch.append(' '.join(
          random.choices(self._vocab_tokens[7:], k=i + 1)))
    # Uses a randomly initialized tagging model.
    predictions = felix_predictor._predict_batch(
        felix_predictor._convert_source_sentences_into_batch(
            source_batch, is_insertion=True)[1],
        is_insertion=True)
    self.assertLen(predictions, batch_size)

    for prediction in predictions:
      self.assertLen(prediction, self._max_predictions)

  @parameterized.parameters(
      # Straightforward.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },
      # Go backwards.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] c b a [SEP]',
          'gold_with_deletions':
              '[CLS] c b a [SEP]',
          'raw_points':
              np.array([
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },
      # Make everything noisier.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] c b a [SEP]',
          'gold_with_deletions':
              '[CLS] c b a [SEP]',
          'raw_points':
              np.array([
                  [2, 3, 4, 10, 5, 6],
                  [2, 3, 4, 5, 10, 6],
                  [2, 10, 3, 4, 5, 6],
                  [2, 3, 10, 4, 5, 7],
                  [10, 2, 4, 5, 6, 7],
                  [1, 1, 1, 1, 1, 1],
              ])
      },

      # A tempting start.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 30, 0, 0, 0],
                  [0, 0, 100, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },

      # A temptation in the middle.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 10, 30, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },

      # Don't revisit the past.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 6, 5, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },
      #  No starting place.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },

      # Lost in the middle.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },

      # Skip to the end.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] a b c [SEP]',
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 1, 0, 0, 100, 0],
                  [0, 0, 10, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },
      # Skip past the end.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold_with_deletions':
              '[CLS] a b c [SEP]',
          'gold':
              '[CLS] a b c [SEP]',
          'raw_points':
              np.array([
                  [0, 1, 0, 0, 100, 0],
                  [0, 0, 10, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 0, 100],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 10, 0],
              ])
      },
      {
          # Don't visit that!
          'pred': [1, 0, 1, 1, 1],
          'sources': ['a b c'],
          'gold':
              '[CLS] b c [SEP]',
          'gold_with_deletions':
              '[CLS] [unused1] a [unused2] b c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 1, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },

      # Straightforward with multiple SEPs.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a [SEP] c'],
          'gold':
              '[CLS] a [SEP] c [SEP]',
          'gold_with_deletions':
              '[CLS] a [SEP] c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },
      # Last SEP becomes middle SEP.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a [SEP] c'],
          'gold':
              '[CLS] a [SEP] c [SEP]',
          'gold_with_deletions':
              '[CLS] a [SEP] c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [0, 0, 0, 10, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },
      # Delete middle SEP.
      {
          'pred': [1, 1, 2, 1, 1],
          'sources': ['a [SEP] c'],
          'gold':
              '[CLS] a c [SEP]',
          'gold_with_deletions':
              '[CLS] a [unused1] [SEP] [unused2] c [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 0, 10, 0, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 10, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },

      # Delete last SEP.
      {
          'pred': [1, 1, 1, 1, 2],
          'sources': ['a [SEP] c'],
          'gold':
              '[CLS] a c [SEP]',
          'gold_with_deletions':
              '[CLS] a c [unused1] [SEP] [unused2] [SEP]',
          'raw_points':
              np.array([
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 10, 0, 0, 0],
                  [10, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
              ])
      },
  )
  def test_predict_and_realize_tagging_batch(self, pred, sources, gold,
                                             gold_with_deletions, raw_points):
    del gold
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=True,
        do_lowercase=True,
        vocab_file=self._vocab_file,
        use_open_vocab=True)
    tagging_model = DummyPredictorTagging(
        _convert_to_one_hot(pred, len(self._label_map)), raw_points)
    felix_predictor._tagging_model = tagging_model
    realized_predictions = felix_predictor._predict_and_realize_batch(
        sources, is_insertion=False)
    self.assertEqual(realized_predictions[0], gold_with_deletions)

  @parameterized.parameters(
      # Straightforward.
      {
          'pred': [1, 1, 1, 1, 1],
          'sources': ['a b c'],
          'gold_with_deletions': '[CLS] a b c [SEP]',
          'insert_after_token': True,
      },
      # Deletion.
      {
          'pred': [1, 2, 1, 1, 1],
          'sources': ['a b c'],
          'gold_with_deletions': '[CLS] [unused1] a [unused2] b c [SEP]',
          'insert_after_token': True,
      },
      # Special handling of [UNK].
      {
          'pred': [1, 1, 2, 1, 1],
          'sources': ['a q c'],
          'gold_with_deletions': '[CLS] a [unused1] [UNK] [unused2] c [SEP]',
          'insert_after_token': True,
      },
      # Insertion.
      {
          'pred': [1, 3, 1, 1, 1],
          'sources': ['a b c'],
          'gold_with_deletions': '[CLS] a [MASK] b c [SEP]',
          'insert_after_token': True,
      },
      # Insertion before token.
      {
          'pred': [1, 3, 1, 1, 1],
          'sources': ['a b c'],
          'gold_with_deletions': '[CLS] [MASK] a b c [SEP]',
          'insert_after_token': False,
      },
  )
  def test_predict_and_realize_tagging_batch_for_felix_insert(
      self, pred, sources, gold_with_deletions, insert_after_token):
    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=False,
        do_lowercase=True,
        vocab_file=self._vocab_file,
        use_open_vocab=True,
        insert_after_token=insert_after_token)
    tagging_model = DummyPredictorTagging(
        _convert_to_one_hot(pred, len(self._label_map)))
    felix_predictor._tagging_model = tagging_model
    realized_predictions = felix_predictor._predict_and_realize_batch(
        sources, is_insertion=False)
    self.assertEqual(realized_predictions[0], gold_with_deletions)

  @parameterized.parameters(
      # No insertions.
      {
          'prediction': ['NOTHING'],
          'sources': ['[CLS] a b c [SEP]'],
          'gold': '[CLS] a b c [SEP]'
      },
      {
          'prediction': ['b'],
          'sources': ['[CLS] a [MASK] c [SEP]'],
          'gold': '[CLS] a b c [SEP]'
      },
      {
          'prediction': ['b', 'c'],
          'sources': ['[CLS] a [MASK] [MASK] [SEP]'],
          'gold': '[CLS] a b c [SEP]'
      },
      {
          'prediction': ['a', 'c'],
          'sources': ['[CLS] [MASK] b [MASK] [SEP]'],
          'gold': '[CLS] a b c [SEP]'
      },
      {
          'prediction': ['c'],
          'sources': ['[CLS] [unused1] a [unused2] b [MASK] [SEP]'],
          'gold': '[CLS] b c [SEP]'
      })
  def test_predict_and_realize_insertion_batch(self, sources, prediction, gold):
    """Test predicting and realizing insertion with fake tensorflow models."""
    prediction = [
        _convert_to_one_hot([self._vocab_to_id[token] for token in prediction],
                            len(self._vocab_tokens))
    ]

    felix_predictor = predict.FelixPredictor(
        bert_config_insertion=self._bert_test_tagging_config,
        bert_config_tagging=self._bert_test_tagging_config,
        vocab_file=self._vocab_file,
        model_tagging_filepath=None,
        model_insertion_filepath=None,
        label_map_file=self._label_map_path,
        sequence_length=self._max_sequence_length,
        is_pointing=True,
        do_lowercase=True,
        use_open_vocab=True)

    insertion_model = DummyPredictorInsertion(prediction)
    felix_predictor._insertion_model = insertion_model
    realized_predictions = felix_predictor._predict_and_realize_batch(
        sources, is_insertion=True)
    self.assertEqual(realized_predictions[0], gold)


if __name__ == '__main__':
  absltest.main()
