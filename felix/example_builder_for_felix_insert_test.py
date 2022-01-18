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

import collections

from absl.testing import absltest
from absl.testing import parameterized

from felix import example_builder_for_felix_insert as builder


class ExampleBuilderTest(parameterized.TestCase):

  def setUp(self):
    super(ExampleBuilderTest, self).setUp()

    vocab_tokens = [
        '[CLS]', '[SEP]', '[PAD]', 'a', 'b', 'c', '##d', '##e', '[unused1]',
        '[unused2]', '[MASK]'
    ]
    vocab_file = self.create_tempfile()
    vocab_file.write_text(''.join([x + '\n' for x in vocab_tokens]))

    label_map = {
        ('KEEP', 0): 1,
        ('DELETE', 0): 2,
        ('KEEP', 1): 3,
        ('DELETE', 1): 4,
        ('KEEP', 2): 5,
        ('DELETE', 2): 6,
    }
    max_seq_length = 8
    self._builder = builder.FelixInsertExampleBuilder(
        label_map,
        vocab_file.full_path,
        True,
        max_seq_length,
        max_predictions_per_seq=4,
        max_insertions_per_token=2,
        insert_after_token=True)
    self._builder_insert_before = builder.FelixInsertExampleBuilder(
        label_map,
        vocab_file.full_path, True,
        max_seq_length,
        max_predictions_per_seq=4,
        max_insertions_per_token=2,
        insert_after_token=False)

  def _check_label_weights(self, labels_mask, labels, input_mask):
    # As the first or last label is removed after the weights are determined
    # these numbers will not match exactly.
    self.assertLess(abs(sum(labels_mask) - sum(input_mask)), 1)
    label_weights = collections.defaultdict(float)
    # Labels should have the same weight.
    for label, label_mask in zip(labels, labels_mask):
      # Ignore pad labels.
      if label == 0:
        continue
      label_weights[label] += label_mask
    label_weights_values = list(label_weights.values())
    for i in range(1, len(label_weights_values)):
      self.assertLess(
          abs(label_weights_values[i] - label_weights_values[i - 1]), 1)

  def test_building_without_insertions(self):
    sources = ['a b cd']  # Tokenized: [CLS] a b c ##d [SEP]
    target = 'a cd'  # Tokenized: [CLS] a c ##d [SEP]
    tagging, insertion = self._builder.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b c ##d [SEP] [PAD] [PAD]
    self.assertEqual(tagging.features['input_ids'], [0, 3, 4, 5, 6, 1, 2, 2])
    self.assertEqual(tagging.features['input_mask'], [1, 1, 1, 1, 1, 1, 0, 0])
    self.assertEqual(tagging.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(tagging.features['labels'], [1, 1, 2, 1, 1, 1, 0, 0])
    self.assertEqual(
        [1 if x > 0 else 0 for x in tagging.features_float['labels_mask']],
        [1, 1, 1, 1, 1, 0, 0, 0])
    self._check_label_weights(tagging.features_float['labels_mask'],
                              tagging.features['labels'],
                              tagging.features['input_mask'])
    # Target doesn't contain any insertions so `insertion` should be None.
    self.assertIsNone(insertion)

  def test_building_with_replacement(self):
    sources = ['a b']  # Tokenized: [CLS] a b [SEP]
    target = 'a cd'  # Tokenized: [CLS] a c ##d [SEP]
    tagging, insertion = self._builder.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b [SEP] [PAD] [PAD] [PAD] [PAD]
    self.assertEqual(tagging.features['input_ids'], [0, 3, 4, 1, 2, 2, 2, 2])
    self.assertEqual(tagging.features['input_mask'], [1, 1, 1, 1, 0, 0, 0, 0])
    self.assertEqual(tagging.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(tagging.features['labels'], [1, 1, 6, 1, 0, 0, 0, 0])
    self.assertEqual(
        [1 if x > 0 else 0 for x in tagging.features_float['labels_mask']],
        [1, 1, 1, 0, 0, 0, 0, 0])
    self._check_label_weights(tagging.features_float['labels_mask'],
                              tagging.features['labels'],
                              tagging.features['input_mask'])
    self.assertIsNotNone(insertion)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a [unused1] b [unused2] [MASK] [MASK] [SEP]
    self.assertEqual(insertion['input_ids'], [[0, 3, 8, 4, 9, 10, 10, 1]])
    self.assertEqual(insertion['input_mask'], [[1, 1, 1, 1, 1, 1, 1, 1]])
    self.assertEqual(insertion['segment_ids'], [[0, 0, 2, 2, 2, 0, 0, 0]])

    self.assertEqual(insertion['masked_lm_positions'], [[5, 6, 0, 0]])
    # masked_lm_ids should contain the IDs for the following tokens: c ##d
    self.assertEqual(insertion['masked_lm_ids'], [[5, 6, 0, 0]])
    self.assertEqual(insertion['masked_lm_weights'], [[1, 1, 0, 0]])

  @parameterized.parameters(True, False)
  def test_building_with_insertion(self, insert_after_token):
    sources = ['a b']  # Tokenized: [CLS] a b [SEP]
    target = 'a cd b'  # Tokenized: [CLS] a c ##d [SEP]

    if insert_after_token:
      example_builder = self._builder
    else:
      example_builder = self._builder_insert_before

    tagging, insertion = example_builder.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b [SEP] [PAD] [PAD] [PAD] [PAD]
    self.assertEqual(tagging.features['input_ids'], [0, 3, 4, 1, 2, 2, 2, 2])
    if insert_after_token:
      # label should contain the IDs for: KEEP KEEP|2 KEEP KEEP
      self.assertEqual(tagging.features['labels'], [1, 5, 1, 1, 0, 0, 0, 0])
      self.assertEqual(
          [1 if x > 0 else 0 for x in tagging.features_float['labels_mask']],
          [1, 1, 1, 0, 0, 0, 0, 0])
      self._check_label_weights(tagging.features_float['labels_mask'],
                                tagging.features['labels'],
                                tagging.features['input_mask'])
    else:
      # label should contain the IDs for: KEEP KEEP KEEP|2 KEEP
      self.assertEqual(tagging.features['labels'], [1, 1, 5, 1, 0, 0, 0, 0])
      self.assertEqual(
          [1 if x > 0 else 0 for x in tagging.features_float['labels_mask']],
          [0, 1, 1, 1, 0, 0, 0, 0])
      self._check_label_weights(tagging.features_float['labels_mask'],
                                tagging.features['labels'],
                                tagging.features['input_mask'])

    self.assertIsNotNone(insertion)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a [MASK] [MASK] b [SEP]
    self.assertEqual(insertion['input_ids'], [[0, 3, 10, 10, 4, 1, 0, 0]])
    self.assertEqual(insertion['input_mask'], [[1, 1, 1, 1, 1, 1, 0, 0]])
    self.assertEqual(insertion['segment_ids'], [[0, 0, 0, 0, 0, 0, 0, 0]])

    self.assertEqual(insertion['masked_lm_positions'], [[2, 3, 0, 0]])
    # masked_lm_ids should contain the IDs for the following tokens: c ##d
    self.assertEqual(insertion['masked_lm_ids'], [[5, 6, 0, 0]])
    self.assertEqual(insertion['masked_lm_weights'], [[1, 1, 0, 0]])

  def test_building_no_target_truncated(self):
    sources = ['ade bed cde']
    tagging, _ = self._builder.build_bert_example(sources)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a ##d ##e b ##e ##d [SEP]
    # where the last token 'cde' has been truncated.
    self.assertEqual(tagging.features['input_ids'], [0, 3, 6, 7, 4, 7, 6, 1])
    self.assertEqual(tagging.features['input_mask'], [1, 1, 1, 1, 1, 1, 1, 1])
    self.assertEqual(tagging.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])

  def test_building_insertion_truncated(self):
    sources = ['a b c bed']
    target = 'a bd c'
    tagging, insertion = self._builder.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b c b ##e ##d [SEP]
    self.assertEqual(tagging.features['input_ids'], [0, 3, 4, 5, 4, 7, 6, 1])
    self.assertEqual(tagging.features['input_mask'], [1, 1, 1, 1, 1, 1, 1, 1])
    self.assertEqual(tagging.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(tagging.features['labels'], [1, 1, 3, 1, 2, 2, 2, 1])
    self.assertEqual(
        [1 if x > 0 else 0 for x in tagging.features_float['labels_mask']],
        [1, 1, 1, 1, 1, 1, 1, 0])
    self._check_label_weights(tagging.features_float['labels_mask'],
                              tagging.features['labels'],
                              tagging.features['input_mask'])

    self.assertIsNotNone(insertion)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b [MASK] c [unused1] b [SEP]
    self.assertEqual(insertion['input_ids'], [[0, 3, 4, 10, 5, 8, 4, 1]])
    self.assertEqual(insertion['input_mask'], [[1, 1, 1, 1, 1, 1, 1, 1]])
    self.assertEqual(insertion['segment_ids'], [[0, 0, 0, 0, 0, 2, 2, 2]])

    self.assertEqual(insertion['masked_lm_positions'], [[3, 0, 0, 0]])
    # masked_lm_ids should contain the ID for the following token: c
    self.assertEqual(insertion['masked_lm_ids'], [[6, 0, 0, 0]])
    self.assertEqual(insertion['masked_lm_weights'], [[1, 0, 0, 0]])

  def test_building_too_long_insertion(self):
    sources = ['a b']
    target = 'a cdd b'
    # `max_insertions_per_token` is 2 but the target contains an insertion of
    # length 3: c ##d ##d"
    tagging, insertion = self._builder.build_bert_example(sources, target)
    self.assertIsNone(tagging)
    self.assertIsNone(insertion)


if __name__ == '__main__':
  absltest.main()
