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

import collections
import os
from typing import Sequence

from absl import flags
from absl.testing import absltest
import tensorflow as tf

from felix import bert_example
from felix import insertion_converter
from felix import pointing_converter

FLAGS = flags.FLAGS


def _get_int_feature(example, key):
  return example.features.feature[key].int64_list.value


class BertExampleTest(absltest.TestCase):

  def setUp(self):
    super(BertExampleTest, self).setUp()

    vocab_tokens = [
        '[CLS]', '[SEP]', '[PAD]', 'a', 'b', 'c', '##d', '##e', '[MASK]',
        '[unused1]', '[unused2]'
    ]
    vocab_file = os.path.join(FLAGS.test_tmpdir, 'vocab.txt')
    with tf.io.gfile.GFile(vocab_file, 'w') as vocab_writer:
      vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    label_map = {'KEEP': 1, 'DELETE': 2, 'KEEP|1': 3, 'KEEP|2:': 4}
    max_seq_length = 8
    do_lower_case = False
    converter = pointing_converter.PointingConverter([])
    self._builder = bert_example.BertExampleBuilder(
        label_map=label_map,
        vocab_file=vocab_file,
        max_seq_length=max_seq_length,
        do_lower_case=do_lower_case,
        converter=converter,
        use_open_vocab=False)
    max_predictions_per_seq = 4
    converter_insertion = insertion_converter.InsertionConverter(
        max_seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        vocab_file=vocab_file,
        label_map=label_map,
    )
    self._builder_mask = bert_example.BertExampleBuilder(
        label_map=label_map,
        vocab_file=vocab_file,
        max_seq_length=max_seq_length,
        do_lower_case=do_lower_case,
        converter=converter,
        use_open_vocab=True,
        converter_insertion=converter_insertion)

    self._pad_id = self._builder._get_pad_id()

  def _check_label_weights(self, labels_mask, labels, input_mask):
    self.assertAlmostEqual(sum(labels_mask), sum(input_mask))
    label_weights = collections.defaultdict(float)
    # Labels should have the same weight.
    for label, label_mask in zip(labels, labels_mask):
      # Ignore pad labels.
      if label == 0:
        continue
      label_weights[label] += label_mask
    label_weights_values = list(label_weights.values())
    for i in range(1, len(label_weights_values)):
      self.assertAlmostEqual(label_weights_values[i],
                             label_weights_values[i - 1])

  def test_building_with_target(self):
    sources = ['a b ade']  # Tokenized: [CLS] a b a ##d ##e [SEP]
    target = 'a ade'  # Tokenized: [CLS] a a ##d ##e [SEP]
    example, _ = self._builder.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b a ##d ##e [SEP] [PAD]
    self.assertEqual(example.features['input_ids'], [0, 3, 4, 3, 6, 7, 1, 2])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 1, 1, 1, 0])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(example.features['labels'], [1, 1, 2, 1, 1, 1, 1, 0])
    self.assertEqual(example.features['point_indexes'],
                     [1, 3, 0, 4, 5, 6, 0, 0])
    self.assertEqual(
        [1 if x > 0 else 0 for x in example.features_float['labels_mask']],
        [1, 1, 1, 1, 1, 1, 1, 0])
    self._check_label_weights(example.features_float['labels_mask'],
                              example.features['labels'],
                              example.features['input_mask'])

  def test_building_no_target_truncated(self):
    sources = ['ade bed cde']
    example, _ = self._builder.build_bert_example(sources)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a ##d ##e b ##e ##d [SEP]
    # where the last token 'cde' has been truncated.
    self.assertEqual(example.features['input_ids'], [0, 3, 6, 7, 4, 7, 6, 1])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 1, 1, 1, 1])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])

  def test_building_with_target_mask(self):
    sources = ['a b ade']  # Tokenized: [CLS] a b a ##d ##e [SEP]
    target = 'a ade'  # Tokenized: [CLS] a a ##d ##e [SEP]
    example, _ = self._builder_mask.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b a ##d ##e [SEP] [PAD]
    self.assertEqual(example.features['input_ids'], [0, 3, 4, 3, 6, 7, 1, 2])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 1, 1, 1, 0])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(example.features['labels'], [1, 1, 2, 1, 1, 1, 1, 0])
    self.assertEqual(example.features['point_indexes'],
                     [1, 3, 0, 4, 5, 6, 0, 0])
    self.assertEqual(
        [1 if x > 0 else 0 for x in example.features_float['labels_mask']],
        [1, 1, 1, 1, 1, 1, 1, 0])
    self._check_label_weights(example.features_float['labels_mask'],
                              example.features['labels'],
                              example.features['input_mask'])

  def test_building_with_insertion(self):
    sources = ['a b']  # Tokenized: [CLS] a b [SEP]
    target = 'a b c'  # Tokenized: [CLS] a b c [SEP]
    example, insertion_example = self._builder_mask.build_bert_example(
        sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b [SEP] [PAD] [PAD] [PAD] [PAD]
    self.assertEqual(example.features['input_ids'], [0, 3, 4, 1, 2, 2, 2, 2])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 0, 0, 0, 0])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(example.features['labels'], [1, 1, 3, 1, 0, 0, 0, 0])
    self.assertEqual(example.features['point_indexes'],
                     [1, 2, 3, 0, 0, 0, 0, 0])
    self.assertEqual(
        [1 if x > 0 else 0 for x in example.features_float['labels_mask']],
        [1, 1, 1, 1, 0, 0, 0, 0])
    self._check_label_weights(example.features_float['labels_mask'],
                              example.features['labels'],
                              example.features['input_mask'])
    self.assertEqual(insertion_example['input_ids'], [[0, 3, 4, 8, 1, 0, 0, 0]])
    self.assertEqual(insertion_example['input_mask'],
                     [[1, 1, 1, 1, 1, 0, 0, 0]])
    self.assertEqual(insertion_example['masked_lm_positions'], [[3, 0, 0, 0]])
    self.assertEqual(insertion_example['masked_lm_ids'], [[5, 0, 0, 0]])

  def test_building_with_custom_source_separator(self):
    vocab_tokens = [
        '[CLS]', '[SEP]', '[PAD]', 'a', 'b', 'c', '##d', '##e', '[MASK]',
        '[unused1]', '[unused2]'
    ]
    vocab_file = self.create_tempfile()
    vocab_file.write_text(''.join([x + '\n' for x in vocab_tokens]))

    builder = bert_example.BertExampleBuilder(
        vocab_file=vocab_file.full_path,
        label_map={
            'KEEP': 1,
            'DELETE': 2,
            'KEEP|1': 3,
            'KEEP|2:': 4
        },
        max_seq_length=9,
        do_lower_case=False,
        converter=pointing_converter.PointingConverter([]),
        use_open_vocab=False,
        special_glue_string_for_sources=' [SEP] ')

    sources = ['a b', 'ade']  # Tokenized: [CLS] a b [SEP] a ##d ##e [SEP]
    target = 'a ade'  # Tokenized: [CLS] a a ##d ##e [SEP]
    example, _ = builder.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b [SEP] a ##d ##e [SEP] [PAD]
    self.assertEqual(example.features['input_ids'], [0, 3, 4, 1, 3, 6, 7, 1, 2])
    self.assertEqual(example.features['input_mask'],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0])
    self.assertEqual(example.features['segment_ids'],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(example.features['labels'], [1, 1, 2, 2, 1, 1, 1, 1, 0])
    self.assertEqual(example.features['point_indexes'],
                     [1, 4, 0, 0, 5, 6, 7, 0, 0])
    self._check_label_weights(example.features_float['labels_mask'],
                              example.features['labels'],
                              example.features['input_mask'])
    self.assertEqual(
        [1 if x > 0 else 0 for x in example.features_float['labels_mask']],
        [1, 1, 1, 1, 1, 1, 1, 1, 0])


if __name__ == '__main__':
  absltest.main()
