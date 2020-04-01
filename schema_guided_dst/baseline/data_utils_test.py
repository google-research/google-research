# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for schema_guided_dst.baseline.data_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest

from schema_guided_dst.baseline import data_utils

_VOCAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'test_data/bert_vocab.txt')
_TEST_DATA_DIR = os.path.dirname(_VOCAB_FILE)
_DO_LOWER_CASE = True
_DATASET = 'train'


class Dstc8DataProcessorTest(absltest.TestCase):
  """Tests for Dstc8DataProcessor."""

  def setUp(self):
    self._processor = data_utils.Dstc8DataProcessor(
        dstc8_data_dir=_TEST_DATA_DIR,
        train_file_range=range(1),
        dev_file_range=None,
        test_file_range=None,
        vocab_file=_VOCAB_FILE,
        do_lower_case=_DO_LOWER_CASE)
    super(Dstc8DataProcessorTest, self).setUp()

  def test_tokenizer(self):
    # Test normal sentence.
    test_utt_1 = 'Watch, Hellboy?'
    utt_1_tokens, utt_1_aligns, utt_1_inv_alignments = (
        self._processor._tokenize(test_utt_1))
    expected_utt_1_tokens = ['watch', ',', 'hell', '##boy', '?']
    expected_utt_1_aligns = {0: 0, 4: 0, 5: 1, 7: 2, 13: 3, 14: 4}
    expected_utt_1_inv_alignments = [(0, 4), (5, 5), (7, 13), (7, 13), (14, 14)]
    self.assertEqual(utt_1_tokens, expected_utt_1_tokens)
    self.assertEqual(utt_1_aligns, expected_utt_1_aligns)
    self.assertEqual(utt_1_inv_alignments, expected_utt_1_inv_alignments)

    # Test extra spaces in the utterance.
    test_utt_2 = 'Extra  ,  spaces'
    utt_2_tokens, utt_2_aligns, utt_2_inv_alignments = (
        self._processor._tokenize(test_utt_2))
    expected_utt_1_inv_alignments = [(0, 4), (5, 5), (7, 13), (7, 13), (14, 14)]
    self.assertEqual(utt_2_tokens, ['extra', ',', 'spaces'])
    self.assertEqual(utt_2_aligns, {0: 0, 4: 0, 7: 1, 10: 2, 15: 2})
    self.assertEqual(utt_2_inv_alignments, [(0, 4), (7, 7), (10, 15)])

    # Test # appearing in the string.
    test_utt_3 = 'Extra## ##abc'
    utt_3_tokens, utt_3_aligns, utt_3_inv_alignments = (
        self._processor._tokenize(test_utt_3))
    self.assertEqual(utt_3_tokens,
                     ['extra', '#', '#', '#', '#', 'a', '##b', '##c'])
    self.assertEqual(utt_3_aligns, {
        0: 0,
        4: 0,
        5: 1,
        6: 2,
        8: 3,
        9: 4,
        10: 5,
        12: 7
    })
    self.assertEqual(utt_3_inv_alignments, [(0, 4), (5, 5), (6, 6), (8, 8),
                                            (9, 9), (10, 12), (10, 12),
                                            (10, 12)])

  def test_get_dialog_examples(self):
    examples = self._processor.get_dialog_examples(_DATASET)
    # Check that the summary of all the turns are correct.
    expected_summaries = [
        {
            'utt_tok_mask_pairs': [('[CLS]', 0), ('[SEP]', 0),
                                   ('i', 1), ("'", 1), ('m', 1), ('looking', 1),
                                   ('for', 1), ('apartments', 1), ('.', 1),
                                   ('[SEP]', 1)],
            'utt_len': 10,
            'num_categorical_slots': 4,
            'num_categorical_slot_values': [2, 4, 4, 2, 0, 0],
            'num_noncategorical_slots': 3,
            'service_name': 'Homes_1',
            'active_intent': 'FindApartment',
            'slot_values_in_state': {}
        },
        {
            'utt_tok_mask_pairs': [('[CLS]', 0), ('which', 0), ('area', 0),
                                   ('are', 0), ('you', 0), ('looking', 0),
                                   ('in', 0), ('?', 0), ('[SEP]', 0), ('i', 1),
                                   ('want', 1), ('an', 1), ('apartment', 1),
                                   ('in', 1), ('sa', 1), ('##n', 1), ('j', 1),
                                   ('##ose', 1), ('.', 1), ('[SEP]', 1)],
            'utt_len': 20,
            'num_categorical_slots': 4,
            'num_categorical_slot_values': [2, 4, 4, 2, 0, 0],
            'num_noncategorical_slots': 3,
            'service_name': 'Homes_1',
            'active_intent': 'FindApartment',
            'slot_values_in_state': {
                'area': 'san jose'
            }
        },
        {
            'utt_tok_mask_pairs': [('[CLS]', 0), ('how', 0), ('many', 0),
                                   ('bedrooms', 0), ('do', 0), ('you', 0),
                                   ('want', 0), ('?', 0), ('[SEP]', 0),
                                   ('2', 1), ('bedrooms', 1), (',', 1),
                                   ('please', 1), ('.', 1), ('[SEP]', 1)],
            'utt_len': 15,
            'num_categorical_slots': 4,
            'num_categorical_slot_values': [2, 4, 4, 2, 0, 0],
            'num_noncategorical_slots': 3,
            'service_name': 'Homes_1',
            'active_intent': 'FindApartment',
            'slot_values_in_state': {
                'number_of_beds': '2'
            }
        },
        {
            'utt_tok_mask_pairs': [
                ('[CLS]', 0), ('there', 0), ("'", 0), ('s', 0), ('a', 0),
                ('nice', 0), ('property', 0), ('called', 0), ('a', 0),
                ('##ege', 0), ('##na', 0), ('at', 0), ('129', 0), ('##0', 0),
                ('sa', 0), ('##n', 0), ('to', 0), ('##mas', 0), ('a', 0),
                ('##quin', 0), ('##o', 0), ('road', 0), ('.', 0), ('it', 0),
                ('has', 0), ('2', 0), ('bedrooms', 0), (',', 0), ('1', 0),
                ('bath', 0), (',', 0), ('and', 0), ('rent', 0), ('##s', 0),
                ('for', 0), ('$', 0), ('2', 0), (',', 0), ('650', 0), ('a', 0),
                ('month', 0), ('.', 0), ('[SEP]', 0), ('can', 1), ('you', 1),
                ('find', 1), ('me', 1), ('a', 1), ('three', 1), ('bedroom', 1),
                ('apartment', 1), ('in', 1), ('liver', 1), ('##more', 1),
                ('?', 1), ('[SEP]', 1)
            ],
            'utt_len': 56,
            'num_categorical_slots': 4,
            'num_categorical_slot_values': [2, 4, 4, 2, 0, 0],
            'num_noncategorical_slots': 3,
            'service_name': 'Homes_1',
            'active_intent': 'FindApartment',
            'slot_values_in_state': {
                'number_of_beds': '3',
                'area': 'livermore'
            }
        },
        {
            'utt_tok_mask_pairs': [
                ('[CLS]', 0), ('there', 0), ("'", 0), ('s', 0), ('a', 0),
                ('##cacia', 0), ('capital', 0), ('co', 0), ('##r', 0), ('-', 0),
                ('iron', 0), ('##wood', 0), ('a', 0), ('##p', 0), ('at', 0),
                ('56', 0), ('##43', 0), ('ch', 0), ('##ar', 0), ('##lot', 0),
                ('##te', 0), ('way', 0), ('.', 0), ('it', 0), ('has', 0),
                ('3', 0), ('bedrooms', 0), (',', 0), ('3', 0), ('baths', 0),
                (',', 0), ('and', 0), ('rent', 0), ('##s', 0), ('for', 0),
                ('$', 0), ('4', 0), (',', 0), ('05', 0), ('##0', 0), ('a', 0),
                ('month', 0), ('.', 0), ('[SEP]', 0), ('that', 1), ('one', 1),
                ('sounds', 1), ('good', 1), ('.', 1), ('thanks', 1), (',', 1),
                ('that', 1), ("'", 1), ('s', 1), ('all', 1), ('i', 1),
                ('need', 1), ('.', 1), ('[SEP]', 1)
            ],
            'utt_len': 59,
            'num_categorical_slots': 4,
            'num_categorical_slot_values': [2, 4, 4, 2, 0, 0],
            'num_noncategorical_slots': 3,
            'service_name': 'Homes_1',
            'active_intent': 'FindApartment',
            'slot_values_in_state': {
                'property_name': 'acacia capital cor - ironwood ap'
            }
        },
    ]
    for example, gold in zip(examples, expected_summaries):
      self.assertEqual(example.readable_summary, gold)


if __name__ == '__main__':
  absltest.main()
