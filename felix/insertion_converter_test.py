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

"""Tests for felix.pointing_converter."""
from absl.testing import absltest
from absl.testing import parameterized

from felix import insertion_converter


class IntegrationInsertionConverterTest(parameterized.TestCase):

  def setUp(self):
    super(IntegrationInsertionConverterTest, self).setUp()

    vocab_tokens = [
        '[CLS]', '[SEP]', '[PAD]', '[unused1]', '[MASK]', '[unused2]', 'a', 'b',
        'c', 'd', 'e'
    ]
    self.label_map = {
        'KEEP': 1,
        'DELETE': 0,
        'MASK|1': 2,
        'MASK|2': 3,
    }
    self.vocab_file = self.create_tempfile('vocab.txt')
    self.vocab_file.write_text(''.join([x + '\n' for x in vocab_tokens]))

  @parameterized.parameters(
      # A simple test.
      {
          'input_texts': 'a b c [SEP]'.split(),
          'target': 'a b c [SEP]',
          'target_points': [1, 2, 3, 0],
          'target_masked': 'a b c [SEP]',
          'labels': [1, 1, 1, 1],
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a b c [SEP]'.split(),
          'with_delete_target': 'a b c [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 0, 0, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },

      # A multiple SEPs (think post editing) test.
      {
          'input_texts': 'a b [SEP] c [SEP]'.split(),
          'target': 'a b [SEP] c [SEP]',
          'target_points': [1, 2, 3, 4, 0],
          'target_masked': 'a b [SEP] c [SEP]',
          'labels': [1, 1, 1, 1, 1],
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a b [SEP] c [SEP]'.split(),
          'with_delete_target': 'a b [SEP] c [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 0, 0, 0, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },
      # A multiple SEPs (sep order changed) test.
      {
          'input_texts': 'a b [SEP] c [SEP]'.split(),
          'target': 'a b [SEP] c [SEP]',
          'target_points': [1, 4, 0, 2, 3],
          'target_masked': 'a b [SEP] c [SEP]',
          'labels': [1, 1, 1, 1, 1],
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a b [SEP] c [SEP]'.split(),
          'with_delete_target': 'a b [SEP] c [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 0, 0, 0, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },
      # A multiple input SEPs with first SEP deleted test.
      {
          'input_texts': 'a b [SEP] c [SEP]'.split(),
          'target': 'a b c [SEP]',
          'target_points': [1, 3, 0, 4, 0],
          'target_masked': 'a b c [SEP]',
          'labels': [1, 1, 0, 1, 1],
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a b [unused1] [SEP] [unused2] c [SEP]'.split(),
          'with_delete_target': 'a b [unused1] [SEP] [unused2] c [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 0, 2, 2, 2, 0, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },
      # A multiple input SEPs with second SEP deleted test.
      {
          'input_texts': 'a b [SEP] c [SEP]'.split(),
          'target': 'a b c [SEP]',
          'target_points': [1, 3, 0, 2, 0],
          'target_masked': 'a b c [SEP]',
          'labels': [1, 1, 1, 1, 0],
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a b c [unused1] [SEP] [unused2] [SEP]'.split(),
          'with_delete_target': 'a b c [unused1] [SEP] [unused2] [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 0, 0, 2, 2, 2, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },

      # A multiple SEPs (sep order changed) with delete test.
      {
          'input_texts': 'a b [SEP] c [SEP]'.split(),
          'target': 'a  [SEP] c [SEP]',
          'target_points': [4, 0, 0, 2, 3],
          'target_masked': 'a [SEP] c [SEP]',
          'labels': [1, 0, 1, 1, 1],
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a [unused1] b [unused2] [SEP] c [SEP]'.split(),
          'with_delete_target': 'a [unused1] b [unused2] [SEP] c [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 2, 2, 2, 0, 0, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },
      # A multiple SEPs (sep order changed) with delete and insert test.
      {
          'input_texts':
              'a b [SEP] c [SEP]'.split(),
          'target':
              'a  [SEP] c d [SEP]',
          'target_points': [4, 0, 0, 2, 3],
          'target_masked':
              'a [SEP] c [MASK] [SEP]',
          'labels': [1, 0, 1, 2, 1],
          'gold_unused_tokens':
              set([]),
          'with_delete_source':
              'a [unused1] b [unused2] [SEP] c [MASK] [SEP]'.split(),
          'with_delete_target':
              'a [unused1] b [unused2] [SEP] c d [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 2, 2, 2, 0, 0, 0, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },

      # Missing a middle token.
      {
          'input_texts': 'a b [SEP]'.split(),
          'target': 'a b c [SEP]',
          'target_points': [1, 2, 0],
          'target_masked': 'a b [MASK] [SEP]',
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a b [MASK] [SEP]'.split(),
          'with_delete_target': 'a b c [SEP]'.split(),
          'labels': [1, 2, 1],
          'feed_dict': {
              'segment_ids': [0, 0, 0, 0],
              'masked_lm_positions': [2],
              'masked_lm_weights': [1.0],
              'masked_lm_ids': ['c']
          },
      },

      # Missing a start token.
      {
          'input_texts': 'a c [SEP]'.split(),
          'target': 'a b c [SEP]',
          'target_points': [1, 2, 0],
          'target_masked': 'a [MASK] c [SEP]',
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a [MASK] c [SEP]'.split(),
          'with_delete_target': 'a b c [SEP] '.split(),
          'labels': [2, 1, 1],
          'feed_dict': {
              'segment_ids': [0, 0, 0, 0],
              'masked_lm_positions': [1],
              'masked_lm_weights': [1.0],
              'masked_lm_ids': ['b']
          },
      },

      # Missing multiple tokens.
      {
          'input_texts': 'a c [SEP]'.split(),
          'target': 'a b e c [SEP]',
          'target_points': [1, 2, 0],
          'target_masked': 'a [MASK] [MASK] c [SEP]',
          'gold_unused_tokens': set([]),
          'with_delete_source': 'a [MASK] [MASK] c [SEP]'.split(),
          'with_delete_target': 'a b e c [SEP] '.split(),
          'labels': [3, 1, 1],
          'feed_dict': {
              'segment_ids': [0, 0, 0, 0, 0],
              'masked_lm_positions': [1, 2],
              'masked_lm_weights': [1.0, 1.0],
              'masked_lm_ids': ['b', 'e']
          },
      },

      # An additional source token.
      {
          'input_texts': 'a b e [SEP]'.split(),
          'target': 'a b [SEP]',
          'target_points': [1, 3, 0, 0],
          'target_masked': 'a b [SEP]',
          'gold_unused_tokens': set([('e', '[SEP]')]),
          'labels': [1, 1, 0, 1],
          'with_delete_source': 'a b [unused1] e [unused2] [SEP]'.split(),
          'with_delete_target': 'a b [unused1] e [unused2] [SEP]'.split(),
          'feed_dict': {
              'segment_ids': [0, 0, 2, 2, 2, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },

      # Missing a middle token + an additional source token.
      {
          'input_texts':
              'a b e [SEP]'.split(),
          'target':
              'a b c [SEP]',
          'target_points': [1, 3, 0, 0],
          'target_masked':
              'a b [MASK] [SEP]',
          'gold_unused_tokens':
              set([('e', '[SEP]')]),
          'with_delete_source':
              'a b [MASK] [unused1] e [unused2] [SEP]'.split(),
          'with_delete_target':
              'a b c [unused1] e [unused2] [SEP]'.split(),
          'labels': [1, 2, 0, 1],
          'feed_dict': {
              'segment_ids': [0, 0, 0, 2, 2, 2, 0],
              'masked_lm_positions': [2],
              'masked_lm_weights': [1.0],
              'masked_lm_ids': ['c']
          },
      },

      # duplicate target and source token.
      {
          'input_texts': 'a d b e [SEP]'.split(),
          'target': 'a b d [SEP]',
          'target_points': [2, 4, 1, 0, 0],
          'target_masked': 'a b d [SEP]',
          'gold_unused_tokens': set([('e', '[SEP]')]),
          'with_delete_source': 'a b [unused1] e [unused2] d [SEP]'.split(),
          'with_delete_target': 'a b [unused1] e [unused2] d [SEP]'.split(),
          'labels': [1, 1, 1, 0, 1],
          'feed_dict': {
              'segment_ids': [0, 0, 2, 2, 2, 0, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },

      # Multiple sequential deleted tokens.
      {
          'input_texts': 'a d b e [SEP]'.split(),
          'target': 'a [SEP]',
          'target_points': [4, 0, 0, 0, 0],
          'target_masked': 'a [SEP]',
          'gold_unused_tokens': set([('d b e', '[SEP]')]),
          'with_delete_source': 'a [unused1] d b e [unused2] [SEP]'.split(),
          'with_delete_target': 'a [unused1] d b e [unused2] [SEP]'.split(),
          'labels': [1, 0, 0, 0, 1],
          'feed_dict': {
              'segment_ids': [0, 2, 2, 2, 2, 2, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      },

      # Multiple non sequential deleted tokens.
      {
          'input_texts':
              'a b c d e [SEP]'.split(),
          'target':
              'a d b [SEP]',
          'target_points': [3, 5, 0, 1, 0, 0],
          'target_masked':
              'a d b [SEP]',
          'gold_unused_tokens':
              set([('c', 'd'), ('e', '[SEP]')]),
          'with_delete_source':
              'a d [unused1] e [unused2] b [unused1] c [unused2] [SEP]'.split(),
          'with_delete_target':
              'a d [unused1] e [unused2] b [unused1] c [unused2] [SEP]'.split(),
          'labels': [1, 1, 0, 1, 0, 1],
          'feed_dict': {
              'segment_ids': [0, 0, 2, 2, 2, 0, 2, 2, 2, 0],
              'masked_lm_positions': [],
              'masked_lm_weights': [],
              'masked_lm_ids': []
          },
      })
  def test_create_insertion_example(self, input_texts, target, target_points,
                                    target_masked, labels, with_delete_source,
                                    with_delete_target, gold_unused_tokens,
                                    feed_dict):
    del target_masked, with_delete_target, gold_unused_tokens

    converter = insertion_converter.InsertionConverter(
        max_seq_length=20,
        max_predictions_per_seq=20,
        vocab_file=self.vocab_file.full_path,
        label_map=self.label_map,
        fall_back_mode='force')
    output_feed_dict = converter.create_insertion_example(
        input_texts, labels, target_points, target.split())
    # Remove padding.
    no_pad_input_ids = []

    seen_padding = False
    for input_id in output_feed_dict['input_ids'][0]:

      if input_id != 0:
        no_pad_input_ids.append(input_id)
        self.assertEqual(seen_padding, False)
      else:
        seen_padding = True

    self.assertEqual(
        converter._tokenizer.convert_ids_to_tokens(no_pad_input_ids),
        with_delete_source)
    # :len(x) ensures padding is not considered.
    self.assertEqual(
        feed_dict['segment_ids'],
        output_feed_dict['segment_ids'][0][:len(feed_dict['segment_ids'])])
    self.assertEqual(
        feed_dict['masked_lm_positions'],
        output_feed_dict['masked_lm_positions'][0][:len((
            feed_dict['masked_lm_positions']))])
    self.assertEqual(
        feed_dict['masked_lm_weights'],
        output_feed_dict['masked_lm_weights'][0][:len((
            feed_dict['masked_lm_weights']))])
    self.assertEqual(
        feed_dict['masked_lm_ids'],
        converter._tokenizer.convert_ids_to_tokens(
            output_feed_dict['masked_lm_ids'][0][:len((
                feed_dict['masked_lm_ids']))]))


if __name__ == '__main__':
  absltest.main()
