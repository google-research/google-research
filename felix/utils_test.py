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

"""Tests for felix.utils."""
import json

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from felix import tokenization
from felix import utils


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()

    vocab_tokens = [
        'NOTHING', '[CLS]', '[SEP]', '[MASK]', '[PAD]', 'a', 'b', 'c', '##d',
        'd', '##e'
    ]
    vocab_file = self.create_tempfile()
    vocab_file.write_text(''.join([x + '\n' for x in vocab_tokens]))
    self._tokenizer = tokenization.FullTokenizer(
        vocab_file.full_path, do_lower_case=True)

  @parameterized.parameters(
      # A simple test.
      {
          'source': 'a [MASK] b'.split(),
          'target': 'a c b'.split(),
          'masks': ['c']
      },
      {
          'source': 'a [MASK] [MASK] b'.split(),
          'target': 'a c d b'.split(),
          'masks': ['c', 'd']
      },
      {
          'source': '[MASK] b [MASK] d'.split(),
          'target': 'a b c d'.split(),
          'masks': ['a', 'c']
      })
  def test_build_feed_dict(self, source, target, masks):
    feed_dict = utils.build_feed_dict(source, self._tokenizer, target)

    for i, mask_id in enumerate(feed_dict['masked_lm_ids'][0]):
      # Ignore padding.
      if mask_id == 0:
        continue
      self.assertEqual(mask_id,
                       self._tokenizer.convert_tokens_to_ids(masks[i])[0])

  def test_read_wikisplit(self):
    path = self.create_tempfile()
    path = path.full_path
    with tf.io.gfile.GFile(path, 'w') as writer:
      writer.write('Source sentence .\tTarget sentence .\n')
      writer.write('2nd source .\t2nd target .')
    examples = list(utils.yield_sources_and_targets(path, 'wikisplit'))
    self.assertEqual(examples, [(['Source sentence .'], 'Target sentence .'),
                                (['2nd source .'], '2nd target .')])

  def test_read_discofuse(self):
    path = self.create_tempfile()
    path = path.full_path
    with tf.io.gfile.GFile(path, 'w') as writer:
      writer.write('coherent_first_sentence\tcoherent_second_sentence\t'
                   'incoherent_first_sentence\tincoherent_second_sentence\t'
                   'discourse_type\tconnective_string\thas_coref_type_pronoun\t'
                   'has_coref_type_nominal\n')
      writer.write(
          '1st sentence .\t2nd sentence .\t1st inc sent .\t2nd inc sent .\t'
          'PAIR_ANAPHORA\t\t1.0\t0.0\n')
      writer.write('1st sentence and 2nd sentence .\t\t1st inc sent .\t'
                   '2nd inc sent .\tSINGLE_S_COORD_ANAPHORA\tand\t1.0\t0.0')
    examples = list(utils.yield_sources_and_targets(path, 'discofuse'))
    self.assertEqual(examples, [(['1st inc sent .', '2nd inc sent .'
                                 ], '1st sentence . 2nd sentence .'),
                                (['1st inc sent .', '2nd inc sent .'
                                 ], '1st sentence and 2nd sentence .')])

  def test_read_label_map_with_tuple_keys(self):
    orig_label_map = {'KEEP': 0, 'DELETE|2': 1, 'DELETE|1': 2}
    path = self.create_tempfile(
        'label_map.json', content=json.dumps(orig_label_map)).full_path
    label_map = utils.read_label_map(path, use_str_keys=False)
    self.assertEqual(label_map, {
        ('KEEP', 0): 0,
        ('DELETE', 2): 1,
        ('DELETE', 1): 2,
    })


if __name__ == '__main__':
  absltest.main()
