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

"""Tests for util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
import six

from summae import util

_TESTDATA_PREFIX = os.path.join(os.path.dirname(__file__), 'testdata')


class UtilTest(absltest.TestCase):

  def setUp(self):
    super(UtilTest, self).setUp()
    self.vocab = os.path.join(
        _TESTDATA_PREFIX,
        'wikitext103_32768.subword_vocab')

  def test_get_tokenizer_with_special(self):
    tk_original = util.get_tokenizer(self.vocab)
    extra_tokens = ['<SPECIAL1>', '<SPECIAL2>']
    tk_with_special, sids = util.get_tokenizer_with_special(self.vocab,
                                                            extra_tokens)
    o_size = tk_original.vocab_size
    self.assertEqual(o_size + 2,
                     tk_with_special.vocab_size)
    self.assertEqual(['<SPECIAL1>_', '<SPECIAL2>_'],
                     tk_with_special.decode_list([o_size, o_size + 1]))
    self.assertEqual(o_size, sids['<SPECIAL1>'])
    self.assertEqual(o_size + 1, sids['<SPECIAL2>'])

  def test_get_tokenizer(self):
    tk = util.get_tokenizer(self.vocab)
    self.assertEqual('<pad><EOS>', tk.decode([util.PAD_ID, util.EOS_ID]))
    self.assertEqual(32583, tk.vocab_size)
    text = 'Tokenize this!'
    enc = tk.encode(text)
    self.assertEqual([15745, 8579, 2131, 61, 32582, 11], enc)
    # It's invertible!
    self.assertEqual(text, tk.decode(enc))

  def test_get_mturk_ground_truth(self):
    mturk_file = os.path.join(
        _TESTDATA_PREFIX,
        'truth.valid.csv')
    story2sum = util.get_mturk_ground_truth(mturk_file)
    self.assertLen(list(story2sum.keys()), 2)  # Total number of stories
    num_summaries = [len(slist) for slist in six.itervalues(story2sum)]
    self.assertEqual(5, sum(num_summaries))

  def test_checkpoint_file_gen(self):

    class FakeEstimator(object):
      model_dir = 'dir'

      def __init__(self):
        self.i = 0

      def latest_checkpoint(self):
        if self.i == 0:
          self.i += 1
          return 'some_checkpoint'
        elif self.i == 1:
          self.i += 1
          return 'some_checkpoint1'
        else:
          return None

    e = FakeEstimator()
    g = util.checkpoint_file_gen(e, '100,200',
                                 1,  # sleep secs
                                 1)  # max sleep secs
    self.assertEqual('dir/model.ckpt-100', next(g))
    self.assertEqual('dir/model.ckpt-200', next(g))

    g = util.checkpoint_file_gen(e, '', 1, 1)
    self.assertEqual('some_checkpoint', next(g))
    self.assertEqual('some_checkpoint1', next(g))
    with self.assertRaises(StopIteration):
      next(g)


if __name__ == '__main__':
  absltest.main()
