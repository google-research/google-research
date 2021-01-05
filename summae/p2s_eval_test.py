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

"""Tests for p2s_eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
import numpy as np
from summae import p2s_eval
from summae import util


class P2sEvalTest(absltest.TestCase):

  def test_p2seval(self):
    seq_exs = util.get_seq_exs(os.path.join(
        os.path.dirname(__file__), 'testdata', 'gt_seqex.tfrecord'))
    e = p2s_eval.P2sEval(seq_exs)
    def to_tuple(arr):
      return tuple(arr.tolist())
    sent1 = to_tuple(np.array([2934, 12, 17, 9, 969, 125], dtype=np.int64))
    sent2 = to_tuple(np.array([8872, 32, 4700, 1467, 85, 125], dtype=np.int64))
    sent3 = to_tuple(np.array((73, 12, 192, 1879, 70, 125), dtype=np.int64))
    model_summ = {
        sent1: 'some summary',
        sent2: 'Ted studied hard for his finals.',
        sent3: 'blah'
    }

    m = e.compute_metrics(model_summ)
    self.assertIn('rouge-1', list(m.metrics.keys()))

  def test_metrics(self):
    m = p2s_eval.Metrics()
    m.add_metric('m2', 0.9)
    m.add_metric('m1', 0.5)
    self.assertEqual('m1,m2\n0.5,0.9\n', str(m))

  def test_get_summary_first_sentence(self):
    s = 'First sent. Second sent.'
    self.assertEqual('First sent.', p2s_eval.get_summary_first_sentence(s))
    s = 'First sent? Second sent, sent.'
    self.assertEqual('First sent?', p2s_eval.get_summary_first_sentence(s))
    s = 'First sent! '
    self.assertEqual('First sent!', p2s_eval.get_summary_first_sentence(s))
    s = 'First sent'
    self.assertEqual('First sent', p2s_eval.get_summary_first_sentence(s))

  def test_get_summary_truncated(self):
    s = 'this is a summary and a summary.'
    self.assertEqual('this is a summary', p2s_eval.get_summary_truncated(s, 4))
    s = 'this is a summary!'
    self.assertEqual('this is a summary!', p2s_eval.get_summary_truncated(s, 4))

  def test_get_summary_n_periods(self):
    s = 'this is a summary.'
    self.assertEqual(1, p2s_eval.get_summary_n_periods(s))
    s = 'first sent.Second sent?third sent! fourth. '
    self.assertEqual(4, p2s_eval.get_summary_n_periods(s))

  def test_count_words(self):
    s = 'this is a summary.'
    self.assertEqual(4, p2s_eval.count_words(s))
    s = 'this is a self-explaining summary.'
    # self-explaining is treated as two words
    self.assertEqual(6, p2s_eval.count_words(s))
    s = 'first sent, second sent.Third sent.'
    self.assertEqual(6, p2s_eval.count_words(s))
    s = "Mary's good,and Kate is good.Third sent."
    # note that Mary's will be treated as two words
    self.assertEqual(9, p2s_eval.count_words(s))


if __name__ == '__main__':
  absltest.main()
