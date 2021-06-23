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

"""Tests for protein_model.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from absl.testing import parameterized
import numpy as np
import protein_model
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class ProteinModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='float values',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[2, 3],
          expected=[[[11], [21], [0]], [[41], [51], [61]]],
          sentinel=0.,
      ),
      dict(
          testcase_name='no padding',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[3, 3],
          expected=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sentinel=0.,
      ),
      dict(
          testcase_name='all padding',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[0, 0],
          expected=[[[0.], [0.], [0.]], [[0.], [0.], [0.]]],
          sentinel=0.,
      ),
      dict(
          testcase_name='different sentinel',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[0, 0],
          expected=[[[-99.], [-99.], [-99.]], [[-99.], [-99.], [-99.]]],
          sentinel=-99.,
      ),
      dict(
          testcase_name='embedding dimension size > 1',
          padded_representations=[[[11., -1.], [21., -2.], [31., -3.]],
                                  [[41., -4.], [51., -5.], [61., -6.]]],
          sequence_lengths=[2, 3],
          expected=[[[11., -1.], [21., -2.], [0., 0.]],
                    [[41., -4.], [51., -5.], [61., -6.]]],
          sentinel=0.,
      ),
  )
  def testSetPaddingToSentinel(self, padded_representations, sequence_lengths,
                               expected, sentinel):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        padded_representations = tf.convert_to_tensor(padded_representations)
        sequence_lengths = tf.convert_to_tensor(sequence_lengths)
        actual = sess.run(
            protein_model._set_padding_to_sentinel(padded_representations,
                                                   sequence_lengths, sentinel))
        np.testing.assert_array_almost_equal(actual, expected)


if __name__ == '__main__':
  tf.test.main()
