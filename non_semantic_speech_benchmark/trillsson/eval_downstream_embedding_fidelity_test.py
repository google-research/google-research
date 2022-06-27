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

"""Tests for eval_downstream."""

import os
from absl.testing import absltest
import mock
import numpy as np
import tensorflow as tf
from non_semantic_speech_benchmark.eval_embedding.sklearn import sklearn_utils
from non_semantic_speech_benchmark.trillsson import eval_downstream_embedding_fidelity


class EvalDownstreamEmbeddingFidelityTest(absltest.TestCase):

  def test_make_tfexample_and_write(self):
    expected = [[1., 2., 3.],
                [4., 5., 6.],
                [-1., -2., -3]]
    embs = tf.constant(expected, tf.float32)
    embeddings_output_dir = os.path.join(
        absltest.get_default_test_tmpdir(), 'embeddings2')
    tfrecord_writer = tf.io.TFRecordWriter(embeddings_output_dir)

    speaker_id = tf.constant('speakr')
    expected_lbl = ['1', '1', '2']
    lbls = tf.constant(expected_lbl, tf.string)
    for emb, lbl in zip(embs, lbls):
      eval_downstream_embedding_fidelity.make_tfexample_and_write(
          emb, lbl, speaker_id, 'speaker_id_name', tfrecord_writer)
    tfrecord_writer.close()
    self.assertTrue(tf.io.gfile.exists(embeddings_output_dir))

    # Check that they can be read correctly.
    embs_np, lbls_np, _ = sklearn_utils.tfexamples_to_nps(
        path=f'{embeddings_output_dir}*',
        embedding_name=eval_downstream_embedding_fidelity.EMBEDDING_KEY_,
        label_name=eval_downstream_embedding_fidelity.LABEL_KEY_,
        label_list=['1', '2'],
        l2_normalization=False)

    # Check correctness.
    np.testing.assert_array_equal(expected, embs_np)
    np.testing.assert_array_equal([0, 0, 1], lbls_np)

  @mock.patch.object(eval_downstream_embedding_fidelity.get_data, 'get_data')
  def test_get_splits_sanity(self, mock_get_data):
    mock_get_data.return_value = None
    eval_downstream_embedding_fidelity._get_splits(
        ['train', 'validation', 'test'],
        ['filepattern1', 'filepattern2', 'filepattern3'],
        os.path.join(absltest.get_default_test_tmpdir(), 'embeddings'),
        step=10)


if __name__ == '__main__':
  assert tf.executing_eagerly()
  absltest.main()
