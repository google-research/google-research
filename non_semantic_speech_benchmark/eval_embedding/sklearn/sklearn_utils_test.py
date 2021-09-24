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

# Lint as: python3
"""Tests for non_semantic_speech_benchmark.eval_embedding.sklearn.sklearn_utils."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
from non_semantic_speech_benchmark.eval_embedding.sklearn import sklearn_utils


class SklearnUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'l2_normalization': True},
      {'l2_normalization': False},
  )
  def test_tfexample_to_nps(self, l2_normalization):
    path = os.path.join(absltest.get_default_test_tmpdir(), 'dummy_tfrecords')
    embedding_name = 'fake_emb'
    label_name = 'label/fake_lbl'
    label_list = ['yes', 'no']

    np.random.seed(10)
    # Generate fake embeddings and labels.

    fake_data = [
        (np.random.rand(100), 1),
        (np.random.rand(100), 0),
        (np.random.rand(100), 1),
    ]
    def _emb_lbl_i_to_tfexample(emb, label_index):
      """Package fake data as a tf.Example."""
      ex = tf.train.Example()
      ex.features.feature[
          f'embedding/{embedding_name}'].float_list.value.extend(emb)
      ex.features.feature[label_name].bytes_list.value.append(
          label_list[label_index].encode('utf-8'))
      return ex

    # Write TFRecord of tf.Examples to disk.
    with tf.python_io.TFRecordWriter(path) as writer:
      for emb, label_index in fake_data:
        ex = _emb_lbl_i_to_tfexample(emb, label_index)
        writer.write(ex.SerializeToString())

    # Convert them back.
    npx, npy, _ = sklearn_utils.tfexamples_to_nps(path, embedding_name,
                                                  label_name, label_list,
                                                  l2_normalization)

    # Check that output is correct.
    expected_embs = np.array([d[0] for d in fake_data], np.float32)
    if l2_normalization:
      expected_embs /= np.linalg.norm(expected_embs, axis=1, ord=2,
                                      keepdims=True)
    self.assertAllEqual(npx, expected_embs)
    self.assertAllEqual(npy, (1, 0, 1))

  def test_speaker_normalization(self):
    original_embeddings = np.array(
        [
            [0.5, 2.1],
            [0.5, 0.6],
            [-.5, 2.1],
        ],
        np.float32)
    speaker_ids = np.array(
        [
            'id1',
            'id2',
            'id1',
        ],
        np.str)
    expected_normalized_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [-1., 0.0],
        ],
        np.float32)
    normalized_embeddings = sklearn_utils._speaker_normalization(
        original_embeddings, speaker_ids)
    self.assertAllEqual(normalized_embeddings, expected_normalized_embeddings)


if __name__ == '__main__':
  tf.test.main()
