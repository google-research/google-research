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

"""Tests for ...utils.sampling_utils."""

import numpy as np
import tensorflow as tf

from tf3d.utils import instance_sampling_utils


class SamplingUtilsTest(tf.test.TestCase):

  def get_test_semantic_scores(self):
    return tf.constant([[0.2, 0.1, 0.0, 1.0, 1.0],
                        [0.0, 0.5, 0.07, 1.2, 1.1],
                        [0.1, 0.43, 0.06, 0.4, 0.95],
                        [0.01, 0.04, 0.08, 0.25, 1.05]],
                       dtype=tf.float32)

  def get_test_embedding(self):
    embedding_dim0 = tf.constant([[0.0, 0.1, 0.0, 1.0, 1.0],
                                  [0.0, 0.05, 0.07, 1.2, 1.1],
                                  [-0.1, 0.03, -0.06, 0.4, 0.95],
                                  [0.01, 0.04, 0.08, 0.25, 1.05]],
                                 dtype=tf.float32)
    embedding_dim0 = tf.expand_dims(embedding_dim0, 2)
    embedding_dim1 = tf.constant([[-1.0, -1.1, -0.9, 0.04, 0.09],
                                  [-1.1, -1.05, -1.07, 0.2, -0.1],
                                  [-1.1, -1.03, -0.06, 0.4, 0.05],
                                  [0.01, 0.04, 0.08, 0.15, 0.05]],
                                 dtype=tf.float32)
    embedding_dim1 = tf.expand_dims(embedding_dim1, 2)
    embedding_dim2 = tf.constant([[0.0, 0.1, 0.0, 0.07, 0.1],
                                  [0.0, 0.05, 0.07, -0.2, -0.1],
                                  [0.1, 0.03, 1.06, 1.1, -0.05],
                                  [1.01, 1.04, 1.08, 1.15, 0.05]],
                                 dtype=tf.float32)
    embedding_dim2 = tf.expand_dims(embedding_dim2, 2)
    embedding = tf.concat([embedding_dim0, embedding_dim1, embedding_dim2],
                          axis=2)
    return embedding

  def test_sample_based_on_scores_and_distances(self):
    embedding = tf.reshape(self.get_test_embedding(), [-1, 3])
    scores = tf.reshape(self.get_test_semantic_scores(), [-1])
    samples, indices = instance_sampling_utils.sample_based_on_scores_and_distances(
        embedding, scores, 6, 1.0)
    expected_samples = tf.gather(embedding, indices)
    sample_scores = tf.gather(scores, indices)
    self.assertAllClose(expected_samples.numpy(), samples.numpy())
    self.assertAllEqual(indices.shape, np.array([6]))
    self.assertAllEqual(sample_scores.shape, np.array([6]))


if __name__ == '__main__':
  tf.test.main()
