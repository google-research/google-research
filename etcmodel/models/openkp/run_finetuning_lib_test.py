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

"""Tests for run_finetuning_lib."""

import tensorflow.compat.v1 as tf

from etcmodel.models.openkp import run_finetuning_lib


class RunFinetuningLibTest(tf.test.TestCase):

  def test_dense_feature_scaler(self):
    tensor = tf.constant([-15., -12., -20., 5., -100.])
    expected = [0, 0.6, -1., 1., -1.]

    scaler = run_finetuning_lib.DenseFeatureScaler(min_value=-20, max_value=-10)
    self.assertAllClose(expected, scaler.transform(tensor))

  def test_dense_feature_scaler_invalid_range(self):
    with self.assertRaises(ValueError):
      run_finetuning_lib.DenseFeatureScaler(min_value=10, max_value=5)

    with self.assertRaises(ValueError):
      run_finetuning_lib.DenseFeatureScaler(min_value=10, max_value=10)

  def test_indicators_to_id(self):
    indicator1 = tf.constant([0, 1, 0, 1], dtype=tf.int32)
    indicator2 = tf.constant([1, 0, 0, 1], dtype=tf.int32)
    indicator3 = tf.constant([0, 1, 1, 1], dtype=tf.int32)

    expected = [2, 5, 1, 7]
    self.assertAllEqual(
        expected,
        run_finetuning_lib.indicators_to_id(indicator1, indicator2, indicator3))

  def test_gather_global_embeddings_to_long(self):
    global_embeddings = [
        [
            [.1, -.1],
            [.2, -.2],
            [.3, -.3],
        ],  #
        [
            [1.1, -1.1],
            [1.2, -1.2],
            [1.3, -1.3],
        ],  #
        [
            [2.1, -2.1],
            [2.2, -2.2],
            [2.3, -2.3],
        ],  #
    ]
    long_vdom_idx = [
        [0, 1, 1, 2, 2],  #
        [0, 0, 0, 1, 2],  #
        [0, 1, 2, 0, 0],  #  Padding can be 0 since their embedding is ignored.
    ]

    expected = [
        [
            [.1, -.1],
            [.2, -.2],
            [.2, -.2],
            [.3, -.3],
            [.3, -.3],
        ],  #
        [
            [1.1, -1.1],
            [1.1, -1.1],
            [1.1, -1.1],
            [1.2, -1.2],
            [1.3, -1.3],
        ],  #
        [
            [2.1, -2.1],
            [2.2, -2.2],
            [2.3, -2.3],
            [2.1, -2.1],
            [2.1, -2.1],
        ],  #
    ]

    self.assertAllClose(
        expected,
        run_finetuning_lib.gather_global_embeddings_to_long(
            global_embeddings, long_vdom_idx))

  def test_batch_segment_sum_embeddings(self):
    # batch_size = 2
    # long_max_length = 8
    # hidden_size = 2

    long_embeddings = tf.constant(
        [
            [
                [0.1, -0.1],
                [0.2, -0.2],
                [0.3, -0.3],
                [0.4, -0.4],
                [0.5, -0.5],
                [100.0, -100.0],  # Padding embeddings may be arbitrary.
                [200.0, -200.0],
                [300.0, -300.0],
            ],  #
            [
                [1.1, -1.1],
                [1.2, -1.2],
                [1.3, -1.3],
                [1.4, -1.4],
                [1.5, -1.5],
                [1.6, -1.6],
                [400.0, 400.0],  # Padding embeddings may be arbitrary.
                [500.0, 500.0],
            ],  #
        ],
        dtype=tf.float32)

    long_word_idx = tf.constant(
        [
            [0, 1, 2, 2, 3, 0, 0, 0],  # Padding indices can just be 0.
            [0, 0, 0, 1, 2, 2, 0, 0],  # Padding indices can just be 0.
        ],
        dtype=tf.int32)

    long_input_mask = tf.constant(
        [
            [1, 1, 1, 1, 1, 0, 0, 0],  #
            [1, 1, 1, 1, 1, 1, 0, 0],  #
        ],
        dtype=tf.int32)

    expected = [
        [
            [0.1, -0.1],
            [0.2, -0.2],
            [0.7, -0.7],
            [0.5, -0.5],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],  #
        [
            [3.6, -3.6],
            [1.4, -1.4],
            [3.1, -3.1],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],  #
    ]

    self.assertAllClose(
        expected,
        run_finetuning_lib.batch_segment_sum_embeddings(
            long_embeddings=long_embeddings,
            long_word_idx=long_word_idx,
            long_input_mask=long_input_mask))

  def test_make_ngram_labels(self):
    label_start_idx = tf.constant([
        [1, -1, -1],
        [2, 3, 0],
    ])

    label_phrase_len = tf.constant([
        [3, -1, -1],
        [2, 1, 2],
    ])

    long_max_length = 4
    kp_max_length = 5

    reshaped_expected = [
        [
            [0, 0, 0, 0],  # 1-grams
            [0, 0, 0, 0],  # 2-grams
            [0, 1, 0, 0],  # 3-grams
            [0, 0, 0, 0],  # 4-grams
            [0, 0, 0, 0],  # 5-grams
        ],
        [
            [0, 0, 0, 1 / 3],  # 1-grams
            [1 / 3, 0, 1 / 3, 0],  # 2-grams
            [0, 0, 0, 0],  # 3-grams
            [0, 0, 0, 0],  # 4-grams
            [0, 0, 0, 0],  # 5-grams
        ],
    ]
    batch_size = len(reshaped_expected)
    expected = tf.reshape(reshaped_expected,
                          [batch_size, kp_max_length * long_max_length])

    self.assertAllClose(
        expected,
        run_finetuning_lib.make_ngram_labels(
            label_start_idx=label_start_idx,
            label_phrase_len=label_phrase_len,
            long_max_length=long_max_length,
            kp_max_length=kp_max_length))

  def test_make_ngram_labels_additive_smoothing(self):
    label_start_idx = tf.constant([
        [1, -1, -1],
        [2, 3, 0],
    ])

    label_phrase_len = tf.constant([
        [3, -1, -1],
        [2, 1, 2],
    ])

    long_max_length = 4
    kp_max_length = 5
    additive_smoothing_mass = 1.0

    smoothed_third = (1 / 3) + 0.05
    reshaped_unnormalized_expected = [
        [
            [0.05, 0.05, 0.05, 0.05],  # 1-grams
            [0.05, 0.05, 0.05, 0.05],  # 2-grams
            [0.05, 1.05, 0.05, 0.05],  # 3-grams
            [0.05, 0.05, 0.05, 0.05],  # 4-grams
            [0.05, 0.05, 0.05, 0.05],  # 5-grams
        ],
        [
            [0.05, 0.05, 0.05, smoothed_third],  # 1-grams
            [smoothed_third, 0.05, smoothed_third, 0.05],  # 2-grams
            [0.05, 0.05, 0.05, 0.05],  # 3-grams
            [0.05, 0.05, 0.05, 0.05],  # 4-grams
            [0.05, 0.05, 0.05, 0.05],  # 5-grams
        ],
    ]
    batch_size = len(reshaped_unnormalized_expected)
    unnormalized_expected = tf.reshape(
        reshaped_unnormalized_expected,
        [batch_size, kp_max_length * long_max_length])
    expected = (
        unnormalized_expected /
        tf.reduce_sum(unnormalized_expected, axis=-1, keepdims=True))

    self.assertAllClose(
        expected,
        run_finetuning_lib.make_ngram_labels(
            label_start_idx=label_start_idx,
            label_phrase_len=label_phrase_len,
            long_max_length=long_max_length,
            kp_max_length=kp_max_length,
            additive_smoothing_mass=additive_smoothing_mass))


if __name__ == '__main__':
  tf.test.main()
