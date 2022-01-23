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

"""Tests for tf_pad_symmetric."""

from absl.testing import absltest
import numpy as np
import tensorflow as tf
from non_semantic_speech_benchmark.export_model import tf_pad


class SamplesToFeats(tf.keras.layers.Layer):
  """Compute features from samples."""

  def call(self, x, padding):
    return tf_pad.tf_pad(x, padding=padding, mode='SYMMETRIC')


class TfPadSymmetricTest(absltest.TestCase):

  def test_tf_pad(self):
    x = tf.expand_dims(tf.constant(range(3)), axis=0)

    y = tf_pad.tf_pad(x, padding=4, mode='CONSTANT')
    np.testing.assert_equal(y.numpy(), [[0, 1, 2, 0, 0, 0, 0]])

    y = tf_pad.tf_pad(x, padding=4, mode='SYMMETRIC')
    np.testing.assert_equal(y.numpy(), [[0, 1, 2, 2, 1, 0, 0]])

    y = tf_pad.tf_pad(x, padding=8, mode='SYMMETRIC')
    np.testing.assert_equal(y.numpy(), [[0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1]])

    x = tf.constant([[1, 2, 3],
                     [4, 5, 6]])
    y = tf_pad.tf_pad(x, padding=4, mode='SYMMETRIC')
    np.testing.assert_equal(y.numpy(),
                            [[1, 2, 3, 3, 2, 1, 1],
                             [4, 5, 6, 6, 5, 4, 4]])

  def test_keras_input_valid(self):
    model_in = tf.keras.Input((None,))
    SamplesToFeats()(model_in, padding=16000)

  def test_keras_lambda(self):
    def _pad_via_mapfn(x):
      map_fn = lambda y: tf_pad.tf_pad(tf.expand_dims(y, 0), padding=16000,  # pylint:disable=g-long-lambda
                                       mode='SYMMETRIC')
      return tf.map_fn(map_fn, x)
    model_in = tf.keras.Input((None,))
    tf.keras.layers.Lambda(_pad_via_mapfn)(model_in)


if __name__ == '__main__':
  absltest.main()
