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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from extrapolation.classifier import classifier
from extrapolation.utils import dataset_utils
from extrapolation.utils import tensor_utils


class TensorUtilsTest(tf.test.TestCase):

  def test_reshape_and_flatten(self):
    x_shape = (64, 28, 28, 1)
    y_shape = (64, 10)
    conv_dims = [20, 10]
    conv_sizes = [5, 5]
    dense_sizes = [100]
    n_classes = 10
    model = classifier.CNN(conv_dims, conv_sizes, dense_sizes,
                           n_classes, onehot=True)
    itr = dataset_utils.get_supervised_batch_noise_iterator(x_shape, y_shape)
    x, y = itr.next()
    _, _ = model.get_loss(x, y)

    w = model.weights
    num_wts = sum([tf.size(x) for x in w])
    v = tf.random.normal((10, num_wts))
    v_as_wts = tensor_utils.reshape_vector_as(w, v)

    for i in range(len(v_as_wts)):
      self.assertEqual(v_as_wts[i].shape[1:], w[i].shape)

    v_as_vec = tensor_utils.flat_concat(v_as_wts)
    self.assertAllClose(v, v_as_vec)

  def test_cosine_similarity(self):
    x = tf.fill((10, 10), 1.)
    y = tf.fill((10, 10), 2.)
    self.assertAllClose(tensor_utils.cosine_similarity(x, y), 1.)

    x = tf.fill((10, 10), 1.)
    y = tf.fill((10, 10), -2.)
    self.assertAllClose(tensor_utils.cosine_similarity(x, y), -1.)

    x = tf.concat([tf.random.normal((10, 10)), tf.fill((10, 10), 0.)], axis=0)
    y = tf.concat([tf.fill((10, 10), 0.), tf.random.normal((10, 10))], axis=0)
    self.assertAllClose(tensor_utils.cosine_similarity(x, y), 0.)

  def test_normalize_weight_shaped_vector(self):
    a = tf.random.normal((10, 20))
    w = [a[:, :5], a[:, 5:16], a[:, 16:]]
    a_normed = a / tf.math.reduce_euclidean_norm(a, axis=1, keepdims=True)
    w_normed = [a_normed[:, :5], a_normed[:, 5:16], a_normed[:, 16:]]
    expected_w_normed = tensor_utils.normalize_weight_shaped_vector(w)

    for i in range(len(w_normed)):
      self.assertAllClose(w_normed[i], expected_w_normed[i])


if __name__ == '__main__':
  tf.test.main()
