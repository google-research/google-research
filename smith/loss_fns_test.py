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

import numpy as np
import tensorflow.compat.v1 as tf
from smith import loss_fns


class LossFnsTest(tf.test.TestCase):

  def test_get_prediction_loss_cosine(self):
    input_tensor_1 = tf.constant(
        [[0.5, 0.7, 0.8, 0.9, 0.1, 0.1], [0.1, 0.3, 0.3, 0.3, 0.1, 0.1]],
        dtype=tf.float32)
    input_tensor_2 = tf.constant(
        [[0.1, 0.2, 0.2, 0.2, 0.2, 0.1], [0.1, 0.4, 0.4, 0.4, 0.1, 0.1]],
        dtype=tf.float32)
    labels = tf.constant([0, 1.0], dtype=tf.float32)
    neg_to_pos_example_ratio = 1.0
    similarity_score_amplifier = 6.0
    loss, per_example_loss, similarities = \
        loss_fns.get_prediction_loss_cosine(
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            labels=labels,
            similarity_score_amplifier=similarity_score_amplifier,
            neg_to_pos_example_ratio=neg_to_pos_example_ratio)
    with tf.Session() as sess:
      sess.run([tf.global_variables_initializer()])
      loss_numpy = sess.run(loss)
      per_example_loss_numpy = sess.run(per_example_loss)
      similarities_numpy = sess.run(similarities)
      self.assertEqual(loss_numpy.shape, ())
      self.assertDTypeEqual(loss_numpy, np.float32)

      self.assertEqual(per_example_loss_numpy.shape, (2,))
      self.assertDTypeEqual(per_example_loss_numpy, np.float32)

      self.assertEqual(similarities_numpy.shape, (2,))
      self.assertDTypeEqual(similarities_numpy, np.float32)

if __name__ == '__main__':
  tf.test.main()
