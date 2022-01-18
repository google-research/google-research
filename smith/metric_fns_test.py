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

import numpy as np
import tensorflow.compat.v1 as tf
from smith import metric_fns


class MetricFnsTest(tf.test.TestCase):

  def test_metric_fn_pretrain(self):
    masked_lm_example_loss_1 = tf.constant([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    masked_lm_weights_1 = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    masked_sent_per_example_loss_1 = tf.constant(
        [[0.3, 0.3, 0.1, 0.2, 0.2, 0.1]])
    masked_sent_weight_1 = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    masked_lm_example_loss_2 = tf.constant([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    masked_lm_weights_2 = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    masked_sent_per_example_loss_2 = tf.constant(
        [[0.3, 0.3, 0.1, 0.2, 0.2, 0.1]])
    masked_sent_weight_2 = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    labels = tf.constant([1, 0])
    predicted_class = tf.constant([0, 0])
    is_real_example = tf.constant([1.0, 1.0])
    metrics_dict = metric_fns.metric_fn_pretrain(
        masked_lm_example_loss_1=masked_lm_example_loss_1,
        masked_lm_weights_1=masked_lm_weights_1,
        masked_sent_per_example_loss_1=masked_sent_per_example_loss_1,
        masked_sent_weight_1=masked_sent_weight_1,
        masked_lm_example_loss_2=masked_lm_example_loss_2,
        masked_lm_weights_2=masked_lm_weights_2,
        masked_sent_per_example_loss_2=masked_sent_per_example_loss_2,
        masked_sent_weight_2=masked_sent_weight_2,
        predicted_class=predicted_class,
        labels=labels,
        is_real_example=is_real_example)
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
      sess.run([init_g, init_l])
      # Runs update_op in metrics before checking the values.
      sess.run(metrics_dict)
      metrics_dict_numpy = sess.run(metrics_dict)
      self.assertEqual(metrics_dict_numpy["masked_lm_loss_1"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["masked_lm_loss_1"][1], 0.1)
      self.assertDTypeEqual(metrics_dict_numpy["masked_lm_loss_1"][1],
                            np.float32)

      self.assertEqual(metrics_dict_numpy["masked_lm_loss_2"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["masked_lm_loss_2"][1], 0.1)
      self.assertDTypeEqual(metrics_dict_numpy["masked_lm_loss_2"][1],
                            np.float32)

      self.assertEqual(metrics_dict_numpy["masked_sent_lm_loss_1"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["masked_sent_lm_loss_1"][1], 0.2)
      self.assertDTypeEqual(metrics_dict_numpy["masked_sent_lm_loss_1"][1],
                            np.float32)

      self.assertEqual(metrics_dict_numpy["masked_sent_lm_loss_2"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["masked_sent_lm_loss_2"][1], 0.2)
      self.assertDTypeEqual(metrics_dict_numpy["masked_sent_lm_loss_2"][1],
                            np.float32)

      self.assertEqual(metrics_dict_numpy["accuracy"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["accuracy"][1], 0.5)
      self.assertDTypeEqual(metrics_dict_numpy["accuracy"][1],
                            np.float32)

  def test_metric_fn_finetune_binary_classification(self):
    labels = tf.constant([1, 0, 1, 1])
    predicted_class = tf.constant([0, 0, 0, 1])
    siamese_example_loss = tf.constant([0.1, 0.2, 0.3, 0.4])
    is_real_example = tf.constant([1.0, 1.0, 1.0, 1.0])
    metrics_dict = metric_fns.metric_fn_finetune(
        predicted_class=predicted_class,
        labels=labels,
        siamese_example_loss=siamese_example_loss,
        is_real_example=is_real_example)
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
      sess.run([init_g, init_l])
      # Runs update_op in metrics before checking the values.
      sess.run(metrics_dict)
      metrics_dict_numpy = sess.run(metrics_dict)
      self.assertEqual(metrics_dict_numpy["accuracy"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["accuracy"][1], 0.5)
      self.assertDTypeEqual(metrics_dict_numpy["accuracy"][1], np.float32)

      self.assertEqual(metrics_dict_numpy["precision"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["precision"][1], 1)
      self.assertDTypeEqual(metrics_dict_numpy["precision"][1], np.float32)

      self.assertEqual(metrics_dict_numpy["recall"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["recall"][1], 0.333333)
      self.assertDTypeEqual(metrics_dict_numpy["recall"][1], np.float32)

      self.assertEqual(metrics_dict_numpy["siamese_loss"][1].shape, ())
      self.assertAllClose(metrics_dict_numpy["siamese_loss"][1], 0.25)
      self.assertDTypeEqual(metrics_dict_numpy["siamese_loss"][1], np.float32)

if __name__ == "__main__":
  tf.test.main()
