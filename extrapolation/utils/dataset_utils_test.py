# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

import numpy as np
import tensorflow as tf
from extrapolation.utils import dataset_utils as du




class DatasetUtilsTest(tf.test.TestCase):

  def test_filter_label_fn_filters_correct_classes(self):
    filter_classes = [[0, 1, 2, 3, 4, 5, 6, 7, 9],
                      [1, 2, 3],
                      [1],
                      [1, 3, 5, 7, 9]]
    labels_keep = [2, 3, 1, 5]
    labels_remove = [8, 4, 2, 0]
    for cl, lab_keep, lab_remove in zip(filter_classes, labels_keep,
                                        labels_remove):
      filt = du.filter_label_function(cl)
      x_keep = {"label": lab_keep}
      self.assertTrue(filt(x_keep))
      x_remove = {"label": lab_remove}
      self.assertFalse(filt(x_remove))

  def test_filter_example_fn_identifies_correct_examples(self):
    shape = (10, 10)
    remove_examples = [tf.zeros(shape), tf.random.normal(shape)]
    x_keep = [x + 1. for x in remove_examples]
    x_remove = [x for x in remove_examples]
    for ex, x_notequal, x_equal in zip(remove_examples, x_keep, x_remove):
      filt = du.filter_example_function(ex)
      self.assertTrue(filt(x_notequal, None).numpy())
      self.assertFalse(filt(x_equal, None).numpy())

  def test_get_hash_is_consistent_and_in_correct_range(self):
    for _ in range(10):
      shape = (10, 10)
      b = 20
      x = tf.random.normal(shape)
      hash_value = du.get_hash(x, b)
      hash_value_again = du.get_hash(x, b)
      self.assertAllEqual(hash_value, hash_value_again)
      self.assertLess(hash_value, b)

  def test_get_hash_filter_fn_returns_correct_hashes(self):
    keep_buckets = [0, 1, 2]
    shape = (10, 10)
    b = 5
    filt = du.get_hash_filter_function(b, keep_buckets)
    for _ in range(20):
      x = {"image": tf.random.normal(shape), "label": 0}
      # The function filt should just wrap this operation.
      xinp = tf.concat([tf.reshape(tf.cast(x["image"], tf.float32), [-1, 1]),
                        tf.reshape(tf.cast(x["label"], tf.float32), [-1, 1])],
                       axis=0)
      hash_value = du.get_hash(xinp, b).numpy()
      keep_hash = filt(x)
      if hash_value in keep_buckets:
        self.assertTrue(keep_hash)
      else:
        self.assertFalse(keep_hash)

  def test_process_image_tuple_fn_works(self):
    img = tf.math.minimum(255 * tf.abs(tf.random.normal((10, 10))), 255.)
    lab = 0
    x = {"image": img, "label": lab}
    threshold = 0.7
    proc_tuple_fn = du.process_image_tuple_function(threshold)
    proc_tuple_img, proc_tuple_lab = proc_tuple_fn(x)
    self.assertEqual(lab, proc_tuple_lab)
    self.assertAllClose(proc_tuple_img, tf.square(proc_tuple_img))
    self.assertAllClose(1., tf.reduce_max(proc_tuple_img))
    self.assertAllClose(0., tf.reduce_min(proc_tuple_img))

    proc_img_fn = du.process_image_function(threshold)
    proc_img = proc_img_fn(x)
    self.assertAllClose(proc_img, proc_tuple_img)

  def test_make_labels_noisy_fn_makes_labels_adequately_noisy(self):
    n_classes = 10
    tf.compat.v1.random.set_random_seed(0)

    for noise_prob in [0., 0.1, 0.3, 0.5]:
      noise_fn = du.make_labels_noisy_function(noise_prob)
      added_noise = []
      for _ in range(10000):
        one_hot_lab = [0. for _ in range(n_classes)]
        lab = np.random.randint(n_classes)
        one_hot_lab[lab] = 1.
        one_hot_lab = tf.constant(one_hot_lab)
        _, noisy_one_hot_lab = noise_fn(None, one_hot_lab)
        was_noisy = tf.reduce_any(tf.not_equal(noisy_one_hot_lab, one_hot_lab))
        added_noise.append(was_noisy.numpy())
      avg_noise = np.mean(added_noise)
      if noise_prob == 0.:
        self.assertEqual(avg_noise, 0.)
      self.assertLessEqual(np.abs(noise_prob - avg_noise), 0.1)

  def test_make_one_hot_works(self):
    n_classes = 10
    for i in range(n_classes):
      one_hot_lab = du.make_onehot(i, n_classes)
      one_hot_lab_true = [0. for _ in range(n_classes)]
      one_hot_lab_true[i] = 1.
      one_hot_lab_true = tf.constant(one_hot_lab_true)
      self.assertAllClose(one_hot_lab, one_hot_lab_true)


if __name__ == "__main__":
  tf.test.main()
