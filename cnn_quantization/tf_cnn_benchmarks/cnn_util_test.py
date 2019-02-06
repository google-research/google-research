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

"""Tests for tf_cnn_benchmarks.cnn_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time

import tensorflow as tf

from cnn_quantization.tf_cnn_benchmarks import cnn_util


class CnnUtilBarrierTest(tf.test.TestCase):

  def testBarrier(self):
    num_tasks = 20
    num_waits = 4
    barrier = cnn_util.Barrier(num_tasks)
    threads = []
    sync_matrix = []
    for i in range(num_tasks):
      sync_times = [0] * num_waits
      thread = threading.Thread(
          target=self._run_task, args=(barrier, sync_times))
      thread.start()
      threads.append(thread)
      sync_matrix.append(sync_times)
    for thread in threads:
      thread.join()
    for wait_index in range(num_waits - 1):
      # Max of times at iteration i < min of times at iteration i + 1
      self.assertLessEqual(
          max([sync_matrix[i][wait_index] for i in range(num_tasks)]),
          min([sync_matrix[i][wait_index + 1] for i in range(num_tasks)]))

  def _run_task(self, barrier, sync_times):
    for wait_index in range(len(sync_times)):
      sync_times[wait_index] = time.time()
      barrier.wait()

  def testBarrierAbort(self):
    num_tasks = 2
    num_waits = 1
    sync_times = [0] * num_waits
    barrier = cnn_util.Barrier(num_tasks)
    thread = threading.Thread(
        target=self._run_task, args=(barrier, sync_times))
    thread.start()
    barrier.abort()
    # thread won't be blocked by done barrier.
    thread.join()


class ImageProducerTest(tf.test.TestCase):

  def _slow_tensorflow_op(self):
    """Returns a TensorFlow op that takes approximately 0.1s to complete."""
    def slow_func(v):
      time.sleep(0.1)
      return v
    return tf.py_func(slow_func, [tf.constant(0.)], tf.float32).op

  def _test_image_producer(self, batch_group_size, put_slower_than_get):
    # We use the variable x to simulate a staging area of images. x represents
    # the number of batches in the staging area.
    x = tf.Variable(0, dtype=tf.int32)
    if put_slower_than_get:
      put_dep = self._slow_tensorflow_op()
      get_dep = tf.no_op()
    else:
      put_dep = tf.no_op()
      get_dep = self._slow_tensorflow_op()
    with tf.control_dependencies([put_dep]):
      put_op = x.assign_add(batch_group_size, use_locking=True)
    with tf.control_dependencies([get_dep]):
      get_op = x.assign_sub(1, use_locking=True)
    with self.test_session() as sess:
      sess.run(tf.variables_initializer([x]))
      image_producer = cnn_util.ImageProducer(sess, put_op, batch_group_size,
                                              use_python32_barrier=False)
      image_producer.start()
      for _ in range(5 * batch_group_size):
        sess.run(get_op)
        # We assert x is nonnegative, to ensure image_producer never causes
        # an unstage op to block. We assert x is at most 2 * batch_group_size,
        # to ensure it doesn't use too much memory by storing too many batches
        # in the staging area.
        self.assertGreaterEqual(sess.run(x), 0)
        self.assertLessEqual(sess.run(x), 2 * batch_group_size)
        image_producer.notify_image_consumption()
        self.assertGreaterEqual(sess.run(x), 0)
        self.assertLessEqual(sess.run(x), 2 * batch_group_size)

      image_producer.done()
      time.sleep(0.1)
      self.assertGreaterEqual(sess.run(x), 0)
      self.assertLessEqual(sess.run(x), 2 * batch_group_size)

  def test_image_producer(self):
    self._test_image_producer(1, False)
    self._test_image_producer(1, True)
    self._test_image_producer(2, False)
    self._test_image_producer(2, True)
    self._test_image_producer(3, False)
    self._test_image_producer(3, True)
    self._test_image_producer(8, False)
    self._test_image_producer(8, True)


if __name__ == '__main__':
  tf.test.main()
