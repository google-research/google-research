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

"""Tests for analyze_mobile_search_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from tunas import analyze_mobile_search_lib


class AnalyzeMobileSearchLibTest(tf.test.TestCase):

  def test_read_tag_logits_with_pattern_v1(self):
    # Create fake logs for the read_tag_logits() function to consume.
    tempdir = self.get_temp_dir()
    writer = tf2.summary.create_file_writer(tempdir, max_queue=0)
    with writer.as_default():
      # Events matching pattern v1.
      tf2.summary.scalar('rllogits/0_0', 1.0, step=42)
      tf2.summary.scalar('rllogits/0_1', 2.0, step=42)
      tf2.summary.scalar('rllogits/0_2', 3.0, step=42)
      tf2.summary.scalar('rllogits/1_0', 4.0, step=42)
      tf2.summary.scalar('rllogits/1_1', 5.0, step=42)
      tf2.summary.scalar('rllogits/1_2', 6.0, step=42)
      tf2.summary.scalar('rlfilterslogits/0_0', 7.0, step=42)
      # Events not matching any pattern.
      tf2.summary.scalar('global_step/sec', 10.0, step=42)

    self.evaluate(writer.init())
    self.evaluate(tf.summary.all_v2_summary_ops())
    self.evaluate(writer.flush())

    # Try to read the events from file.
    self.assertAllClose(
        analyze_mobile_search_lib.read_tag_logits(tempdir),
        {
            42: {
                'rllogits': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                'rlfilterslogits': [[7.0]],
            }
        })

  def test_read_tag_logits_with_pattern_v2(self):
    # Create fake logs for the read_tag_logits() function to consume.
    tempdir = self.get_temp_dir()
    writer = tf2.summary.create_file_writer(tempdir, max_queue=0)
    with writer.as_default():
      # Events matching pattern v2.
      tf2.summary.scalar('rltaglogits/op_indices_0/0', 1.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_0/1', 2.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_0/2', 3.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_1/0', 4.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_1/1', 5.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_1/2', 6.0, step=42)
      tf2.summary.scalar('rltaglogits/filters_indices_0/0', 7.0, step=42)
      # Events not matching any pattern.
      tf2.summary.scalar('global_step/sec', 10.0, step=42)

    self.evaluate(writer.init())
    self.evaluate(tf.summary.all_v2_summary_ops())
    self.evaluate(writer.flush())

    # Try to read the events from file.
    self.assertAllClose(
        analyze_mobile_search_lib.read_tag_logits(tempdir),
        {
            42: {
                'op_indices': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                'filters_indices': [[7.0]],
            }
        })

  def test_read_tag_logits_with_two_valid_steps(self):
    # Create fake logs for the read_tag_logits() function to consume.
    tempdir = self.get_temp_dir()
    writer = tf2.summary.create_file_writer(tempdir, max_queue=0)
    values = tf.placeholder(tf.float32, [8])
    global_step = tf.placeholder(tf.int64, ())
    with writer.as_default():
      tf2.summary.scalar(
          'rltaglogits/op_indices_0/0', values[0], step=global_step)
      tf2.summary.scalar(
          'rltaglogits/op_indices_0/1', values[1], step=global_step)
      tf2.summary.scalar(
          'rltaglogits/op_indices_0/2', values[2], step=global_step)
      tf2.summary.scalar(
          'rltaglogits/op_indices_1/0', values[3], step=global_step)
      tf2.summary.scalar(
          'rltaglogits/op_indices_1/1', values[4], step=global_step)
      tf2.summary.scalar(
          'rltaglogits/op_indices_1/2', values[5], step=global_step)
      tf2.summary.scalar(
          'rltaglogits/filters_indices_0/0', values[6], step=global_step)
      tf2.summary.scalar(
          'rltaglogits/filters_indices_0/1', values[7], step=global_step)

    self.evaluate(writer.init())
    summary_op = tf.summary.all_v2_summary_ops()
    flush_op = writer.flush()
    with self.cached_session() as sess:
      sess.run(summary_op, {
          values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          global_step: 31,
      })
      sess.run(flush_op)

      sess.run(summary_op, {
          values: [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
          global_step: 42,
      })
      sess.run(flush_op)

    # Now check that read_tag_logits() processes the events correctly.
    self.assertAllClose(
        analyze_mobile_search_lib.read_tag_logits(tempdir),
        {
            31: {
                'op_indices': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                'filters_indices': [[7.0, 8.0]],
            },
            42: {
                'op_indices': [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                'filters_indices': [[15.0, 16.0]],
            },
        })

  def test_read_tag_logits_with_invalid_entry(self):
    # Create fake logs for the read_tag_logits() function to consume.
    tempdir = self.get_temp_dir()
    writer = tf2.summary.create_file_writer(tempdir, max_queue=0)
    with writer.as_default():
      tf2.summary.scalar('rltaglogits/op_indices_0/0', 1.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_0/1', 2.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_0/2', 3.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_1/0', 4.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_1/1', 5.0, step=42)
      tf2.summary.scalar('rltaglogits/op_indices_1/2', 6.0, step=42)
      # 'rltaglogits/filters_indices_0/1' is missing from the logs.
      tf2.summary.scalar('rltaglogits/filters_indices_0/1', 8.0, step=42)

    self.evaluate(writer.init())
    self.evaluate(tf.summary.all_v2_summary_ops())
    self.evaluate(writer.flush())

    # Try to read the events from file. The events from Step 42 should be
    # skipped, since some of the data is incomplete.
    self.assertEmpty(analyze_mobile_search_lib.read_tag_logits(tempdir))

  def test_read_path_logits(self):
    # Create fake logs for the read_path_logits() function to consume.
    tempdir = self.get_temp_dir()
    writer = tf2.summary.create_file_writer(tempdir, max_queue=0)
    with writer.as_default():
      # Events matching the path logits pattern
      tf2.summary.scalar('rlpathlogits/filters/0', 1.0, step=42)
      tf2.summary.scalar('rlpathlogits/filters/1', 2.0, step=42)
      tf2.summary.scalar('rlpathlogits/filters/2', 3.0, step=42)
      tf2.summary.scalar('rlpathlogits/layers/0/choices/0', 4.0, step=42)
      tf2.summary.scalar('rlpathlogits/layers/0/choices/1', 5.0, step=42)
      tf2.summary.scalar('rlpathlogits/layers/0/choices/2', 6.0, step=42)
      # Events not matching any pattern.
      tf2.summary.scalar('global_step/sec', 10.0, step=42)

    self.evaluate(writer.init())
    self.evaluate(tf.summary.all_v2_summary_ops())
    self.evaluate(writer.flush())

    # Try to read the events from file.
    self.assertAllClose(
        analyze_mobile_search_lib.read_path_logits(tempdir),
        {
            42: {
                'filters': [1.0, 2.0, 3.0],
                'layers/0/choices': [4.0, 5.0, 6.0],
            }
        })

  def test_read_path_logits_with_invalid_entry(self):
    # Create fake logs for the read_path_logits() function to consume.
    tempdir = self.get_temp_dir()
    writer = tf2.summary.create_file_writer(tempdir, max_queue=0)
    with writer.as_default():
      # Events matching the path logits pattern
      tf2.summary.scalar('rlpathlogits/filters/0', 1.0, step=42)
      tf2.summary.scalar('rlpathlogits/filters/2', 3.0, step=42)
      tf2.summary.scalar('rlpathlogits/layers/0/choices/0', 4.0, step=42)
      tf2.summary.scalar('rlpathlogits/layers/0/choices/1', 5.0, step=42)
      tf2.summary.scalar('rlpathlogits/layers/0/choices/2', 6.0, step=42)
      # 'rlpathlogits/filters/1' is missing from the logs

    self.evaluate(writer.init())
    self.evaluate(tf.summary.all_v2_summary_ops())
    self.evaluate(writer.flush())

    # Try to read the events from file. The events from Step 42 should be
    # skipped, since some of the data is incomplete.
    self.assertEmpty(analyze_mobile_search_lib.read_path_logits(tempdir))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
