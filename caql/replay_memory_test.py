# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for replay_memory."""

import os.path

from absl import flags
import tensorflow.compat.v1 as tf

from caql import replay_memory

FLAGS = flags.FLAGS


class ReplayMemoryTest(tf.test.TestCase):

  def setUp(self):
    super(ReplayMemoryTest, self).setUp()
    if tf.gfile.Exists(FLAGS.test_tmpdir):
      tf.gfile.DeleteRecursively(FLAGS.test_tmpdir)

  def testSize(self):
    rm = replay_memory.ReplayMemory('test', 13)
    self.assertEqual(rm.capacity, 13)
    rm.extend([i for i in range(7)])
    self.assertEqual(rm.size, 7)
    rm.clear()
    self.assertEqual(rm.capacity, 13)
    self.assertEqual(rm.size, 0)

  def testGetLatestCheckpointNumberFromNotExstingDirectory(self):
    rm = replay_memory.ReplayMemory('test', 13)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    self.assertEqual(rm._get_latest_checkpoint_number(checkpoint_dir_path), -1)

  def testGetLatestCheckpointNumberFromEmptyDirectory(self):
    rm = replay_memory.ReplayMemory('test', 13)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    tf.gfile.MakeDirs(checkpoint_dir_path)
    self.assertEqual(rm._get_latest_checkpoint_number(checkpoint_dir_path), -1)

  def testGetLatestCheckpointNumber(self):
    rm = replay_memory.ReplayMemory('test', 13)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    tf.gfile.MakeDirs(checkpoint_dir_path)
    file_path = os.path.join(checkpoint_dir_path, 'replay_memory-test.pkl-2')
    with tf.gfile.Open(file_path, 'w') as f:
      f.write('abc')
    file_path = os.path.join(checkpoint_dir_path, 'replay_memory-test.pkl-3')
    with tf.gfile.Open(file_path, 'w') as f:
      f.write('def')
    self.assertEqual(rm._get_latest_checkpoint_number(checkpoint_dir_path), 3)

  def testGetLatestCheckpointNumberIgnoreTemporaryFile(self):
    rm = replay_memory.ReplayMemory('test', 13)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    tf.gfile.MakeDirs(checkpoint_dir_path)
    file_path = os.path.join(checkpoint_dir_path, 'replay_memory-test.pkl-2')
    with tf.gfile.Open(file_path, 'w') as f:
      f.write('abc')
    file_path = os.path.join(checkpoint_dir_path,
                             'replay_memory-test.pkl-3.tmp')
    with tf.gfile.Open(file_path, 'w') as f:
      f.write('def')
    self.assertEqual(rm._get_latest_checkpoint_number(checkpoint_dir_path), 2)

  def testGetLatestCheckpointNumberOnlyTemporaryFile(self):
    rm = replay_memory.ReplayMemory('test', 13)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    tf.gfile.MakeDirs(checkpoint_dir_path)
    file_path = os.path.join(checkpoint_dir_path,
                             'replay_memory-test.pkl-1.tmp')
    with tf.gfile.Open(file_path, 'w') as f:
      f.write('abc')
    self.assertEqual(rm._get_latest_checkpoint_number(checkpoint_dir_path), -1)

  def testSaveFirstCheckpoint(self):
    rm = replay_memory.ReplayMemory('test', 13)
    rm.extend([i for i in range(7)])
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    rm.save(checkpoint_dir_path)
    self.assertTrue(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(0))))

  def testSaveFirstCheckpointAndDeleteOld(self):
    rm = replay_memory.ReplayMemory('test', 13)
    rm.extend([i for i in range(7)])
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    rm.save(checkpoint_dir_path, delete_old=True)
    self.assertTrue(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(0))))

  def testSaveCheckpointsAndDeleteOld(self):
    rm = replay_memory.ReplayMemory('test', 13)
    rm.extend([i for i in range(7)])
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    rm.save(checkpoint_dir_path)
    self.assertTrue(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(0))))
    rm.extend([i for i in range(7, 13)])
    rm.save(checkpoint_dir_path)
    self.assertTrue(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(0))))
    self.assertTrue(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(1))))
    rm.extend([i for i in range(13, 5)])
    rm.save(checkpoint_dir_path, delete_old=True)
    self.assertTrue(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(0))))
    self.assertFalse(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(1))))
    self.assertTrue(tf.gfile.Exists(os.path.join(
        checkpoint_dir_path, rm._get_checkpoint_filename(2))))

  def testRestoreButNoCheckpoint(self):
    rm = replay_memory.ReplayMemory('test', 13)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    self.assertIsNone(rm.restore(checkpoint_dir_path))

  def testSingleCheckpoint(self):
    rm = replay_memory.ReplayMemory('test', 5)
    experiences = [['a', 0, 0.0], ['b', 1, 0.1], ['c', 2, 0.2], ['d', 3, 0.3]]
    rm.extend(experiences)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    rm.save(checkpoint_dir_path)
    rm.clear()
    rm.restore(checkpoint_dir_path)
    for experience in experiences:
      self.assertIn(experience, rm._buffer)

  def testMultiCheckpoints(self):
    rm = replay_memory.ReplayMemory('test', 5)
    experiences = [['a', 0, 0.0], ['b', 1, 0.1], ['c', 2, 0.2], ['d', 3, 0.3]]
    rm.extend(experiences)
    checkpoint_dir_path = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    rm.save(checkpoint_dir_path)
    rm.clear()
    rm.restore(checkpoint_dir_path)
    for experience in experiences:
      self.assertIn(experience, rm._buffer)
    experiences2 = [['e', 4, 0.4], ['f', 5, 0.5], ['g', 6, 0.6]]
    rm.extend(experiences2)
    rm.save(checkpoint_dir_path)
    rm.clear()
    rm.restore(checkpoint_dir_path)
    for experience in experiences[2:] + experiences2:
      self.assertIn(experience, rm._buffer)


if __name__ == '__main__':
  tf.test.main()
