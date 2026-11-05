# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

import tensorflow.compat.v2 as tf

from representation_batch_rl.representation_batch_rl import logging_module
from representation_batch_rl.representation_batch_rl import test_utils


class _LoggingModule(tf.Module, logging_module.LoggingModule):

  def dummy_function(self, tensor):
    self.log('test', tf.reduce_mean(tensor))


class _OtherModule(tf.Module):

  def __init__(self):
    self._inner_module = _LoggingModule()

  def dummy_function(self, tensor):
    self._inner_module.dummy_function(tensor)


class _WrappingModule(tf.Module):

  def __init__(self):
    self.other_module = _OtherModule()

  @tf.function
  def step(self, tensor):
    with logging_module.LoggingTape(self.other_module) as logged_tensors:
      self.other_module.dummy_function(tensor)
    return logged_tensors


class LoggingModuleTest(test_utils.TestCase):

  def test_log_no_op(self):
    """Tests that the default no-op log method."""
    module = _LoggingModule()
    module.dummy_function(tf.zeros((20, 10)))

  def test_log_to_dict(self):
    """Tests that we can log to an external dictionary."""
    my_dict = {}
    tensor = tf.range((200), dtype=tf.float32)
    module = _LoggingModule()
    module.set_logging_dict(my_dict)
    module.dummy_function(tensor)
    module.unset_logging_dict()

    self.assertAllClose(my_dict['test'], tf.reduce_mean(tensor))


class LoggingTapeTest(test_utils.TestCase):

  def test_tf_function(self):
    """Tests that by default the log method is a no-op."""
    module = _WrappingModule()
    tensor1 = tf.range((200), dtype=tf.float32)
    tensor2 = tf.ones((200), dtype=tf.float32)
    self.assertAllClose(module.step(tensor1)['test'], tf.reduce_mean(tensor1))
    self.assertAllClose(module.step(tensor2)['test'], tf.reduce_mean(tensor2))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
