# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for utils."""

import time

from absl.testing import absltest
from internal import utils
import numpy as np


class UtilsTest(absltest.TestCase):

  def test_dummy_rays(self):
    """Ensures that the dummy Rays object is correctly initialized."""
    rays = utils.dummy_rays()
    self.assertEqual(rays.origins.shape[-1], 3)

  def test_invalid_stepfun_raises_exception(self):
    t = np.ones(10)
    y = np.ones(10)
    self.assertRaises(ValueError, lambda: utils.assert_valid_stepfun(t, y))

  def test_invalid_linspline_raises_exception(self):
    t = np.ones(10)
    y = np.ones(11)
    self.assertRaises(ValueError, lambda: utils.assert_valid_linspline(t, y))

  def test_iterate_in_separate_thread_iterates_simple_function(self):
    @utils.iterate_in_separate_thread(queue_size=100)
    def sample_fn():
      for i in range(4):
        yield i

    sample_fn_generator = sample_fn()
    self.assertEqual(next(sample_fn_generator), 0)
    time.sleep(0.1)
    self.assertEqual(next(sample_fn_generator), 1)
    time.sleep(0.1)
    self.assertEqual(next(sample_fn_generator), 2)
    time.sleep(0.1)
    self.assertEqual(next(sample_fn_generator), 3)

  def test_iterate_in_separate_thread_iterates_simple_function_limited_queue_size(
      self,
  ):
    @utils.iterate_in_separate_thread(queue_size=1)
    def sample_fn():
      for i in range(4):
        yield i

    sample_fn_generator = sample_fn()
    self.assertEqual(next(sample_fn_generator), 0)
    time.sleep(0.1)
    self.assertEqual(next(sample_fn_generator), 1)
    time.sleep(0.1)
    self.assertEqual(next(sample_fn_generator), 2)
    time.sleep(0.1)
    self.assertEqual(next(sample_fn_generator), 3)

  def test_iterate_in_separate_thread_raises_exception(
      self,
  ):
    @utils.iterate_in_separate_thread(queue_size=10)
    def sample_fn():
      for i in range(3):
        yield i
      raise ValueError('Test error')

    sample_fn_generator = sample_fn()
    self.assertEqual(next(sample_fn_generator), 0)
    self.assertEqual(next(sample_fn_generator), 1)
    self.assertEqual(next(sample_fn_generator), 2)
    self.assertRaises(ValueError, lambda: next(sample_fn_generator))

  def test_iterate_in_separate_thread_raises_exception_after_waiting(self):
    @utils.iterate_in_separate_thread(queue_size=10)
    def sample_fn():
      for i in range(3):
        yield i
      time.sleep(1)
      raise ValueError('Test error')

    sample_fn_generator = sample_fn()
    self.assertEqual(next(sample_fn_generator), 0)
    self.assertEqual(next(sample_fn_generator), 1)
    self.assertEqual(next(sample_fn_generator), 2)
    self.assertRaises(ValueError, lambda: next(sample_fn_generator))


if __name__ == '__main__':
  absltest.main()
