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

# Lint as: python3
"""Tests for gfsa.datasets.data_loading."""

import itertools
import threading
import time

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.test_util
import numpy as np
from gfsa.datasets import data_loading


class DataLoadingTest(parameterized.TestCase):

  def test_randomly_interleave(self):
    # Just test that all of the elements are produced exactly once
    source = range(100)
    splits = [source[i:i + 10] for i in range(0, 100, 10)]
    interleaved = list(data_loading.randomly_interleave(splits, simultaneous=3))
    self.assertCountEqual(source, interleaved)

  def test_shuffle_with_buffer(self):
    # Just test that all of the elements are produced exactly once
    source = range(100)
    shuffled = list(data_loading.shuffle_with_buffer(source, buffer_size=10))
    self.assertCountEqual(source, shuffled)

  def test_repeat(self):
    i = 0

    def go():
      nonlocal i
      for j in range(5):
        yield (i, j)
      i += 1

    repeated = list(itertools.islice(data_loading.repeat(go), 10))
    expected = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2),
                (1, 3), (1, 4)]
    self.assertEqual(repeated, expected)

  def test_batch(self):
    values = [{"v": np.array([i])} for i in range(18)]
    batched = list(data_loading.batch(values, (3, 2)))
    expected = [
        {
            "v": np.array([[[0], [1]], [[2], [3]], [[4], [5]]])
        },
        {
            "v": np.array([[[6], [7]], [[8], [9]], [[10], [11]]])
        },
        {
            "v": np.array([[[12], [13]], [[14], [15]], [[16], [17]]])
        },
    ]
    jax.test_util.check_eq(batched, expected)

  def test_batch_uneven_error(self):
    values = range(10)
    with self.assertRaisesRegex(ValueError, "not divisible by batch size"):
      for _ in data_loading.batch(values, (3,)):
        pass

  def test_batch_uneven_pad(self):
    values = range(10)
    batched = list(
        data_loading.batch(
            values, (3,),
            remainder_behavior=data_loading.BatchRemainderBehavior.PAD_ZERO))
    expected = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9, 0, 0]),
    ]
    jax.test_util.check_eq(batched, expected)

  def test_batch_bucketed(self):
    values = [("a", (1,)), ("b", (2,)), ("c", (3,)), ("a", (4,)), ("b", (5,)),
              ("c", (6,)), ("a", (7,)), ("b", (8,)), ("c", (9,)), ("a", (10,)),
              ("b", (11,)), ("c", (12,))]
    batched = list(
        data_loading.batch_bucketed(
            values, {
                "a": (2,),
                "b": (3,),
                "c": (5,)
            },
            remainder_behavior=data_loading.BatchRemainderBehavior.PAD_ZERO))
    expected = [
        ("a", (np.array([1, 4]),)),
        ("b", (np.array([2, 5, 8]),)),
        ("a", (np.array([7, 10]),)),
        ("b", (np.array([11, 0, 0]),)),
        ("c", (np.array([3, 6, 9, 12, 0]),)),
    ]
    self.assertEqual(len(batched), len(expected))
    for (bk, bv), (ek, ev) in zip(batched, expected):
      self.assertEqual(bk, ek)
      jax.test_util.check_eq(bv, ev)

  def test_batch_into_prototype(self):
    # Silence expected warning for this test.
    logging.set_verbosity(logging.ERROR)
    values = [{"v1": np.arange(i), "v2": np.arange(10 - i)} for i in range(10)]
    batched = list(
        data_loading.batch_and_pad_to_prototype(
            values, (1, 2), {
                "v1": jax.ShapeDtypeStruct((8,), np.int32),
                "v2": jax.ShapeDtypeStruct((8,), np.int32)
            },
            remainder_behavior=data_loading.BatchRemainderBehavior.PAD_ZERO,
            drop_too_large=True))
    expected = [
        {
            "v1":
                np.array([[
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 2, 0, 0, 0, 0, 0],
                ]]),
            "v2":
                np.array([[
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [0, 1, 2, 3, 4, 5, 6, 0],
                ]]),
        },
        {
            "v1":
                np.array([[
                    [0, 1, 2, 3, 0, 0, 0, 0],
                    [0, 1, 2, 3, 4, 0, 0, 0],
                ]]),
            "v2":
                np.array([[
                    [0, 1, 2, 3, 4, 5, 0, 0],
                    [0, 1, 2, 3, 4, 0, 0, 0],
                ]]),
        },
        {
            "v1":
                np.array([[
                    [0, 1, 2, 3, 4, 5, 0, 0],
                    [0, 1, 2, 3, 4, 5, 6, 0],
                ]]),
            "v2":
                np.array([[
                    [0, 1, 2, 3, 0, 0, 0, 0],
                    [0, 1, 2, 0, 0, 0, 0, 0],
                ]]),
        },
        {
            "v1":
                np.array([[
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]]),
            "v2":
                np.array([[
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]]),
        },
    ]
    jax.test_util.check_eq(batched, expected)

  def test_prefetch_to_end(self):
    with data_loading.ThreadedPrefetcher(range(100), 10) as prefetched:
      values = list(prefetched)
    self.assertEqual(values, list(range(100)))

  def test_prefetch_with_error(self):

    def raise_after(n):
      yield from range(n)
      raise RuntimeError("error in generation")

    with data_loading.ThreadedPrefetcher(raise_after(100), 10) as prefetched:
      with self.assertRaisesRegex(RuntimeError, "error in generation"):
        for _ in prefetched:
          pass

  def test_prefetch_slow_interrupted(self):
    lock = threading.Lock()
    kept_going = False

    def slow_prefetch():
      nonlocal kept_going
      for i in range(5):
        time.sleep(0.1)
        yield i
      time.sleep(1)
      yield "prefetched, but not consumed"
      with lock:
        kept_going = True
      yield "should not be prefetched"

    with data_loading.ThreadedPrefetcher(slow_prefetch(), 10) as prefetched:
      # Wait for the first five values to be ready.
      first_five = list(itertools.islice(prefetched, 5))
      # The thread starts prefetching the sixth element...
      time.sleep(0.1)
      # ... and we exit the context manager before it finishes.

    time.sleep(2)
    # Main thread should have waited for the first five elements.
    self.assertEqual(first_five, list(range(5)))
    with lock:
      # Worker should NOT have prefetched the seventh element after we exited
      # the context manager.
      self.assertFalse(kept_going)

    # Main thread is not allowed to keep using the iterator at this point.
    with self.assertRaisesRegex(
        RuntimeError,
        "Iteration is only allowed inside the prefetching context manager!"):
      next(prefetched)


if __name__ == "__main__":
  absltest.main()
