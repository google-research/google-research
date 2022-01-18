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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from simulation_research.traffic import memoize_util


class MemoizeUtilTest(absltest.TestCase):

  def test_memoize_class_function_one_run(self):

    # For this class, each function within a certain instance can only be run
    # once.
    class TestMemoizeClassFunctionOneRun(object):

      function7_counter = 0

      @memoize_util.MemoizeClassFunctionOneRun
      def function1(self, x):
        y = x + 3
        return y

      @memoize_util.MemoizeClassFunctionOneRun
      def function2(self, x):
        y = x + 5
        return y

      @memoize_util.MemoizeClassFunctionOneRun
      def function3(self, x):
        return x, x

      @memoize_util.MemoizeClassFunctionOneRun
      def function4(self):
        x = 1
        del x

      @memoize_util.MemoizeClassFunctionOneRun
      def function5(self, x, y):
        return x, y

      @memoize_util.MemoizeCacheInputOutput
      def function7(self, x, y):
        self.function7_counter += 1
        return x, y, [1, 2, None, 4]

    # Case 1: Different instances and the same functions have different caches.
    test_class_1 = TestMemoizeClassFunctionOneRun()
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(111)
    self.assertEqual(y, 6)
    test_class_2 = TestMemoizeClassFunctionOneRun()
    y = test_class_2.function1(4)
    self.assertEqual(y, 7)
    y = test_class_2.function1(123)
    self.assertEqual(y, 7)

    # Case 2: Different instances and the same functions have different caches.
    #     Note the order of initlization does not matter.
    test_class_1 = TestMemoizeClassFunctionOneRun()
    test_class_2 = TestMemoizeClassFunctionOneRun()
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(11)
    self.assertEqual(y, 6)
    y = test_class_2.function1(4)
    self.assertEqual(y, 7)
    y = test_class_2.function1(12)
    self.assertEqual(y, 7)

    # Case 3: Different instances and different functions have different caches.
    test_class_1 = TestMemoizeClassFunctionOneRun()
    test_class_2 = TestMemoizeClassFunctionOneRun()
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(111)
    self.assertEqual(y, 6)
    y = test_class_2.function2(4)
    self.assertEqual(y, 9)
    y = test_class_2.function2(123)
    self.assertEqual(y, 9)

    # Case 4: For the same instance, different functions have different caches.
    test_class_1 = TestMemoizeClassFunctionOneRun()
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(111)
    self.assertEqual(y, 6)
    y = test_class_1.function2(4)
    self.assertEqual(y, 9)
    y = test_class_1.function2(123)
    self.assertEqual(y, 9)

    # Case 5: Special cases with unhashable input and unhashable output.
    test_class_1 = TestMemoizeClassFunctionOneRun()
    test_class_2 = TestMemoizeClassFunctionOneRun()
    y = test_class_1.function3([1, 2, 3])
    self.assertListEqual(y[0], [1, 2, 3])
    self.assertListEqual(y[1], [1, 2, 3])
    y = test_class_1.function3([4, 5, None, 7])
    self.assertListEqual(y[0], [1, 2, 3])
    self.assertListEqual(y[1], [1, 2, 3])
    y = test_class_2.function3([11, 2, 3])
    self.assertListEqual(y[0], [11, 2, 3])
    self.assertListEqual(y[1], [11, 2, 3])
    y = test_class_2.function3([44, 56, None, 7])
    self.assertListEqual(y[0], [11, 2, 3])
    self.assertListEqual(y[1], [11, 2, 3])

    # Case 5: No input, no return.
    test_class_1 = TestMemoizeClassFunctionOneRun()
    y = test_class_1.function4()
    self.assertEqual(y, None)
    y = test_class_1.function4()
    self.assertEqual(y, None)

    # Case 6: Multiple arguments.
    test_class_1 = TestMemoizeClassFunctionOneRun()
    y = test_class_1.function5(1, 2)
    self.assertEqual(y[0], 1)
    self.assertEqual(y[1], 2)
    y = test_class_1.function5(4, 5)
    self.assertEqual(y[0], 1)
    self.assertEqual(y[1], 2)
    test_class_2 = TestMemoizeClassFunctionOneRun()
    y = test_class_2.function5(1, [2, 3, 4])
    self.assertEqual(y[0], 1)
    self.assertListEqual(y[1], [2, 3, 4])
    y = test_class_2.function5(4, [5, 6, 7])
    self.assertEqual(y[0], 1)
    self.assertListEqual(y[1], [2, 3, 4])

    # Case 7. If the arguments are unhashable, the function should only be run
    # once during the whole process. Since the input is unhashable, the function
    # needs to be executed with the same argument from the first time.
    test_class = TestMemoizeClassFunctionOneRun()
    y = test_class.function7([1, 2, 3], [4, 5, None])
    self.assertEqual(test_class.function7_counter, 1)
    y = test_class.function7([1, 2, 3], [4, 5, None])
    self.assertEqual(test_class.function7_counter, 2)
    y = test_class.function7([1, 2, 3], [4, 5, None])
    self.assertEqual(test_class.function7_counter, 3)
    y = test_class.function7([11, 22, 3], [4, None])
    self.assertEqual(test_class.function7_counter, 4)
    y = test_class.function7([3, 3, 2], [7])
    self.assertEqual(test_class.function7_counter, 5)

  def test_memoize_cache_input_output(self):

    # This memoize caches the input-output.
    class TestMemoizeCacheInputOutput(object):

      function5_counter = 0

      @memoize_util.MemoizeCacheInputOutput
      def function1(self, x):
        y = x + 3
        return y

      @memoize_util.MemoizeCacheInputOutput
      def function2(self, x):
        y = x + 5
        return y

      @memoize_util.MemoizeCacheInputOutput
      def function3(self, x):
        return x, x, [1, 2, None, 4]

      @memoize_util.MemoizeCacheInputOutput
      def function4(self, x, y):
        return x, y, [1, 2, None, 4]

      @memoize_util.MemoizeCacheInputOutput
      def function5(self, x, y):
        self.function5_counter += 1
        return x, y, [1, 2, None, 4]

    # Case 1: Different instances and the same functions have different caches.
    test_class_1 = TestMemoizeCacheInputOutput()
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(4)
    self.assertEqual(y, 7)
    y = test_class_1.function1(4)
    self.assertEqual(y, 7)
    test_class_2 = TestMemoizeCacheInputOutput()
    y = test_class_2.function1(4)
    self.assertEqual(y, 7)
    y = test_class_2.function1(8)
    self.assertEqual(y, 11)

    # Case 2: Same instance and the different functions have different caches.
    test_class_1 = TestMemoizeCacheInputOutput()
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(3)
    self.assertEqual(y, 6)
    y = test_class_1.function1(4)
    self.assertEqual(y, 7)
    y = test_class_1.function1(4)
    self.assertEqual(y, 7)
    y = test_class_1.function2(5)
    self.assertEqual(y, 10)
    y = test_class_1.function2(5)
    self.assertEqual(y, 10)

    # Case 3: Special cases with unhashable input.
    test_class_1 = TestMemoizeCacheInputOutput()
    test_class_2 = TestMemoizeCacheInputOutput()
    y = test_class_1.function3([1, 2, 3])
    self.assertListEqual(y[0], [1, 2, 3])
    self.assertListEqual(y[1], [1, 2, 3])
    self.assertListEqual(y[2], [1, 2, None, 4])
    y = test_class_1.function3([4, 5, None, 7])
    self.assertListEqual(y[0], [4, 5, None, 7])
    self.assertListEqual(y[1], [4, 5, None, 7])
    self.assertListEqual(y[2], [1, 2, None, 4])
    y = test_class_2.function3([11, 2, 3])
    self.assertListEqual(y[0], [11, 2, 3])
    self.assertListEqual(y[1], [11, 2, 3])
    self.assertListEqual(y[2], [1, 2, None, 4])
    y = test_class_2.function3([11, 2, 3])
    self.assertListEqual(y[0], [11, 2, 3])
    self.assertListEqual(y[1], [11, 2, 3])
    self.assertListEqual(y[2], [1, 2, None, 4])

    # Case 4. Multiple returns can be cached.
    test_class_3 = TestMemoizeCacheInputOutput()
    y = test_class_3.function3(444)
    self.assertEqual(y[0], 444)
    self.assertEqual(y[1], 444)
    self.assertListEqual(y[2], [1, 2, None, 4])
    y = test_class_3.function3(444)
    self.assertEqual(y[0], 444)
    self.assertEqual(y[1], 444)
    self.assertListEqual(y[2], [1, 2, None, 4])

    # Case 5. If the arguments are unhashable, the function should only be run
    # once. Since the input is unhashable, the function needs to be executed
    # with the same argument from the first time.
    test_class = TestMemoizeCacheInputOutput()
    y = test_class.function5([1, 2, 3], [4, 5, None])
    self.assertEqual(test_class.function5_counter, 1)
    y = test_class.function5([1, 2, 3], [4, 5, None])
    self.assertEqual(test_class.function5_counter, 2)
    y = test_class.function5([1, 2, 3], [4, 5, None])
    self.assertEqual(test_class.function5_counter, 3)
    y = test_class.function5([11, 22, 3], [4, None])
    self.assertEqual(test_class.function5_counter, 4)
    y = test_class.function5([3, 3, 2], [7])
    self.assertEqual(test_class.function5_counter, 5)


if __name__ == '__main__':
  absltest.main()
