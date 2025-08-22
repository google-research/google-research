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

"""Run all tests in study_recommend."""
import unittest

from study_recommend import datasource_test
from study_recommend import inference_test
from study_recommend import models_test
from study_recommend import training_loop_test
from study_recommend.utils import evaluations_test
from study_recommend.utils import input_pipeline_test
from study_recommend.utils import train_utils_test

test_modules = [
    datasource_test,
    inference_test,
    models_test,
    training_loop_test,
    evaluations_test,
    input_pipeline_test,
    train_utils_test,
]


def assert_len(self, iterable, length):
  self.assertEqual(len(iterable), length)

unittest.TestCase.assertLen = assert_len


def test_suite():
  """Build a test suite with all tests in study_recommend."""
  suite = unittest.TestSuite()
  loader = unittest.TestLoader()
  for test_module in test_modules:
    suite.addTests(loader.loadTestsFromModule(test_module))

  return suite

if __name__ == '__main__':
  runner = unittest.TextTestRunner(verbosity=3)
  runner.run(test_suite())
