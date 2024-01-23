# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Example program tests."""

from collections.abc import Sequence
from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized

import jax.config
import numpy as np

from jax_effects._src import core
from jax_effects.examples import examples_testdata
from jax_effects.examples import interleaved_effects
from jax_effects.examples import linear_regression
from jax_effects.examples import parameterized_handler
from jax_effects.examples import parameterized_returning_handler
from jax_effects.examples import qlearning
from jax_effects.examples import returning_handler
from jax_effects.examples import simple_loss
from jax_effects.examples import while_loop

Program = Callable[[], Any]

TESTCASES = [
    dict(
        testcase_name='interleaved_effects',
        program=interleaved_effects.interleaved_effects_example,
        expected_result=(5.0, 49),
    ),
    dict(
        testcase_name='linear_regression',
        program=linear_regression.linear_regression_example,
        expected_result=(
            7.018235683441162,
            (0.6667861938476562, -0.9708063006401062),
        ),
    ),
    dict(
        testcase_name='parameterized_handler',
        program=parameterized_handler.state_example,
        expected_result=20,
    ),
    dict(
        testcase_name='parameterized_returning_handler',
        program=parameterized_returning_handler.state_example,
        expected_result=43,
    ),
    dict(
        testcase_name='qlearning',
        program=qlearning.q_learning_example,
        expected_result=(
            examples_testdata.qlearning_expected_qtable,
            np.asarray([5, 0]),
        ),
    ),
    dict(
        testcase_name='returning_handler',
        program=returning_handler.returning_handler_example,
        expected_result=60,
    ),
    dict(
        testcase_name='simple_loss',
        program=simple_loss.simple_loss_example,
        expected_result=8,
    ),
    dict(
        testcase_name='while_loop',
        program=while_loop.simple_while_loop,
        expected_result=(10, 45),
    ),
]


class ExamplesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    core.effect_primitives.clear()

  def assertAllClose(self, x, y):
    jax.tree_map(np.testing.assert_allclose, x, y)

  @parameterized.named_parameters(*TESTCASES)
  def test_example(
      self,
      program,
      expected_result,
      arguments = (),
  ):
    actual_result = program(*arguments)
    self.assertAllClose(actual_result, expected_result)


if __name__ == '__main__':
  absltest.main()
