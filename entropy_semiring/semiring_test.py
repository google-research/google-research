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

"""Tests for semiring."""

from absl.testing import parameterized
from lingvo import compat as tf
import semiring


class SemiringTest(parameterized.TestCase, tf.test.TestCase):

  def _TestAdditiveMonoid(self, r, elem_1, elem_2, elem_3):
    """Checks that addition is a commutative monoid."""
    self.assertAllClose(r.add(elem_1, elem_2), r.add(elem_2, elem_1))
    self.assertAllClose(r.add(elem_1, r.additive_identity()), elem_1)
    self.assertAllClose(r.add(r.additive_identity(), elem_1), elem_1)
    self.assertAllClose(
        r.add(r.add(elem_1, elem_2), elem_3),
        r.add(elem_1, r.add(elem_2, elem_3)))

  def _TestMultiplicativeMonoid(self, r, elem_1, elem_2, elem_3):
    """Checks that multiplication is a monoid."""
    self.assertAllClose(r.multiply(elem_1, r.multiplicative_identity()), elem_1)
    self.assertAllClose(r.multiply(r.multiplicative_identity(), elem_1), elem_1)
    self.assertAllClose(
        r.multiply(r.multiply(elem_1, elem_2), elem_3),
        r.multiply(elem_1, r.multiply(elem_2, elem_3)))

  def _TestAdditionList(self, r, elem_list):
    """Compare result of r.add_list with r.add."""
    manual_sum = elem_list[0]
    for elem in elem_list[1:]:
      manual_sum = r.add(manual_sum, elem)
    self.assertAllClose(manual_sum, r.add_list(elem_list))

  def _TestMultiplicationList(self, r, elem_list):
    """Compare result of r.multiply_list with r.multiply."""
    manual_prod = elem_list[0]
    for elem in elem_list[1:]:
      manual_prod = r.multiply(manual_prod, elem)
    self.assertAllClose(manual_prod, r.multiply_list(elem_list))

  def _TestDistributiveProperty(self, r, elem_1, elem_2, elem_3):
    """Checks that multiplication distributes over addition."""
    self.assertAllClose(
        r.multiply(elem_1, r.add(elem_2, elem_3)),
        r.add(r.multiply(elem_1, elem_2), r.multiply(elem_1, elem_3)))
    self.assertAllClose(
        r.multiply(r.add(elem_2, elem_3), elem_1),
        r.add(r.multiply(elem_2, elem_1), r.multiply(elem_3, elem_1)))

  def _TestAnnihilation(self, r, elem_1):
    """Checks that additive identity is a multiplicative annihilator."""
    self.assertAllClose(
        r.multiply(r.additive_identity(), elem_1), r.additive_identity())
    self.assertAllClose(
        r.multiply(elem_1, r.additive_identity()), r.additive_identity())

  @parameterized.named_parameters(
      (
          'Log Semiring',
          semiring.LogSemiring(),
          (tf.constant([-2.0]),),
          (tf.constant([-3.0]),),
          (tf.constant([-4.0]),),
      ),
      (
          'Log Entropy Semiring',
          semiring.LogEntropySemiring(),
          (tf.constant([-2.0]), tf.constant([-2.5])),
          (tf.constant([-3.0]), tf.constant([-3.5])),
          (tf.constant([-4.0]), tf.constant([-4.5])),
      ),
      (
          'Log Reverse-KL Semiring',
          semiring.LogReverseKLSemiring(),
          (tf.constant([-2.0]), tf.constant([-2.5]), tf.constant(
              [-2.6]), tf.constant([-2.7])),
          (tf.constant([-3.0]), tf.constant([-3.5]), tf.constant(
              [-3.6]), tf.constant([-3.7])),
          (tf.constant([-4.0]), tf.constant([-4.5]), tf.constant(
              [-4.6]), tf.constant([-4.7])),
      ),
  )
  def testSemiring(self, r, elem_1, elem_2, elem_3):
    """Tests if r is a semiring."""
    self._TestAdditiveMonoid(r, elem_1, elem_2, elem_3)
    self._TestMultiplicativeMonoid(r, elem_1, elem_2, elem_3)
    self._TestAdditionList(r, [elem_1, elem_2, elem_3])
    self._TestMultiplicationList(r, [elem_1, elem_2, elem_3])
    self._TestDistributiveProperty(r, elem_1, elem_2, elem_3)
    self._TestAnnihilation(r, elem_1)


if __name__ == '__main__':
  tf.test.main()
