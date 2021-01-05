# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# python3
# pylint: disable=invalid-name
import numpy as np
from task_set.tasks import quadratic_helper
import tensorflow.compat.v1 as tf


class QuadraticHelperTest(tf.test.TestCase):

  def test_eigen_spectrum_matrix_distribution(self):
    """Ensures that the sampled matrix has the correct eigen spectrum."""
    dims = 5

    specturm = np.linspace(1.0, 2.0, dims).astype(np.float32)
    A_dist = quadratic_helper.FixedEigenSpectrumMatrixDistribution(specturm)
    s = A_dist.sample()

    with self.test_session() as sess:
      mat = sess.run(s)
      # Ensure the matrix is not simply a diagonal.
      sum_square_not_diag = np.sum(np.square(mat - np.eye(dims) * mat))
      self.assertGreater(sum_square_not_diag, 1e-3)

      eig_value, _ = np.linalg.eig(mat)
      np.testing.assert_allclose(
          np.linspace(1, 2, 5), sorted(eig_value), rtol=1e-5)

  def test_fixed_dim_sample_quadratic(self):
    """Ensures that not passing `seed` produces different losses."""
    dims = 10
    initial_dist = tf.distributions.Normal(
        loc=0., scale=tf.ones([dims], dtype=tf.float32))

    specturm = np.linspace(1.0, 2.0, dims).astype(np.float32)
    A_dist = quadratic_helper.FixedEigenSpectrumMatrixDistribution(specturm)

    B_dist = tf.distributions.Normal(
        loc=0., scale=tf.ones([dims], dtype=tf.float32))
    C_dist = tf.distributions.Exponential(2.)

    lossmod = quadratic_helper.QuadraticBasedTask(
        dims=dims,
        initial_dist=initial_dist,
        A_dist=A_dist,
        B_dist=B_dist,
        C_dist=C_dist)
    s = lossmod.initial_params()
    loss = lossmod.call_split(s, None)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      loss1 = sess.run(loss)
      sess.run(tf.global_variables_initializer())
      loss2 = sess.run(loss)
      self.assertNotEqual(loss1, loss2)

  def test_fixed_seed_dim_sample_quadratic(self):
    """Ensures that passing `seed` produces the same loss."""
    dims = 10
    initial_dist = quadratic_helper.ConstantDistribution(
        np.random.normal(0, 1, [dims]))

    specturm = np.linspace(1.0, 2.0, dims).astype(np.float32)
    A_dist = quadratic_helper.FixedEigenSpectrumMatrixDistribution(specturm)

    B_dist = quadratic_helper.ConstantDistribution(
        tf.ones([dims], dtype=tf.float32))
    C_dist = quadratic_helper.ConstantDistribution(1.0)

    lossmod = quadratic_helper.QuadraticBasedTask(
        dims=dims,
        initial_dist=initial_dist,
        A_dist=A_dist,
        B_dist=B_dist,
        C_dist=C_dist,
        seed=23,
    )
    s = lossmod.initial_params()
    loss = lossmod.call_split(s, None)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      loss1 = sess.run(loss)
      sess.run(tf.global_variables_initializer())
      loss2 = sess.run(loss)
      self.assertNear(loss1, loss2, 1e-9)


if __name__ == "__main__":
  tf.test.main()
