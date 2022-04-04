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

"""Tests for gp_utils."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from spaceopt import gp_utils as gp


def obj(x):
  return -x - x*jnp.sin(10.0*x)


class GpUtilsTest(absltest.TestCase):

  def setUp(self):
    super(GpUtilsTest, self).setUp()
    key = jax.random.PRNGKey(0)
    num_obs = 10
    x_obs = jax.random.uniform(
        key, shape=(num_obs,), minval=-4, maxval=4.)[:, None]
    y_obs = obj(x_obs)
    self.x_obs = x_obs
    self.y_obs = y_obs
    self.x_test = jnp.linspace(-4., 4., 100)[:, None]
    dims = x_obs.shape[1]
    self.params_init = {
        "amplitude": jnp.zeros((1, 1)),
        "noise": jnp.zeros((1, 1)) - 5.0,
        "lengthscale": jnp.ones((1, dims))
    }
    self.n_restarts_optimizer = 10

  def test_mu_cov(self):
    gp_util = gp.GPUtils()
    params = gp_util.fit_gp(self.x_obs, self.y_obs, self.params_init)
    mu, cov = gp_util.posterior_mean_cov(params, self.x_obs, self.y_obs,
                                         self.x_test)
    cov_fun = gp.cov_function_sklearn(self.params_init)
    sklearn_gp = GaussianProcessRegressor(
        kernel=cov_fun,
        alpha=1e-5,
        _restarts_optimizer=self.n_restarts_optimizer,
        optimizer="fmin_l_bfgs_b")
    sklearn_gp.fit(self.x_obs, self.y_obs)
    mu_expected, cov_expected = sklearn_gp.predict(self.x_test, return_cov=True)
    self.assertTrue(np.isclose(mu, mu_expected, rtol=1e-2, atol=1e-3).all())
    self.assertTrue(np.isclose(cov, cov_expected, rtol=1e-2, atol=1e-3).all())


if __name__ == "__main__":
  absltest.main()
