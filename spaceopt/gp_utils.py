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

"""Preparing the GP utility functions for evaluting the search space scores."""
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from  sklearn.gaussian_process import kernels
from tensorflow_probability.substrates import jax as tfp

PRECISION = jax.lax.Precision.DEFAULT


def sqdist(x1, x2=None, precision=PRECISION):
  """Computes the matrix of squared distances between two tensors.

  Args:
    x1: (n, ...) shaped tensor.
    x2: (m, ...) shaped tensor where x1.shape[1:] and x2.shape[1:] are assumed
      to be compatible.
    precision: argument for jax functions controlling the tradeoff between
      accuracy and speed.

  Returns:
    out: (n, m) shaped array of squared distances between x1 and x2.
  """
  if x2 is None:
    x2 = x1
  sum_axis = list(range(1, x1.ndim))
  out = jnp.float32(-2) * jnp.tensordot(
      x1, x2, (sum_axis, sum_axis), precision=precision)
  out += jnp.sum(x1**2, axis=sum_axis)[:, jnp.newaxis]
  out += jnp.sum(x2**2, axis=sum_axis)[jnp.newaxis]
  return out


def matern_5_2(x, y, length_scale):
  dists = jnp.sqrt(sqdist(x / length_scale, y / length_scale))
  k = dists * jnp.sqrt(5.)
  k = (1. + k + k ** 2 / 3.0) * jnp.exp(-k)
  return k

PARAMS_BOUNDS = {
    'amplitude': (0.05, 2.),
    'noise': (0.0005, .1),
    'lengthscale': (0.005, 20.)
}

N_RESTARTS_OPTIMIZER = 10


def cov_function_sklearn(params, nu = 5/2):
  """Generates a default covariance function.

  Args:
    params: A dictionary with GP hyperparameters.
    nu: Degree of the matern kernel.

  Returns:
    cov_fun: an ARD Matern covariance function with diagonal noise for
    numerical stability.

  """
  amplitude = params['amplitude']
  noise = params['noise']
  lengthscale = params['lengthscale'].flatten()

  amplitude_bounds = PARAMS_BOUNDS['amplitude']
  lengthscale_bounds = PARAMS_BOUNDS['lengthscale']
  noise_bounds = PARAMS_BOUNDS['noise']

  cov_fun = kernels.ConstantKernel(
      amplitude, constant_value_bounds=amplitude_bounds) * kernels.Matern(
          lengthscale, nu=nu,
          length_scale_bounds=lengthscale_bounds) + kernels.WhiteKernel(
              noise, noise_level_bounds=noise_bounds)
  return cov_fun


def cov_function_jax(params, x, y=None, add_noise=False):
  """Evaluates the default matern 5/2 covariance function."""
  amplitude = params['amplitude']
  noise = params['noise']
  lengthscale = params['lengthscale'].flatten()

  if y is None:
    y = x
    add_noise = True
  cov = amplitude * matern_5_2(x, y, lengthscale)
  if add_noise:
    cov += np.eye(cov.shape[0]) * noise**2
  return cov


def extract_params_from_sklearn_gp(gaussian_process):
  """Extracts parameter values from the fitted sklearn gp object.

  Following https://arxiv.org/pdf/1206.2944.pdf we assume an ARD Matern 5/2
  kernel with observation noise. The input to this function is a fitted sklearn
  GP object and the output is a dictionary including the values of learned
  hyperparameters and GP statistics.

  Args:
    gaussian_process: GP object from sklearn implementation.

  Returns:
   Dictionary of learned GP hyperparameters and statistics from the sklearn GP
   implementation.
  """
  kernel = gaussian_process.kernel_
  assert isinstance(kernel, sklearn.gaussian_process.kernels.Sum)
  matern_kernel = kernel.k1
  noise_kernel = kernel.k2
  assert isinstance(matern_kernel, sklearn.gaussian_process.kernels.Product)
  assert isinstance(noise_kernel, sklearn.gaussian_process.kernels.WhiteKernel)
  params = {
      'noise': noise_kernel.noise_level,
      'lengthscale': matern_kernel.k2.length_scale,
      'amplitude': matern_kernel.k1.constant_value,
      'l_': gaussian_process.L_,
      # pylint: disable=protected-access
      'y_train_std_': gaussian_process._y_train_std,
      'y_train_mean_': gaussian_process._y_train_mean,
      # pylint: enable=protected-access
      'alpha_': gaussian_process.alpha_
  }
  return params


class GPUtils:
  """Class for GP utilities."""

  def __init__(self,
               cov_fun = None,
               gp_noise_eps = 1e-5):
    """Initialize the GP class."""
    self.cov_fun = cov_fun
    self.gp_noise_eps = gp_noise_eps

  def fit_gp(self,
             x_obs,
             y_obs,
             params,
             steps = 1000):
    """Fit a GP to the observed data and return the optimized params.

    Args:
      x_obs: (n, d) shaped array of n observed x-locations in dimension d.
      y_obs: (n, 1) shaped array of objective values at x_obs.
      params: A dictionary of model hyperparameters.
      steps: Number of optimization steps.
      Note that this argument is ignored for sklearn GP, however might be
      included for other GP backends.
    Returns:
      Dictionary of learned parameters from the sklearn GP implementation.
    """
    del steps
    if self.cov_fun is None:
      self.cov_fun = cov_function_sklearn(params)

    gaussian_process = GaussianProcessRegressor(
        kernel=self.cov_fun,
        alpha=self.gp_noise_eps,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        optimizer='fmin_l_bfgs_b')
    gaussian_process.fit(np.array(x_obs), np.array(y_obs))
    self.gaussian_process = gaussian_process
    params = extract_params_from_sklearn_gp(gaussian_process)
    return params

  def posterior_mean_cov(self, params, x_obs,
                         y_obs, x_test):
    """Evaluate the posterior mean and cov of the test x-locations.

    Args:
      params: Dictionary of learned parameters from the sklearn GP
        implementation.
      x_obs: (n, d) shaped array of n observed x-locations in dimension d.
      y_obs: (n, 1) shaped array of objective values at x_obs.
      Note that this argument is ignored for sklearn GP since we alternatively
      use the already calculated statistics from sklearn GP object, however
      might be included for other GP backends.
      x_test: (m, d) shaped array of m test x-locations in dimension d.

    Returns:
      mu: (m, 1) shaped array of mean at x_test.
      cov: (m, m) shaped array of covariance at x_test.
    """
    del y_obs

    l_ = params['l_']
    y_train_std_ = params['y_train_std_']
    y_train_mean_ = params['y_train_mean_']
    alpha_ = params['alpha_']
    cross_cov = cov_function_jax(params, x_test, x_obs)
    mu = cross_cov @ alpha_
    mu = y_train_std_ * mu + y_train_mean_
    v = jax.scipy.linalg.solve_triangular(l_, cross_cov.T, lower=True)
    other_cov = cov_function_jax(params, x_test)
    other_cov += jnp.eye(other_cov.shape[0]) * self.gp_noise_eps
    cov = (other_cov - jnp.dot(v.T, v))
    cov = jnp.outer(cov, y_train_std_ ** 2).reshape(*cov.shape, -1)
    if cov.shape[2] == 1:
      cov = jnp.squeeze(cov, axis=2)

    return mu, cov

  def draw_gp_samples(self,
                      key,
                      mu,
                      cov,
                      num_samples = 1,
                      method = 'cholesky',
                      tol = 1e-4):
    """Draw multivariate-normal samples given mu and cov.

    Args:
     key: a jax random.PRNGKey.
     mu: (m, 1) shaped array of mean values.
     cov: (m, m) shaped array of covariance values.
     num_samples: number of samples.
     method: method of sampling from  'own', 'cholesky', 'svd' and 'tfp'.
     tol: additional tolerance for numerical stability issue.

    Returns:
     samples: (num_samples, m) shaped array of drawn samples.
    """

    if (method == 'cholesky') or (method == 'svd'):
      samples = jax.random.multivariate_normal(
          key, mu.T, cov, shape=(num_samples,), method=method)
    elif method == 'own':
      y_rand = jax.random.normal(key, (num_samples, cov.shape[0]))
      chol = jax.scipy.linalg.cholesky(
          cov + jnp.eye(cov.shape[0]) * tol, lower=True)
      samples = jnp.dot(y_rand, chol) + mu.T
    elif method == 'tfp':
      tfd = tfp.distributions
      mvn = tfd.MultivariateNormalFullCovariance(
          loc=mu.flatten(), covariance_matrix=cov)
      samples = mvn.sample(num_samples, key)
    else:
      raise ValueError('Accepted methods include own, cholesky, svd and tfp.')
    return samples
