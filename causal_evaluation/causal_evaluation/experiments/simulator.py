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

"""Code to generate simulated data."""

import jax
import numpy as np
import scipy
import sklearn.preprocessing


class Simulator:
  """Generates simulated data following a causal generative process."""

  def __init__(self, **kwargs):
    """Initializes simulation.

    Arguments:
      **kwargs: provided kwargs will override default parameters.
    """

    self.param_dict = self.get_default_param_dict()

    if kwargs is not None:
      for key, value in kwargs.items():
        self.param_dict[key] = value

  def get_default_param_dict(self):
    """Returns default parameters for the simulation."""
    param_dict = {
        'num_samples': 10000,
        'k_x': 1,
        'k_y': 1,
        'mu_x_u': np.array([-2, 0]),
        'beta_a': 1,
        'pi_a': np.array([0.5, 0.5]),
        'mu_y_a': np.array([0, 0]),
        'mu_y_x_base': 0.5,
        'a_to_y': False,
        'sd_x': 1,
        'p_u': [0.5, 0.5],
    }
    return param_dict

  def get_samples(self, p_u=None, seed=42):
    """Generates samples from the simulation.

    Arguments:
      p_u: array that specifies the mixture proportions over latent categories u
      seed: a random seed

    Returns:
      a dict containing generated data
    """

    rng = jax.random.PRNGKey(seed)
    _, k0, _ = jax.random.split(rng, 3)

    ## Generate u
    if p_u is None:
      p_u = self.param_dict['p_u']

    u = np.random.binomial(1, p_u[1], size=self.param_dict['num_samples'])
    u_one_hot = sklearn.preprocessing.OneHotEncoder(
        sparse_output=False
    ).fit_transform(u.reshape(-1, 1))

    ## Generate x
    x = jax.random.multivariate_normal(
        key=k0,
        mean=(u_one_hot @ self.param_dict['mu_x_u']).reshape(-1, 1),
        cov=self.param_dict['sd_x'] * np.eye(self.param_dict['k_x']),
    )
    x = np.array(x).astype(np.float64)

    ## Generate a
    p_a = (
        self.param_dict['beta_a'] * u_one_hot
        + (1 - self.param_dict['beta_a']) * self.param_dict['pi_a']
    )
    a = np.random.binomial(1, p_a[:, 1], size=self.param_dict['num_samples'])
    a_one_hot = sklearn.preprocessing.OneHotEncoder(
        sparse_output=False
    ).fit_transform(a.reshape(-1, 1))

    ## Generate y
    if self.param_dict['a_to_y']:
      mu_y_x = np.array([[
          self.param_dict['mu_y_x_base'],
          -2 * self.param_dict['mu_y_x_base'],
      ]])
    else:
      mu_y_x = np.array(
          [[self.param_dict['mu_y_x_base'], self.param_dict['mu_y_x_base']]]
      )

    y_logits = x.dot(mu_y_x)[
        np.arange(self.param_dict['num_samples']), np.squeeze(a)
    ].reshape(-1, 1) + (a_one_hot @ self.param_dict['mu_y_a']).reshape(-1, 1)
    p_y = scipy.special.expit(y_logits)

    y = np.squeeze(np.random.binomial(n=1, p=p_y))
    y_one_hot = sklearn.preprocessing.OneHotEncoder(
        sparse_output=False
    ).fit_transform(y.reshape(-1, 1))

    selected, p_selected = self.selection_function(u=u, a=a, x=x, y=y)

    return {
        'u': u,
        'a': a,
        'x': x,
        'y': y,
        'y_logits': y_logits,
        'p_y': p_y,
        'y_one_hot': y_one_hot,
        'selected': selected,
        'p_selected': p_selected,
    }

  def update_param_dict(self, **kwargs):
    if kwargs is not None:
      for key, value in kwargs.items():
        self.param_dict[key] = value

  def selection_function(self, **kwargs):
    """Default selection function."""
    u = kwargs['u']
    return np.ones((u.shape[0], 1)), np.ones((u.shape[0], 1))


class SimulatorAnticausal(Simulator):
  """Generates simulated data following an anticausal generative process."""

  def get_default_param_dict(self):
    param_dict = {
        'num_samples': 10000,
        'k_x': 1,
        'k_y': 2,
        'mu_x_ay': np.array([[-1, 1], [-1, 1]]),  # k_a x k_y
        'mu_y_u': np.array([[0.5, 0.5], [0.9, 0.1]]),
        'sd_x': 1,
        'p_u': [0.5, 0.5],
        'beta_a': 1,
        'pi_a': np.array([0.5, 0.5]),
    }
    return param_dict

  def get_samples(self, p_u=None, seed=42):
    """Generates samples from the simulation.

    Arguments:
      p_u: array that specifies the mixture proportions over latent categories u
      seed: a random seed

    Returns:
      a dict containing generated data
    """

    rng = jax.random.PRNGKey(seed)
    _, _, k1 = jax.random.split(rng, 3)

    ## Generate u
    if p_u is None:
      p_u = self.param_dict['p_u']

    u = np.random.binomial(1, p_u[1], size=self.param_dict['num_samples'])
    u_one_hot = sklearn.preprocessing.OneHotEncoder(
        sparse_output=False
    ).fit_transform(u.reshape(-1, 1))

    ## Generate a
    p_a = (
        self.param_dict['beta_a'] * u_one_hot
        + (1 - self.param_dict['beta_a']) * self.param_dict['pi_a']
    )
    a = np.random.binomial(1, p_a[:, 1], size=self.param_dict['num_samples'])
    a_one_hot = sklearn.preprocessing.OneHotEncoder(
        sparse_output=False
    ).fit_transform(a.reshape(-1, 1))

    ## Generate y
    p_y = u_one_hot @ self.param_dict['mu_y_u']
    p_y_1 = p_y[:, -1]
    y = (np.log(p_y) + np.random.gumbel(size=p_y.shape)).argmax(1)
    y_one_hot = sklearn.preprocessing.OneHotEncoder(
        sparse_output=False
    ).fit_transform(y.reshape(-1, 1))

    ## Generate x
    x = jax.random.multivariate_normal(
        key=k1,
        mean=self.param_dict['mu_x_ay'][a, y].reshape(
            -1, self.param_dict['k_x']
        ),
        cov=self.param_dict['sd_x'] * np.eye(self.param_dict['k_x']),
    )
    x = np.array(x).astype(np.float64)

    selected, p_selected = self.selection_function(u=u, a=a, x=x, y=y)

    return {
        'u': u,
        'u_one_hot': u_one_hot,
        'a': a,
        'a_one_hot': a_one_hot,
        'x': x,
        'y': y,
        'p_y': p_y_1,
        'y_one_hot': y_one_hot,
        'selected': selected,
        'p_selected': p_selected,
    }
