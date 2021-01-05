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

"""Class and utils for linear dynamical systems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pylds.models import DefaultLDS
from scipy.stats import ortho_group
import seaborn as sns
from statsmodels import api as sm_api
from statsmodels.tools import sm_exceptions


class LinearDynamicalSystem(object):
  """Class to represent a linear dynamical system."""

  def __init__(self, transition_matrix, input_matrix, output_matrix):
    """Initializes a linear dynamical system object.

    Args:
      transition_matrix: The transition matrix of shape [hidden_state_dim,
        hidden_state_dim].
      input_matrix: The input matrix of shape [hidden_state_dim, input_dim].
      output_matrix: The measurement matrix of shape [output_dim,
        hidden_state_dim].
    """
    self.hidden_state_dim = transition_matrix.shape[0]
    self.input_dim = input_matrix.shape[1]
    self.output_dim = output_matrix.shape[0]
    if transition_matrix.shape != (self.hidden_state_dim,
                                   self.hidden_state_dim):
      raise ValueError('Dimension mismatch.')
    if input_matrix.shape != (self.hidden_state_dim, self.input_dim):
      raise ValueError('Dimension mismatch.')
    if output_matrix.shape != (self.output_dim, self.hidden_state_dim):
      raise ValueError('Dimension mismatch.')
    self.transition_matrix = transition_matrix
    self.input_matrix = input_matrix
    self.output_matrix = output_matrix

  def get_spectrum(self):
    eigs = np.linalg.eig(self.transition_matrix)[0]
    return eigs[np.argsort(eigs.real)[::-1]]

  def get_expected_arparams(self):
    return -np.poly(self.get_spectrum())[1:]


class LinearDynamicalSystemSequence(object):
  """Wrapper around input seq, hidden state seq, and output seq from LDS."""

  def __init__(self, input_seq, hidden_state_seq, output_seq):
    self.seq_len = np.shape(input_seq)[0]
    if self.seq_len != np.shape(hidden_state_seq)[0]:
      raise ValueError('Sequence length mismatch.')
    if self.seq_len != np.shape(output_seq)[0]:
      raise ValueError('Sequence length mismatch.')
    self.inputs = input_seq
    self.hidden_states = hidden_state_seq
    self.outputs = output_seq
    self.input_dim = np.shape(input_seq)[1]
    self.output_dim = np.shape(output_seq)[1]

  def plot(self, input_colors=None, output_colors=None, output_only=False):
    """Plots the sequence in seaborn lineplot.

    Args:
      input_colors: A list of strings, optional.
      output_colors: A list of strings, optional.
      output_only: Whether to only plot output.
    """
    plt.figure(figsize=(10, 6))
    if not output_only:
      for i in xrange(self.input_dim):
        c = input_colors[i] if input_colors else None
        sns.lineplot(
            np.arange(self.seq_len),
            self.inputs[:, i],
            label='input_' + str(i),
            alpha=0.5,
            color=c)
    for i in xrange(self.output_dim):
      c = output_colors[i] if output_colors else None
      sns.lineplot(
          np.arange(self.seq_len),
          self.outputs[:, i],
          label='output_' + str(i),
          color=c)
    plt.title('Generated Sequence')

  def get_pred_error(self, predictions, num_warm_start_steps, relative=True):
    if num_warm_start_steps != self.seq_len - np.shape(predictions)[0]:
      raise ValueError('Sequence length mismatch.')
    err = np.linalg.norm(self.outputs[num_warm_start_steps:] - predictions)
    if relative:
      return err / np.linalg.norm(self.outputs[num_warm_start_steps:])
    else:
      return err

  def plot_pred(self,
                predictions,
                num_warm_start_steps,
                plot_error=False,
                seq_colors=None,
                pred_colors=None):
    """Plots predictions vs ground truth.

    Args:
      predictions: A numpy array of length self.seq_len - num_warm_start_steps.
      num_warm_start_steps: Number of steps before generating predictions.
      plot_error: Whether to plot errors.
      seq_colors: A list of strings, optional.
      pred_colors: A list of strings, optional.
    """
    self.plot(output_colors=seq_colors, output_only=True)
    for i in xrange(self.output_dim):
      c = pred_colors[i] if pred_colors else None
      sns.lineplot(
          np.arange(num_warm_start_steps, self.seq_len),
          predictions[:, i],
          label='pred_' + str(i),
          color=c)
      if plot_error:
        sns.lineplot(
            np.arange(num_warm_start_steps, self.seq_len),
            predictions[:, i] - self.outputs[num_warm_start_steps:, i],
            label='error_' + str(i),
            color='black')
    plt.title('Predictions, relative error: ' +
              str(self.get_pred_error(predictions, num_warm_start_steps)))


class SequenceGenerator(object):
  """Class for generating sequences according to linear dynamical systems."""

  def __init__(self,
               input_mean,
               input_stddev,
               output_noise_stddev,
               init_state_mean=0.0,
               init_state_stddev=0.0):
    """Initializes SequenceGenerator.

    Args:
      input_mean: The mean of the input distribution. If the input is Gaussian
        noise then the input_mean is 0.
      input_stddev: The stddev of the input distribution.
      output_noise_stddev: The stddev of the output noise distribution.
      init_state_mean: The mean of the initial hidden state distribution.
      init_state_stddev: The stddev of the initial hidden state distribution.
    """
    self.input_mean = input_mean
    self.input_stddev = input_stddev
    self.output_noise_stddev = output_noise_stddev
    self.init_state_mean = init_state_mean
    self.init_state_stddev = init_state_stddev

  def _random_normal(self, mean, stddev, dim):
    return mean + stddev * np.random.randn(np.prod(dim)).reshape(dim)

  def generate_seq(self, system, seq_len):
    """Generate seq with random initial state, inputs, and output noise.

    Args:
      system: A LinearDynamicalSystem instance.
      seq_len: The desired length of the sequence.

    Returns:
      A LinearDynamicalSystemSequence object with:
      - outputs: A numpy array of shape [seq_len, output_dim].
      - hidden_states: A numpy array of shape [seq_len, hidden_state_dim].
      - inputs: A numpy array of shape [seq_len, input_dim].
    """
    inputs = self._random_normal(self.input_mean, self.input_stddev,
                                 [seq_len, system.input_dim])
    outputs = np.zeros([seq_len, system.output_dim])
    output_noises = self._random_normal(0.0, self.output_noise_stddev,
                                        [seq_len, system.output_dim])
    hidden_states = np.zeros([seq_len, system.hidden_state_dim])
    # Initial state.
    hidden_states[0, :] = self._random_normal(self.init_state_mean,
                                              self.init_state_stddev,
                                              system.hidden_state_dim)
    for j in xrange(1, seq_len):
      hidden_states[j, :] = (
          np.matmul(system.transition_matrix, hidden_states[j - 1, :]) +
          np.matmul(system.input_matrix, inputs[j, :]))
    for j in xrange(seq_len):
      outputs[j, :] = np.matmul(system.output_matrix,
                                hidden_states[j, :]) + output_noises[j, :]
    return LinearDynamicalSystemSequence(inputs, hidden_states, outputs)


def _generate_stable_symmetric_matrix(hidden_state_dim, eigvalues=None):
  """Generates a symmetric matrix with spectral radius <= 1.

  Args:
    hidden_state_dim: Desired dimension.
    eigvalues: Specified eigenvalues, optional. If None, random eigenvalues will
      be generated from uniform[-1, 1].

  Returns:
    A numpy array of shape [hidden_state_dim, hidden_state_dim] represending a
    symmetric matrix with spectral radius <= 1.
  """
  # Generate eigenvalues.
  if eigvalues is None:
    eigvalues = np.random.uniform(-1.0, 1.0, hidden_state_dim)
  diag_matrix = np.diag(eigvalues)
  if hidden_state_dim == 1:
    change_of_basis = np.ones([1, 1])
  else:
    change_of_basis = ortho_group.rvs(hidden_state_dim)
  # transition_matrix = change_of_basis diag_matrix change_of_basis^T
  transition_matrix = np.matmul(
      np.matmul(change_of_basis, diag_matrix), change_of_basis.transpose())
  # Check that the transition_matrix has to recover the correct eigvalues.
  if np.linalg.norm(
      np.sort(np.linalg.eigvals(transition_matrix)) -
      np.sort(eigvalues)) > 1e-6:
    raise ValueError('Eigenvalues do not match.')
  return transition_matrix


def generate_linear_dynamical_system(hidden_state_dim,
                                     input_dim=1,
                                     output_dim=1,
                                     eigvalues=None,
                                     diagonalizable=True):
  """Generates a LinearDynamicalSystem with given dimensions.

  Args:
    hidden_state_dim: Desired hidden state dim.
    input_dim: The input dim.
    output_dim: Desired output dim.
    eigvalues: Specified eigenvalues, optional. If None, random eigenvalues will
      be generated from uniform[-1, 1] when diagonalizable = True.
    diagonalizable: Whether to generate diagonalizable LDS. If True, generate
      eigvalues first and then transform the eigenvalues through orthogonal
      matrices to get transition matrix. Otherwise, if False, generate random
      hidden_state_dim x hidden_state_dim transition matrix.

  Returns:
    A LinearDynamicalSystem object with
    - A random stable symmetric transition matrx.
    - Identity input matrix.
    - A random output matrix.
  """
  if not diagonalizable and eigvalues is None:
    spectral_radius = np.inf
    while spectral_radius > 1.0:
      transition_matrix = np.random.rand(hidden_state_dim, hidden_state_dim)
      spectral_radius = np.max(np.abs(np.linalg.eig(transition_matrix)[0]))
      # print(np.linalg.eig(transition_matrix)[0])
  else:
    if eigvalues is None:
      eigvalues = np.random.uniform(size=hidden_state_dim)
    # Generate a symmetric transition matrix A.
    transition_matrix = _generate_stable_symmetric_matrix(
        hidden_state_dim, eigvalues)
  input_matrix = np.random.rand(hidden_state_dim, input_dim)
  output_matrix = np.random.rand(output_dim, hidden_state_dim)
  return LinearDynamicalSystem(transition_matrix, input_matrix, output_matrix)


def eig_dist(system1, system2):
  """Computes the eigenvalue distance between two LDS's.

  Args:
    system1: A LinearDynamicalSystem object.
    system2: A LinearDynamicalSystem object.

  Returns:
    Frobenious norm between ordered eigenvalues.
  """
  return np.linalg.norm(system1.get_spectrum() - system2.get_spectrum())


class LinearDynamicalSystemMLEModel(sm_api.tsa.statespace.MLEModel):
  """Class for MLE Model for linear dynamical system.

  For more details on the parameters, see
  https://www.statsmodels.org/dev/generated/
  statsmodels.tsa.statespace.representation.Representation.html
  """

  def __init__(self, endog, k_states, exog=None):
    """Initializes the MLE model.

    Args:
      endog: The observed sequences of shape [n_obs, obs_dim].
      k_states: Hidden state dim.
      exog: Exogenous variables to include in the model.
    """
    endog = endog.reshape((-1, 1))
    super(LinearDynamicalSystemMLEModel, self).__init__(
        endog, k_states, exog, initialization='stationary')
    self['selection'] = np.eye(k_states)
    self.param_shape_dict = collections.OrderedDict()
    # We assume the transition matrix is diagonal with stable real eigenvalues.
    # Here the variable contains the eigenvalues, and we translate that into
    # the transition matrix in the update function.
    self.param_shape_dict['transition'] = (self.k_states)
    self.param_shape_dict['design'] = (1, self.k_states, 1)
    self.param_shape_dict['obs_cov'] = (1, 1, 1)
    self.param_shape_dict['state_intercept'] = (self.k_states, 1)
    self.param_shape_dict['obs_intercept'] = (1, 1)
    # We assume that all state dimensions have equal var and no cov.
    self.param_shape_dict['state_cov'] = (1)
    self.total_param_len = sum(
        [np.prod(s) for s in self.param_shape_dict.values()])

  def _sigmoid(self, x):
    return 1. / (1. + np.exp(-x))

  def _inverse_sigmoid(self, y):
    return np.log(y / (1. - y))

  def transform_params(self, unconstrained):
    """Transform unconstrained parameters used by the optimizer to constrained.

    Args:
      unconstrained: Array of unconstrained parameters used by the optimizer, to
        be transformed.

    Returns:
      Array of constrained parameters which may be used in likelihood evalation.
    """
    constrained = np.copy(unconstrained)
    param_ind = 0
    for k, s in self.param_shape_dict.iteritems():
      if k == 'transition':
        # Constrain the eigenvalues of transition matrix to between -1 and 1.
        # Map the real line to (-1, 1) by sigmoid(x) * 2 - 1.
        constrained[param_ind:param_ind + int(np.prod(s))] = self._sigmoid(
            unconstrained[param_ind:param_ind + int(np.prod(s))]) * 2. - 1.
      if k == 'state_cov' or k == 'obs_cov':
        assert np.prod(s) == 1
        # Constrain covariance to be positive by taking exponent.
        constrained[param_ind] = np.exp(unconstrained[param_ind])
      param_ind += np.prod(s)
    return np.array(constrained, ndmin=1)

  def untransform_params(self, constrained):
    """Transform unconstrained parameters used by the optimizer to constrained.

    Args:
      constrained: Array of transformed constrained parameters.

    Returns:
      Array of unconstrained parameters used by the optimizer.
    """
    unconstrained = np.copy(constrained)
    # Constrain the eigenvalues of the transition matrix to be between -1 and 1.
    param_ind = 0
    for k, s in self.param_shape_dict.iteritems():
      if k == 'transition':
        # Inverse of mapping the real line to (-1, 1) by sigmoid(x) * 2 - 1.
        # The inverse is x = inverse_sigmoid((y + 1) / 2).
        constrained_eig = constrained[param_ind:param_ind + int(np.prod(s))]
        unconstrained[param_ind:param_ind + int(np.prod(s))] = (
            self._inverse_sigmoid((constrained_eig + 1.) / 2.))
      if k == 'state_cov' or k == 'obs_cov':
        assert np.prod(s) == 1
        # Undo constrain covariance to be positive by taking exponent.
        unconstrained[param_ind] = np.log(constrained[param_ind])
      param_ind += np.prod(s)
    return np.array(unconstrained, ndmin=1)

  def update(self, params, transformed=True, **kwargs):
    """Updates the linear dynamical system params."""
    params = super(LinearDynamicalSystemMLEModel, self).update(params, **kwargs)
    param_ind = 0
    for k, s in self.param_shape_dict.iteritems():
      if k == 'transition':
        self[k] = np.diag(params[param_ind:param_ind + int(np.prod(s))])
      elif k == 'state_cov':
        assert np.prod(s) == 1
        state_var = params[param_ind]
        self[k] = state_var * np.eye(self.k_states).reshape(
            self.k_states, self.k_states, 1)
      else:
        self[k] = params[param_ind:param_ind + int(np.prod(s))].reshape(s)
      param_ind += np.prod(s)

  @property
  def start_params(self):
    """Returns the default start params (transformed)."""
    start_values_dict = collections.OrderedDict()
    start_values_dict['transition'] = np.random.uniform(
        low=-1, high=1, size=(self.k_states))
    start_values_dict['design'] = np.random.rand(self.k_states).reshape(
        (1, self.k_states, 1))
    start_values_dict['obs_cov'] = np.ones((1, 1, 1))
    start_values_dict['state_intercept'] = 0.1 * np.random.rand(
        self.k_states).reshape((self.k_states, 1))
    start_values_dict['obs_intercept'] = 0.1 * np.random.rand(1).reshape((1, 1))
    start_values_dict['state_cov'] = np.ones((1))
    assert start_values_dict.keys() == self.param_shape_dict.keys()
    return np.concatenate([p.flatten() for p in start_values_dict.values()],
                          axis=0)


def fit_lds_gibbs(seq, inputs, guessed_dim, num_update_samples):
  """Fits LDS model via Gibbs sampling and EM. Returns fitted eigenvalues."""
  if inputs is None:
    model = DefaultLDS(D_obs=1, D_latent=guessed_dim, D_input=0)
  else:
    model = DefaultLDS(D_obs=1, D_latent=guessed_dim, D_input=1)
  model.add_data(seq, inputs=inputs)
  ll = np.zeros(num_update_samples)
  # Run the Gibbs sampler
  for i in xrange(num_update_samples):
    try:
      model.resample_model()
    except AssertionError as e:
      warnings.warn(str(e), sm_exceptions.ConvergenceWarning)
      eigs = np.linalg.eigvals(model.A)
      return eigs[np.argsort(np.abs(eigs))[::-1]]
    ll[i] = model.log_likelihood()
  # Rough estimate of convergence: judge converged if the change of maximum
  # log likelihood is less than tolerance.
  recent_steps = int(num_update_samples / 10)
  tol = 1.0
  if np.max(ll[-recent_steps:]) - np.max(ll[:-recent_steps]) > tol:
    warnings.warn('Questionable convergence. Log likelihood values: ' + str(ll),
                  sm_exceptions.ConvergenceWarning)
  eigs = np.linalg.eigvals(model.A)
  return eigs[np.argsort(eigs.real)[::-1]]


def fit_lds_mle(seq, inputs, guessed_dim):
  """Returns the fitted eigenvalues via MLE estimation."""
  model = LinearDynamicalSystemMLEModel(seq, guessed_dim, exog=inputs)
  try:
    model.fit(disp=0)
  except np.linalg.LinAlgError:
    # Try to restart with different random initialization.
    try:
      model.fit(disp=0)
    except np.linalg.LinAlgError as e:
      warnings.warn(str(e), sm_exceptions.ConvergenceWarning)
      return np.zeros(guessed_dim)
  eigs = np.linalg.eig(model['transition'])[0]
  return eigs[np.argsort(eigs.real)[::-1]]
