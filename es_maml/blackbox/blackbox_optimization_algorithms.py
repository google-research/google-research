# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

r"""Library of blackbox optimization algorithms.

Library of stateful blackbox optimization algorithms taking as input the values
of the blackbox function in the neighborhood of a given point and outputting new
point obtained after conducting one optimization step.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
import numpy as np

from sklearn import linear_model


def filter_top_directions(perturbations, function_values, est_type,
                          num_top_directions):
  """Select the subset of top-performing perturbations.

  TODO(b/139662389): In the future, we may want (either here or inside the
  perturbation generator) to add assertions that Antithetic perturbations are
  delivered in the expected order (i.e (p_1, -p_1, p_2, -p_2,...)).

  Args:
    perturbations: np array of perturbations
                   For antithetic, it is assumed that the input puts the pair of
                   p, -p in the even/odd entries, so the directions p_1,...,p_n
                   will be ordered (p_1, -p_1, p_2, -p_2,...)
    function_values: np array of reward values (maximization)
    est_type: (forward_fd | antithetic)
    num_top_directions: the number of top directions to include
                        For antithetic, the total number of perturbations will
                        be 2* this number, because we count p, -p as a single
                        direction
  Returns:
    A pair (perturbations, function_values) consisting of the top perturbations.
    function_values[i] is the reward of perturbations[i]
    For antithetic, the perturbations will be reordered so that we have
    (p_1,...,p_n, -p_1,...,-p_n).
  """
  if not num_top_directions > 0:
    return (perturbations, function_values)
  if est_type == "forward_fd":
    top_index = np.argsort(-function_values)
  elif est_type == "antithetic":
    top_index = np.argsort(-np.abs(function_values[0::2] - function_values[1::2]
                                  ))
  top_index = top_index[:num_top_directions]
  if est_type == "forward_fd":
    perturbations = perturbations[top_index]
    function_values = function_values[top_index]
  elif est_type == "antithetic":
    perturbations = np.concatenate((perturbations[2*top_index],
                                    perturbations[2*top_index + 1]), axis=0)
    function_values = np.concatenate((function_values[2*top_index],
                                      function_values[2*top_index + 1]), axis=0)
  return (perturbations, function_values)


class BlackboxOptimizer(object):
  """Abstract class for general blackbox optimization.

  Class is responsible for encoding different blackbox optimization techniques.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def run_step(self, perturbations, function_values, current_input,
               current_value):
    """Conducts a single step of blackbox optimization procedure.

    Conducts a single step of blackbox optimization procedure, given values of
    the blackox function in the neighborhood of the current input.

    Args:
      perturbations: perturbation directions encoded as 1D numpy arrays
      function_values: corresponding function values
      current_input: current input to the blackbox function
      current_value: value of the blackbox function for the current input

    Returns:
      New input obtained by conducting a single step of the blackbox
      optimization procedure.
    """
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def get_hyperparameters(self):
    """Returns the list of hyperparameters for blackbox function runs.

    Returns the list of hyperparameters for blackbox function runs that can be
    updated on the fly.

    Args:

    Returns:
      The set of hyperparameters for blackbox function runs.
    """
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def get_state(self):
    """Returns the state of the optimizer.

    Returns the state of the optimizer.

    Args:

    Returns:
      The state of the optimizer.
    """
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def update_state(self, evaluation_stats):
    """Updates the state for blackbox function runs.

    Updates the state of the optimizer for blackbox function runs.

    Args:
      evaluation_stats: stats from evaluation used to update hyperparameters

    Returns:
    """
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def set_state(self, state):
    """Sets up the internal state of the optimizer.

    Sets up the internal state of the optimizer.

    Args:
      state: state to be set up

    Returns:
    """
    raise NotImplementedError("Abstract method")


class MCBlackboxOptimizer(BlackboxOptimizer):
  """Class implementing GD optimizer with MC estimation of the gradient."""

  def __init__(self, precision_parameter, est_type, normalize_fvalues,
               hyperparameters_update_method, extra_params, step_size,
               num_top_directions):
    self.precision_parameter = precision_parameter
    self.est_type = est_type
    self.normalize_fvalues = normalize_fvalues
    self.hyperparameters_update_method = hyperparameters_update_method
    self.step_size = step_size
    self.num_top_directions = num_top_directions
    if hyperparameters_update_method == "state_normalization":
      self.state_dim = extra_params[0]
      self.nb_steps = 0
      self.sum_state_vector = [0.0] * self.state_dim
      self.squares_state_vector = [0.0] * self.state_dim
      self.mean_state_vector = [0.0] * self.state_dim
      self.std_state_vector = [1.0] * self.state_dim

  def run_step(self, perturbations, function_values, current_input,
               current_value):
    dim = len(current_input)
    if self.normalize_fvalues:
      values = function_values.tolist()
      values.append(current_value)
      mean = sum(values) / float(len(values))
      stdev = np.std(np.array(values))
      normalized_values = [(x - mean) / stdev for x in values]
      function_values = np.array(normalized_values[:-1])
      current_value = normalized_values[-1]
    top_ps, top_fs = filter_top_directions(perturbations, function_values,
                                           self.est_type,
                                           self.num_top_directions)
    gradient = np.zeros(dim)
    for i, perturbation in enumerate(top_ps):
      function_value = top_fs[i]
      if self.est_type == "forward_fd":
        gradient_sample = (function_value - current_value) * perturbation
      elif self.est_type == "antithetic":
        gradient_sample = function_value * perturbation
      gradient_sample /= self.precision_parameter ** 2
      gradient += gradient_sample
    gradient /= len(top_ps)
    # this next line is for compatibility with the Blackbox used for Toaster.
    # in that code, the denominator for antithetic was num_top_directions.
    # we maintain compatibility for now so that the same hyperparameters
    # currently used in Toaster will have the same effect
    if self.est_type == "antithetic" and len(top_ps) < len(perturbations):
      gradient *= 2
    return current_input + self.step_size * gradient

  def get_hyperparameters(self):
    if self.hyperparameters_update_method == "state_normalization":
      return self.mean_state_vector + self.std_state_vector
    else:
      return []

  def get_state(self):
    if self.hyperparameters_update_method == "state_normalization":
      current_state = [self.nb_steps]
      current_state += self.sum_state_vector
      current_state += self.squares_state_vector
      current_state += self.mean_state_vector + self.std_state_vector
      return current_state
    else:
      return []

  def update_state(self, evaluation_stats):
    if self.hyperparameters_update_method == "state_normalization":
      self.nb_steps += evaluation_stats[0]
      evaluation_stats = evaluation_stats[1:]
      first_half = evaluation_stats[:self.state_dim]
      second_half = evaluation_stats[self.state_dim:]
      self.sum_state_vector = [
          sum(x) for x in zip(self.sum_state_vector, first_half)
      ]
      self.squares_state_vector = [
          sum(x) for x in zip(self.squares_state_vector, second_half)
      ]
      self.mean_state_vector = [
          x / float(self.nb_steps) for x in self.sum_state_vector
      ]
      mean_squares_state_vector = [
          x / float(self.nb_steps) for x in self.squares_state_vector
      ]

      self.std_state_vector = [
          math.sqrt(max(a - b * b, 0.0))
          for a, b in zip(mean_squares_state_vector, self.mean_state_vector)
      ]

  def set_state(self, state):
    if self.hyperparameters_update_method == "state_normalization":
      self.nb_steps = state[0]
      state = state[1:]
      self.sum_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.squares_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.mean_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.std_state_vector = state[:self.state_dim]


class GeneralRegressionBlackboxOptimizer(BlackboxOptimizer):
  """Class implementing GD optimizer with regression for grad. estimation."""

  def __init__(self, regression_method, regularizer, est_type,
               normalize_fvalues, hyperparameters_update_method, extra_params,
               step_size):
    self.normalize_fvalues = normalize_fvalues
    self.regression_method = regression_method
    self.regularizer = regularizer
    self.est_type = est_type
    self.step_size = step_size
    self.hyperparameters_update_method = hyperparameters_update_method
    if hyperparameters_update_method == "state_normalization":
      self.state_dim = extra_params[0]
      self.nb_steps = 0
      self.sum_state_vector = [0.0] * self.state_dim
      self.squares_state_vector = [0.0] * self.state_dim
      self.mean_state_vector = [0.0] * self.state_dim
      self.std_state_vector = [1.0] * self.state_dim

  def run_step(self, perturbations, function_values, current_input,
               current_value):
    dim = len(current_input)
    if self.normalize_fvalues:
      values = function_values.tolist()
      values.append(current_value)
      mean = sum(values) / float(len(values))
      stdev = np.std(np.array(values))
      normalized_values = [(x - mean) / stdev for x in values]
      function_values = np.array(normalized_values[:-1])
      current_value = normalized_values[-1]
    atranspose = None
    b_vector = None
    if self.est_type == "forward_fd":
      atranspose = np.transpose(np.array(perturbations))
      b_vector = (
          function_values - np.array([current_value] * len(function_values)))
    elif self.est_type == "antithetic":
      atranspose = np.transpose(np.array(perturbations[::2]))
      function_even_values = np.array(function_values.tolist()[::2])
      function_odd_values = np.array(function_values.tolist()[1::2])
      b_vector = (function_even_values - function_odd_values) / 2.0
    else:
      raise ValueError("FD method not available.")
    b_vector_transformed = []
    for i in range(len(b_vector)):
      b_vector_transformed.append([b_vector[i]])
    b_vector_transformed = np.array(b_vector_transformed)
    gradient = self.regression_method(atranspose, b_vector_transformed,
                                      self.regularizer)
    np.reshape(gradient, (dim))
    return current_input + self.step_size * np.reshape(gradient,
                                                       (len(current_input)))

  def get_hyperparameters(self):
    if self.hyperparameters_update_method == "state_normalization":
      return self.mean_state_vector + self.std_state_vector
    else:
      return []

  def get_state(self):
    if self.hyperparameters_update_method == "state_normalization":
      current_state = [self.nb_steps]
      current_state += self.sum_state_vector
      current_state += self.squares_state_vector
      current_state += self.mean_state_vector + self.std_state_vector
      return current_state
    else:
      return []

  def update_state(self, evaluation_stats):
    if self.hyperparameters_update_method == "state_normalization":
      self.nb_steps += evaluation_stats[0]
      evaluation_stats = evaluation_stats[1:]
      first_half = evaluation_stats[:self.state_dim]
      second_half = evaluation_stats[self.state_dim:]
      self.sum_state_vector = [
          sum(x) for x in zip(self.sum_state_vector, first_half)
      ]
      self.squares_state_vector = [
          sum(x) for x in zip(self.squares_state_vector, second_half)
      ]
      self.mean_state_vector = [
          x / float(self.nb_steps) for x in self.sum_state_vector
      ]
      mean_squares_state_vector = [
          x / float(self.nb_steps) for x in self.squares_state_vector
      ]

      self.std_state_vector = [
          math.sqrt(max(a - b * b, 0.0))
          for a, b in zip(mean_squares_state_vector, self.mean_state_vector)
      ]

  def set_state(self, state):
    if self.hyperparameters_update_method == "state_normalization":
      self.nb_steps = state[0]
      state = state[1:]
      self.sum_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.squares_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.mean_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.std_state_vector = state[:self.state_dim]


class SklearnRegressionBlackboxOptimizer(BlackboxOptimizer):
  """Class implementing GD optimizer with regression for grad. estimation."""

  def __init__(self, regression_method, regularizer, est_type,
               normalize_fvalues, hyperparameters_update_method, extra_params,
               step_size):
    if regression_method == "lasso":
      self.clf = linear_model.Lasso(alpha=regularizer)
    elif regression_method == "ridge":
      self.clf = linear_model.Ridge(alpha=regularizer)
    elif regression_method == "vanilla_regression":
      self.clf = linear_model.LinearRegression()
    else:
      raise ValueError("Optimization procedure option not available")
    self.normalize_fvalues = normalize_fvalues
    self.est_type = est_type
    self.step_size = step_size
    self.hyperparameters_update_method = hyperparameters_update_method
    if hyperparameters_update_method == "state_normalization":
      self.state_dim = extra_params[0]
      self.nb_steps = 0
      self.sum_state_vector = [0.0] * self.state_dim
      self.squares_state_vector = [0.0] * self.state_dim
      self.mean_state_vector = [0.0] * self.state_dim
      self.std_state_vector = [1.0] * self.state_dim

  def run_step(self, perturbations, function_values, current_input,
               current_value):
    dim = len(current_input)
    if self.normalize_fvalues:
      values = function_values.tolist()
      values.append(current_value)
      mean = sum(values) / float(len(values))
      stdev = np.std(np.array(values))
      normalized_values = [(x - mean) / stdev for x in values]
      function_values = np.array(normalized_values[:-1])
      current_value = normalized_values[-1]

    matrix = None
    b_vector = None
    if self.est_type == "forward_fd":
      matrix = np.array(perturbations)
      b_vector = (
          function_values - np.array([current_value] * len(function_values)))
    elif self.est_type == "antithetic":
      matrix = np.transpose(np.array(perturbations[::2]))
      function_even_values = np.array(function_values.tolist()[::2])
      function_odd_values = np.array(function_values.tolist()[1::2])
      b_vector = (function_even_values - function_odd_values) / 2.0
    else:
      raise ValueError("FD method not available.")

    self.clf.fit(matrix, b_vector)
    return current_input + self.step_size * self.clf.coef_[0:dim]

  def get_hyperparameters(self):
    if self.hyperparameters_update_method == "state_normalization":
      return self.mean_state_vector + self.std_state_vector
    else:
      return []

  def get_state(self):
    if self.hyperparameters_update_method == "state_normalization":
      current_state = [self.nb_steps]
      current_state += self.sum_state_vector
      current_state += self.squares_state_vector
      current_state += self.mean_state_vector + self.std_state_vector
      return current_state
    else:
      return []

  def update_state(self, evaluation_stats):
    if self.hyperparameters_update_method == "state_normalization":
      self.nb_steps += evaluation_stats[0]
      evaluation_stats = evaluation_stats[1:]
      first_half = evaluation_stats[:self.state_dim]
      second_half = evaluation_stats[self.state_dim:]
      self.sum_state_vector = [
          sum(x) for x in zip(self.sum_state_vector, first_half)
      ]
      self.squares_state_vector = [
          sum(x) for x in zip(self.squares_state_vector, second_half)
      ]
      self.mean_state_vector = [
          x / float(self.nb_steps) for x in self.sum_state_vector
      ]
      mean_squares_state_vector = [
          x / float(self.nb_steps) for x in self.squares_state_vector
      ]

      self.std_state_vector = [
          math.sqrt(max(a - b * b, 0.0))
          for a, b in zip(mean_squares_state_vector, self.mean_state_vector)
      ]

  def set_state(self, state):
    if self.hyperparameters_update_method == "state_normalization":
      self.nb_steps = state[0]
      state = state[1:]
      self.sum_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.squares_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.mean_state_vector = state[:self.state_dim]
      state = state[self.state_dim:]
      self.std_state_vector = state[:self.state_dim]
