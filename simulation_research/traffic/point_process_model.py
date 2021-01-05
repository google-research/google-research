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

# Lint as: python3
"""Point process model for the traffic flow fitting and generation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import scipy.interpolate


# TODO(albertyuchen): Find better basis creation tools.
def create_bspline_basis(knots, spline_order, dt=0.02):
  """Create B-spline basis."""
  # The repeated boundary knots are appended as it is required for Cox de Boor
  # recursive algorithm. See https://math.stackexchange.com/questions/2817170/
  # what-is-the-purpose-of-having-repeated-knots-in-a-b-spline and the link
  # https://en.wikipedia.org/wiki/De_Boor%27s_algorithm.
  knots = list(knots)
  knots = [knots[0]] * spline_order + knots + [knots[-1]] * spline_order
  num_basis = len(knots) - spline_order - 1
  # Query token is in format: [knots, basis coefficients, spline order]
  # See https://docs.scipy.org/doc/scipy/reference/generated/
  # scipy.interpolate.splev.html
  query_token = [0, 0, spline_order]
  query_token[0] = np.array(knots)
  time_line = np.linspace(knots[0], knots[-1], int(np.round(knots[-1]/dt)) + 1)
  # Add column for the constent term.
  basis_matrix = np.zeros((len(time_line), num_basis + 1))
  basis_matrix[:, -1] = np.ones(len(time_line))  # Constant term.

  for basis_index in range(num_basis):
    basis_coefficients = np.zeros(num_basis)
    basis_coefficients[basis_index] = 1.0
    query_token[1] = basis_coefficients.tolist()
    base = scipy.interpolate.splev(time_line, query_token)
    basis_matrix[:, basis_index] = base
  return basis_matrix, time_line


class PointProcessModel(object):
  """Generates random traffic using inhomogeneous Poisson models."""

  # def __init__(self):

  @classmethod
  def generator(cls, rates, time_step_size):
    """Generate events according to the rates.

    If we know the underlying event rate of a point process, the number of
    events in a certain time interval follows Poisson distribution with
    parameter lambda. The lambda is the integral of the event rate in the time
    interval. See reference:

    Args:
      rates:
      time_step_size:

    Returns:
      Number of events for each bin w.r.t. the `rates`.
    """
    num_events = np.zeros(len(rates))
    for t, rate in enumerate(rates):
      num_events[t] = np.random.poisson(time_step_size * rate, 1)
    return num_events

  @classmethod
  def fit_homo_poisson(cls, events, time_step_size):
    """Fit the homogeneous Poisson model.

    For homogeneous Poisson process, the maximum likelihood estimator for the
    event rate is the mean of the data.

    Args:
      events: An array of numbers of evetns in each time bin.
      time_step_size: Bin width. We assume the bin sizes are equal.
    Returns:
      Event rate.
    """
    return np.mean(events) / time_step_size

  # TODO(albertyuchen): Bin the time points into the
  def time_bin_data(self, timestamps, time_step_size):
    """Bin the events timestamps into time bins.

    This function discretize the timestamps into different time bins, so that
    the model can be fitted using generalized linear models in this class.

    Args:
      timestamps: A list of observed events.
      time_step_size: The time step size.

    Returns:
      events: Time binned events count.
    """
    # events, _ = np.histogram(timestamps, bins=xxx)
    # return events
    pass

  @classmethod
  def fit_inhomo_poisson(cls,
                         events,
                         time_step_size,
                         spline_order=3,
                         num_knots=5):
    """Fits the inhomogeneous Poisson model.

    Args:
      events: A sequence of number of events.
      time_step_size: Time step size.
      spline_order: The order of the spline.
      num_knots: Number of knots inbetween the two ends. The knots distribute
          uniformly.

    Returns:
      Estimated event rates.
    """
    # Creates knots between [0, 1]
    knots = np.linspace(0, 1, num_knots + 2)  # +2 to includes two ends.
    # The number of sampled pooints is the same as those in the basis.
    dt = 1 / (len(events) - 1)
    xx, _ = create_bspline_basis(knots, spline_order, dt)
    yy = events
    beta = np.zeros(xx.shape[1])

    max_iterations = 300
    for _ in range(max_iterations):
      negative_log_likelihoods = -yy.T @ (xx @ beta) + np.exp(xx @ beta)
      negative_log_likelihood = np.sum(negative_log_likelihoods)
      logging.info('Negative log likelihood: %s', negative_log_likelihood)
      gradient = -xx.T @ yy + xx.T @ np.exp(xx @ beta)
      # TODO(albertyuchen): Apply backtracking line search.
      # The method is described: https://www.stat.cmu.edu/~ryantibs/convexopt/
      # lectures/grad-descent.pdf.
      beta -= gradient * 0.001
      # TODO(albertyuchen): Apply Newton method here by multiplying the Hessian.
      # The Newton's method requires careful backtracking line search.
      # hessian = xx.T @ (np.exp(xx @ beta).reshape(-1, 1) * xx)
      # beta -= hessian @ gradient * 0.0001
      # TODO(albertyuchen): Add convergence condition.
      # if |NLL(t) - NLL(t + 1)| < delta

    return np.exp(xx @ beta) / time_step_size
