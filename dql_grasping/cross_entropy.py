# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Cross-entropy method for continuous optimization.

Given some parametric family of sampling densities, the cross-entropy method
will adaptively select a set of parameters that minimizes the KL divergence
(cross-entropy) between the sampling distribution and points with high objective
function value.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import numpy as np


def CrossEntropyMethod(sample_fn,
                       objective_fn,
                       update_fn,
                       initial_params,
                       num_elites,
                       num_iterations=1,
                       threshold_to_terminate=None):
  """Uses the cross-entropy method (CEM) to maximize an objective function.

  Definition of 'sample batches':
  This function operates on 'sample batches' returned by sample_fn and received
  by objective_fn and update_fn. Sample batches can be either represented as
  lists `[x0, ..., xn]` of n samples or as dicts that map `str` keys to sample
  lists.

  Args:
    sample_fn: A sampling function that produces samples from some
      distribution. Inputs are arbitrary parameters `**params` to the sampling
      function; output is a sample batch as specified above.
    objective_fn: An objective function to evaluate the sampled points. Input is
      a sample batch as specified above; output is a list of scalars
      `[v0, ..., vn]` representing the objective function evaluated at the
      sampled points.
    update_fn: An update function that chooses new parameters to the sampling
      function. Inputs are a dictionary `params` representing the current
      parameters to the sampling function and a (elite) sample batch as
      specified above`; outputs is a dictionary `updated_params` representing
      the updated parameters to the sampling function.
    initial_params: A dictionary of initial parameters to the sampling function.
    num_elites: The number of elite samples to pass on to the update function.
    num_iterations: The number of iterations to perform.
    threshold_to_terminate: When provided, the function may terminate earlier
        than specified num_iterations if the best inference value is greater
        than threshold_to_terminate.

  Returns:
    final_samples: The final list of sampled points `[x0, ..., xn]`.
    final_values: The final list of scalars `[v0, ..., vn]` representing the
      objective function evaluated at the sampled points.
    final_params: A dictionary of final parameters to the sampling function.
  """
  updated_params = initial_params

  for _ in range(num_iterations):
    # Draw samples from the sampling function.
    samples = sample_fn(**updated_params)

    # Evaluate the samples with the objective function.
    values = objective_fn(samples)

    if isinstance(samples, dict):
      # Sort the samples in ascending order.
      sample_order = [
          i for i, _ in sorted(enumerate(values), key=operator.itemgetter(1))
      ]
      sorted_samples = {
          k: [v[i] for i in sample_order] for k, v in samples.items()
      }

      # Identify the elite samples.
      elite_samples = {k: v[-num_elites:] for k, v in sorted_samples.items()}
    else:
      # Sort the samples in ascending order.
      sorted_samples = [
          s for s, _ in sorted(zip(samples, values), key=operator.itemgetter(1))
      ]

      # Identify the elite samples.
      elite_samples = sorted_samples[-num_elites:]

    # Update the parameters of the sampling distribution.
    updated_params = update_fn(updated_params, elite_samples)

    if ((threshold_to_terminate is not None) and
        (max(values) > threshold_to_terminate)):
      break

  return samples, values, updated_params


def NormalCrossEntropyMethod(objective_fn,
                             mean,
                             stddev,
                             num_samples,
                             num_elites,
                             num_iterations=1):
  """Uses CEM with a normal distribution as the sampling function.

  Args:
    objective_fn: An objective function to evaluate the sampled points. Input is
      a list of sampled points `[x0, ..., xn]`, output is a list of scalars
      `[v0, ..., vn]` representing the objective function evaluated at the
      sampled points.
    mean: A scalar or list of scalars representing the initial means.
    stddev: A scalar or list of scalars representing the initial stddevs.
    num_samples: The number of samples at each iteration.
    num_elites: The number of elite samples at each iteration.
    num_iterations: The number of iterations to perform.

  Returns:
    mean: A list of scalars representing the final means.
    stddev: A list of scalars representing the final stddevs.
  """
  size = np.broadcast(mean, stddev).size

  def _SampleFn(mean, stddev):
    return mean + stddev * np.random.randn(num_samples, size)

  def _UpdateFn(params, elite_samples):
    del params
    return {
        'mean': np.mean(elite_samples, axis=0),
        'stddev': np.std(elite_samples, axis=0, ddof=1),  # Bessel's correction
    }

  initial_params = {'mean': mean, 'stddev': stddev}
  _, _, final_params = CrossEntropyMethod(
      _SampleFn,
      objective_fn,
      _UpdateFn,
      initial_params,
      num_elites,
      num_iterations=num_iterations)

  return final_params['mean'], final_params['stddev']
