# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Experiment and plotting code for hitting set mechanisms."""

import collections
import enum
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

from dp_mm_domain import hitting_set_mechanisms
from dp_mm_domain import peeling_mechanisms
from dp_mm_domain import utils


class Metric(enum.Enum):
  """Enum class for hitting set error metrics."""
  MISSED_USERS = 1
  RUNTIME = 2

_METRIC_NAMES = {
    Metric.MISSED_USERS: "No. Missed Users",
    Metric.RUNTIME: "Runtime",
}


class HittingSetMethod(enum.Enum):
  """Enum class for hitting set methods."""
  WGM_AND_PEEL = 1
  GREEDY = 2
  NON_PRIVATE_DOMAIN_AND_PEEL = 3


_PARTIAL_METHODS = {
    HittingSetMethod.WGM_AND_PEEL: peeling_mechanisms.wgm_then_peel_mechanism,
    HittingSetMethod.GREEDY: hitting_set_mechanisms.greedy_hitting_set,
    HittingSetMethod.NON_PRIVATE_DOMAIN_AND_PEEL: (
        peeling_mechanisms.user_peeling_mechanism
    ),
}

_PLOT_LABELS = {
    HittingSetMethod.WGM_AND_PEEL: "WGM-And-Peel",
    HittingSetMethod.GREEDY: "Greedy",
    HittingSetMethod.NON_PRIVATE_DOMAIN_AND_PEEL: "Non-Private-Domain-And-Peel",
}


def compute_missed_users(input_data, output):
  """Computes the number of users who do not have any items in the output set.

  Args:
    input_data: A list where each inner list represents a user's items. This is
      used to compute the frequency histogram.
    output: A list of items selected by the mechanism.

  Returns:
    The number of users who do not have any items in the output set.
  """

  items_to_users = utils.get_items_to_users(input_data)
  hit_users = set()
  for item in output:
    hit_users.update(items_to_users[item])
  return len(input_data) - len(hit_users)


def run_method_across_ks(input_data, method, k_range, params, num_trials=1):
  """Runs a method for the hitting set problem across given k_range.

  Args:
    input_data: A list where each inner list represents a user's items. For
      example, `[[1, 2, 3], [4, 5]]`.
    method: A `HittingSetMethod` enum to run.
    k_range: A list of values for k.
    params: A dictionary containing method-specific parameters. For
      `WGM_AND_PEEL`, this includes `l0_bound`, `eps`, and `delta`. For
      `NON_PRIVATE_DOMAIN_AND_PEEL`, this includes `eps`, and `delta`.
    num_trials: The number of trials to run for each k.

  Returns:
    A dictionary where the keys are k values and the values are dictionaries
    containing the mean and standard error of the metric values across all
    trials.
  """
  results_per_k = {}

  for k in k_range:
    print("running " + _PLOT_LABELS[method] + " with k = " + str(k))

    trial_results = collections.defaultdict(list)
    for _ in range(num_trials):
      begin = time.time()
      l0_bound = params["l0_bound"]
      eps = params["eps"]
      delta = params["delta"]
      if method == HittingSetMethod.WGM_AND_PEEL:
        eps_schedule = [eps/2, eps/2]
        delta_schedule = [delta/2, delta/2]
        output = _PARTIAL_METHODS[method](
            input_data,
            k,
            eps_schedule,
            delta_schedule,
            l0_bound,
            peel_users=True,
        )
      elif method == HittingSetMethod.GREEDY:
        output = _PARTIAL_METHODS[method](input_data, k)
      else:
        domain = set(itertools.chain.from_iterable(input_data))
        output = _PARTIAL_METHODS[method](input_data, domain, k, eps, delta)

      end = time.time()
      missed_users = compute_missed_users(input_data, output)

      trial_results[Metric.MISSED_USERS].append(missed_users)
      trial_results[Metric.RUNTIME].append(end - begin)

    results_per_k[k] = {
        Metric.MISSED_USERS: (
            np.mean(trial_results[Metric.MISSED_USERS]),
            np.std(trial_results[Metric.MISSED_USERS]) / np.sqrt(num_trials),
        ),
        Metric.RUNTIME: (
            np.mean(trial_results[Metric.RUNTIME]),
            np.std(trial_results[Metric.RUNTIME]) / np.sqrt(num_trials),
        ),
    }

    print("finished " + _PLOT_LABELS[method] + " with k = " + str(k))
  return results_per_k


def compare_methods(
    input_data,
    methods,
    k_range,
    epsilon,
    delta,
    l0_bound,
    num_trials,
):
  """Compares all methods with the given l0_bound, epsilon, and delta.

  Args:
    input_data: A list where each inner list represents a user's items. For
      example, `[[1, 2, 3], [4, 5]]`.
    methods: A list of `HittingSetMethod` enums to compare.
    k_range: A list of k values to try for each method.
    epsilon: The privacy parameter epsilon.
    delta: The privacy parameter delta.
    l0_bound: The l0_bound parameter.
    num_trials: The number of trials to run for each configuration.

  Returns:
    Dictionary of results for each method.
  Raises:
    ValueError: If a method is not supported.
  """
  results_per_method = {}
  params = {
      "l0_bound": l0_bound,
      "eps": epsilon,
      "delta": delta,
  }
  for method in methods:
    if method not in _PARTIAL_METHODS:
      raise ValueError(f"Unsupported method: {method}")
    else:
      results_per_method[method] = run_method_across_ks(
          input_data, method, k_range, params, num_trials
      )

  return results_per_method


def plot_results(results, k_range, epsilon, delta, output_path):
  """Plots the results for all methods across all k values and metrics.

  The `results` dictionary is expected to be the output of `compare_methods`
  and has the following structure:
  {
      HittingSetMethod.WGM_AND_PEEL: {
          k_1: {
              Metric.MISSED_USERS: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          k_2: {
              Metric.MISSED_USERS: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          ...
      },
      HittingSetMethod.GREEDY: {
          k_1: {
              Metric.MISSED_USERS: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          k_2: {
              Metric.MISSED_USERS: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          ...
      },
      HittingSetMethod.KNOWN_DOMAIN_PEELING: {
          k_1: {
              Metric.MISSED_USERS: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          k_2: {
              Metric.MISSED_USERS: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          ...
      },
      ...
  }
  The outer dictionary is keyed by method. The inner dictionary is keyed by k
  value. The innermost dictionaries contain the mean and standard error of the
  metric values across all trials.

  Args:
    results: A dictionary of results for each method.
    k_range: A list of k values used in the experiment.
    epsilon: The privacy parameter epsilon.
    delta: The privacy parameter delta.
    output_path: The base path to save the plot images.
  """

  for metric in Metric:
    for method in results.keys():
      result_lists = [results[method][k] for k in k_range]
      mean_metric_values = [result[metric][0] for result in result_lists]
      stderr_metric_values = [result[metric][1] for result in result_lists]
      plt.errorbar(
          k_range,
          mean_metric_values,
          yerr=stderr_metric_values,
          label=_PLOT_LABELS[method],
          linewidth=3,
      )
      plt.xlabel("k")
      plt.ylabel(_METRIC_NAMES[metric])
      ax = plt.gca()
      ax.tick_params(labelsize=18)
      plt.legend(
          loc="lower center",
          bbox_to_anchor=(0.45, -0.4),
          ncol=3,
          frameon=False,
          fontsize=16,
      )
    path = (
        output_path
        + "/"
        + _METRIC_NAMES[metric]
        + "_epsilon_"
        + str(epsilon)
        + "_delta_"
        + str(delta)
        + ".png"
    )
    plt.savefig(path)
    plt.close()
