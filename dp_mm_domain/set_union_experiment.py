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

"""Experiment and plotting code for set union mechanisms."""

import collections
import enum
import time

import matplotlib.pyplot as plt
import numpy as np

from dp_mm_domain import policy_mechanisms
from dp_mm_domain import utils
from dp_mm_domain import weighted_gaussian


class Metric(enum.Enum):
  CAPTURED_MASS = 1
  CARDINALITY = 2
  RUNTIME = 3


class SetUnionMethod(enum.Enum):
  WGM = 1
  POLICY_GAUSSIAN = 2
  POLICY_GREEDY = 3


_PARTIAL_METHODS = {
    SetUnionMethod.WGM: weighted_gaussian.weighted_gaussian_mechanism,
    SetUnionMethod.POLICY_GAUSSIAN: policy_mechanisms.policy_gaussian_mechanism,
    SetUnionMethod.POLICY_GREEDY: policy_mechanisms.policy_greedy_mechanism,
}

_PLOT_LABELS = {
    SetUnionMethod.WGM: "WGM",
    SetUnionMethod.POLICY_GAUSSIAN: "Policy Gaussian",
    SetUnionMethod.POLICY_GREEDY: "Policy Greedy",
}

_METRIC_NAMES = {
    Metric.CAPTURED_MASS: "Captured Mass",
    Metric.CARDINALITY: "Cardinality",
    Metric.RUNTIME: "Runtime",
}

_PLOT_COLORS = {
    SetUnionMethod.WGM: "palevioletred",
    SetUnionMethod.POLICY_GREEDY: "darkorange",
    SetUnionMethod.POLICY_GAUSSIAN: "darkgreen",
}


def compute_captured_mass(input_data, output):
  """Computes the "Captured Mass" (CM) given a frequency histogram and a set of selected items.

  The Captured Mass represents the proportion of the total frequency (or "mass")
  in the 'hist' that is covered by the 'output'. In other words, it quantifies
  how much of the original distribution is 'missed' by the selection.

  Args:
    input_data: A list where each inner list represents a user's items. This is
      used to compute the frequency histogram.
    output: A list of items selected by the mechanism.

  Returns:
  The Captured Mass, a value between 0.0 and 1.0.
           - A value of 1.0 means all the mass in `hist` is captured by
           `output`.
           - A value of 0.0 means none of the mass in `hist` is captured.
  """
  hist = utils.get_hist(input_data)
  total_mass = sum(hist.values())
  captured_mass = np.sum([hist[element] for element in output])
  return captured_mass / total_mass


def run_method_across_l0_bounds(
    input_data, method, l0_bounds, epsilon, delta, num_trials=1
):
  """Runs a method for the set union problem on the given data with the given parameters across all l0_bounds.

  Args:
    input_data: A list where each inner list represents a user's items. For
      example, `[[1, 2, 3], [4, 5]]`.
    method: A `SetUnionMethod` enum to run.
    l0_bounds: A list of l0 bounds to use for different experiments.
    epsilon: The privacy parameter epsilon.
    delta: The privacy parameter delta.
    num_trials: The number of trials to run for each l0_bound.

  Returns:
    A dictionary where the keys are l0_bounds and the values are dictionaries
    containing the mean and standard error of the metric values across all
    trials.
  """
  results_per_l0_bound = {}

  for l0_bound in l0_bounds:
    print(
        "running " + _PLOT_LABELS[method] + " with l0_bound = " + str(l0_bound)
    )

    trial_results = collections.defaultdict(list)

    for _ in range(num_trials):
      begin = time.time()
      if method == SetUnionMethod.POLICY_GREEDY:
        output = _PARTIAL_METHODS[method](input_data, epsilon, delta)
      else:
        output = _PARTIAL_METHODS[method](input_data, l0_bound, epsilon, delta)
      end = time.time()
      captured_mass = compute_captured_mass(input_data, output)

      trial_results[Metric.CAPTURED_MASS].append(captured_mass)
      trial_results[Metric.CARDINALITY].append(len(output))
      trial_results[Metric.RUNTIME].append(end - begin)

    results_per_l0_bound[l0_bound] = {
        Metric.CAPTURED_MASS: (
            np.mean(trial_results[Metric.CAPTURED_MASS]),
            np.std(trial_results[Metric.CAPTURED_MASS]) / np.sqrt(num_trials),
        ),
        Metric.CARDINALITY: (
            np.mean(trial_results[Metric.CARDINALITY]),
            np.std(trial_results[Metric.CARDINALITY]) / np.sqrt(num_trials),
        ),
        Metric.RUNTIME: (
            np.mean(trial_results[Metric.RUNTIME]),
            np.std(trial_results[Metric.RUNTIME]) / np.sqrt(num_trials),
        ),
    }

    print(
        "finished " + _PLOT_LABELS[method] + " with l0_bound = " + str(l0_bound)
    )
  return results_per_l0_bound


def compare_methods(
    input_data, methods, l0_bound_range, epsilon, delta, num_trials
):
  """Compares all methods with the given l0_bound, epsilon, and delta.

  Args:
    input_data: A list where each inner list represents a user's
      items. For example, `[[1, 2, 3], [4, 5]]`.
    methods: A list of `SetUnionMethod` enums to compare.
    l0_bound_range: A range of l0_bounds to compare. For example, `[5, 10, 15]`
      will compare the methods with l0_bounds of 5, 10, and 15.
    epsilon: The privacy parameter epsilon.
    delta: The privacy parameter delta.
    num_trials: The number of trials to run for each l0_bound.

  Returns:
    Dictionary of results for each method.
  """
  results_per_method = {}
  for method in methods:
    results_per_method[method] = run_method_across_l0_bounds(
        input_data, method, l0_bound_range, epsilon, delta, num_trials
    )

  return results_per_method


def plot_results(results, l0_bound_range, epsilon, delta, output_path):
  """Plots the results for all methods with the given l0_bound, epsilon, and delta.

  The `results` dictionary is expected to be the output of `compare_methods`
  and has the following structure:
  {
      SetUnionMethod.METHOD_1: {
          l0_bound_1: {
              Metric.CAPTURED_MASS: (mean, stderr),
              Metric.CARDINALITY: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          l0_bound_2: {
              Metric.CAPTURED_MASS: (mean, stderr),
              Metric.CARDINALITY: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          ...
      },
      SetUnionMethod.METHOD_2: {
          l0_bound_1: {
              Metric.CAPTURED_MASS: (mean, stderr),
              Metric.CARDINALITY: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          l0_bound_2: {
              Metric.CAPTURED_MASS: (mean, stderr),
              Metric.CARDINALITY: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          ...
      },
      ...
  }
  The outer dictionary is keyed by method. The inner dictionary is keyed by
  l0_bound. The inner dictionaries contain dictionaries of metric values with
  the mean and standard error across all trials.

  Args:
    results: A dictionary of results for each method.
    l0_bound_range: A range of l0_bounds that were used in the experiment.
    epsilon: The privacy parameter epsilon.
    delta: The privacy parameter delta.
    output_path: The base path to save the plot images.

  Raises:
    ValueError: If the results do not contain the given metric.
  """

  for metric in Metric:
    for method in results.keys():
      result_lists = [results[method][l0_bound] for l0_bound in l0_bound_range]
      mean_metric_values = [result[metric][0] for result in result_lists]
      stderr_metric_values = [result[metric][1] for result in result_lists]

      plt.errorbar(
          l0_bound_range,
          mean_metric_values,
          yerr=stderr_metric_values,
          label=_PLOT_LABELS[method],
          color=_PLOT_COLORS[method],
          linewidth=3,
      )

      plt.xlabel("l0_bound")
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
