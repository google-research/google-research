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

"""Experiment and plotting code for top-k mechanisms."""

import collections
import enum
import time

import matplotlib.pyplot as plt
import numpy as np

from dp_mm_domain import peeling_mechanisms
from dp_mm_domain import top_k
from dp_mm_domain import utils


class Metric(enum.Enum):
  TOPK_MM = 1
  TOPK_L1 = 2
  RUNTIME = 3

_METRIC_NAMES = {
    Metric.TOPK_MM: "TopK Missing Mass",
    Metric.TOPK_L1: "TopK L1 Loss",
    Metric.RUNTIME: "Runtime",
}


class TopKMethod(enum.Enum):
  WGM_THEN_PEEL = 1
  LIMITED_DOMAIN = 2


_PARTIAL_METHODS = {
    TopKMethod.WGM_THEN_PEEL: peeling_mechanisms.wgm_then_peel_mechanism,
    TopKMethod.LIMITED_DOMAIN: top_k.limited_domain_mechanism,
}

_PLOT_LABELS = {
    TopKMethod.WGM_THEN_PEEL: "WGM-Then-Peel",
    TopKMethod.LIMITED_DOMAIN: "Limited Domain",
}


def compute_topk_missing_mass(input_data, output, k):
  """Returns the TopK Missing Mass given a histogram and set of selected items.

  The TopK Missing Mass represents the difference in cumulative frequency
  between the sum of the top-k most frequent items and the sum of the selected
  items. This is a measure of how well the selected items capture the top-k most
  frequent items.

  Args:
    input_data: A list where each inner list represents a user's items. This is
      used to compute the frequency histogram.
    output: A list of items selected by the mechanism.
    k: The number of top items to consider.
  """
  hist = utils.get_hist(input_data)
  total_mass = sum(hist.values())
  captured_mass = np.sum([hist[element] for element in output])
  sorted_hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)
  topk_mass = sum([sorted_hist[i][1] for i in range(k)])
  return (topk_mass - captured_mass) / total_mass


def compute_topk_l1_loss(input_data, output, k):
  """Returns TopK L1 Loss given a histogram and set of selected items.

  The TopK L1 Loss is the L1 norm of the difference between the frequency
  histogram of the top-k most frequent items and the frequency histogram of the
  selected items after sorting both in descending order.

  Args:
    input_data: A list where each inner list represents a user's items. This is
      used to compute the frequency histogram.
    output: A list of items selected by the mechanism.
    k: The number of top items to consider.
  """
  hist = utils.get_hist(input_data)
  sorted_hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)
  top_k_item_freq = [sorted_hist[i][1] for i in range(k)]
  output_item_freq = [hist[i] for i in output] + [0.0] * (k - len(output))
  l1_loss = np.linalg.norm(
      np.array(top_k_item_freq) - np.array(output_item_freq), ord=1
  )
  return l1_loss


def run_method_across_ks(input_data, method, k_range, params, num_trials=1):
  """Runs a method for the top-k selection problem on the given data with the given parameters across all k values.

  Args:
    input_data: A list where each inner list represents a user's items. For
      example, `[[1, 2, 3], [4, 5]]`.
    method: A `TopKMethod` enum to run.
    k_range: A list of k values to run the method on.
    params: A dictionary containing method-specific parameters. For
      `WGM_THEN_PEEL`, this includes `l0_bound`, `eps`, and
      `delta`. For `LIMITED_DOMAIN`, this includes `l0_bound`, `eps`,
      `delta`, and `k_bar`.
    num_trials: The number of trials to run for each k.

  Returns:
    A dictionary where the keys are k values and the values are dictionaries
    containing the mean and standard error of the metric values across all
    trials.
  """
  results_per_k_bound = {}

  for k in k_range:
    print("running " + _PLOT_LABELS[method] + " with k = " + str(k))

    trial_results = collections.defaultdict(list)

    for _ in range(num_trials):
      begin = time.time()
      l0_bound = params["l0_bound"]
      eps = params["eps"]
      delta = params["delta"]
      if method == TopKMethod.WGM_THEN_PEEL:
        eps_schedule = [eps/2, eps/2]
        delta_schedule = [delta/2, delta/2]
        output = _PARTIAL_METHODS[method](
            input_data,
            k,
            eps_schedule,
            delta_schedule,
            l0_bound,
            peel_users=False,
        )
      else:
        k_bar_multiplier = params["k_bar_multiplier"]
        output = _PARTIAL_METHODS[method](
            input_data, k, k * k_bar_multiplier, eps, delta, l0_bound
        )
      end = time.time()
      topk_missing_mass = compute_topk_missing_mass(input_data, output, k)
      topk_l1_loss = compute_topk_l1_loss(input_data, output, k)

      trial_results[Metric.TOPK_MM].append(topk_missing_mass)
      trial_results[Metric.TOPK_L1].append(topk_l1_loss)
      trial_results[Metric.RUNTIME].append(end - begin)

    results_per_k_bound[k] = {
        Metric.TOPK_MM: (
            np.mean(trial_results[Metric.TOPK_MM]),
            np.std(trial_results[Metric.TOPK_MM]) / np.sqrt(num_trials),
        ),
        Metric.TOPK_L1: (
            np.mean(trial_results[Metric.TOPK_L1]),
            np.std(trial_results[Metric.TOPK_L1]) / np.sqrt(num_trials),
        ),
        Metric.RUNTIME: (
            np.mean(trial_results[Metric.RUNTIME]),
            np.std(trial_results[Metric.RUNTIME]) / np.sqrt(num_trials),
        ),
    }

    print("finished " + _PLOT_LABELS[method] + " with k = " + str(k))
  return results_per_k_bound


def compare_methods(
    input_data,
    methods,
    k_range,
    epsilon,
    delta,
    l0_bound,
    num_trials,
    k_bar_multiplier_range=None,
):
  """Compares all methods with the given l0_bound, epsilon, and delta.

  Args:
    input_data: A list where each inner list represents a user's items. For
      example, `[[1, 2, 3], [4, 5]]`.
    methods: A list of `TopKMethod` enums to compare.
    k_range: A list of k values to try for each method.
    epsilon: The privacy parameter epsilon.
    delta: The privacy parameter delta.
    l0_bound: The l0_bound parameter.
    num_trials: The number of trials to run for each configuration.
    k_bar_multiplier_range: A list of k_bar_multiplier values to use for the
      LIMITED_DOMAIN method.

  Returns:
    Dictionary of results for each method.
  Raises:
    ValueError: If the parameters are not provided for the given methods.
  """
  results_per_method = {}
  params = {
      "l0_bound": l0_bound,
      "eps": epsilon,
      "delta": delta,
  }
  for method in methods:
    if method == TopKMethod.WGM_THEN_PEEL:
      results_per_method[method] = run_method_across_ks(
          input_data, method, k_range, params, num_trials
      )
    elif method == TopKMethod.LIMITED_DOMAIN:
      if k_bar_multiplier_range is None:
        raise ValueError(
            "k_bar_multiplier_range must be provided for LIMITED_DOMAIN."
        )
      results_per_method[method] = {}
      for k_bar_multiplier in k_bar_multiplier_range:
        params["k_bar_multiplier"] = k_bar_multiplier
        results_per_method[method][k_bar_multiplier] = run_method_across_ks(
            input_data, method, k_range, params, num_trials
        )
    else:
      raise ValueError(f"Unsupported method: {method}")

  return results_per_method


def plot_results(results, k_range, epsilon, delta, output_path):
  """Plots the results for all methods across all k values and metrics.

  The `results` dictionary is expected to be the output of `compare_methods`
  and has the following structure:
  {
      TopKMethod.WGM_THEN_PEEL: {
          k_1: {
              Metric.TOPK_MM: (mean, stderr),
              Metric.TOPK_L1: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          k_2: {
              Metric.TOPK_MM: (mean, stderr),
              Metric.TOPK_L1: (mean, stderr),
              Metric.RUNTIME: (mean, stderr)
          },
          ...
      },
      TopKMethod.LIMITED_DOMAIN: {
          k_bar_multiplier_1: {
              k_1: {
                  Metric.TOPK_MM: (mean, stderr),
                  Metric.TOPK_L1: (mean, stderr),
                  Metric.RUNTIME: (mean, stderr)
              },
              k_2: {
                  Metric.TOPK_MM: (mean, stderr),
                  Metric.TOPK_L1: (mean, stderr),
                  Metric.RUNTIME: (mean, stderr)
              },
              ...
          },
          k_bar_multiplier_2: {
              k_1: {
                  Metric.TOPK_MM: (mean, stderr),
                  Metric.TOPK_L1: (mean, stderr),
                  Metric.RUNTIME: (mean, stderr)
              },
              k_2: {
                  Metric.TOPK_MM: (mean, stderr),
                  Metric.TOPK_L1: (mean, stderr),
                  Metric.RUNTIME: (mean, stderr)
              },
              ...
          },
          ...
      },
      ...
  }
  The outer dictionary is keyed by method. The inner dictionary for the
  WGM_THEN_PEEL method is keyed by k value. The inner dictionary for the
  LIMITED_DOMAIN method is keyed by k_bar_multiplier value. The innermost
  dictionaries for both methods contain dictionaries of metric values with
  the mean and standard error across all trials.

  Args:
    results: A dictionary of results for each method.
    k_range: A list of k values used in the experiment.
    epsilon: The privacy parameter epsilon.
    delta: The privacy parameter delta.
    output_path: The base path to save the plot images.

  Raises:
    ValueError: If the results do not contain the given metric.
  """

  for metric in Metric:
    for method in results.keys():
      if method == TopKMethod.WGM_THEN_PEEL:
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
      elif method == TopKMethod.LIMITED_DOMAIN:
        for k_bar_multiplier in results[method].keys():
          result_lists = [results[method][k_bar_multiplier][k] for k in k_range]
          mean_metric_values = [result[metric][0] for result in result_lists]
          stderr_metric_values = [result[metric][1] for result in result_lists]
          plt.errorbar(
              k_range,
              mean_metric_values,
              yerr=stderr_metric_values,
              label=_PLOT_LABELS[method]
              + " (k_bar_multiplier = "
              + str(k_bar_multiplier)
              + ")",
              linewidth=3,
          )
      else:
        raise ValueError(f"Unsupported method: {method}")

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
