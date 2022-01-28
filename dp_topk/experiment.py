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

"""Differentially private top-k experiment and plotting code."""

import enum
import functools
import time

import matplotlib.pyplot as plt
import numpy as np

from dp_topk import baseline_mechanisms
from dp_topk import joint


class TopKEstimationMethod(enum.Enum):
  JOINT = 1
  CDP_PEEL = 2
  PNF_PEEL = 3
  LAP = 4
  GAMMA = 5


_PARTIAL_METHODS = {
    TopKEstimationMethod.JOINT: joint.joint,
    TopKEstimationMethod.CDP_PEEL: baseline_mechanisms.cdp_peeling_mechanism,
    TopKEstimationMethod.PNF_PEEL: baseline_mechanisms.pnf_peeling_mechanism,
    TopKEstimationMethod.LAP: baseline_mechanisms.laplace_mechanism,
    TopKEstimationMethod.GAMMA: baseline_mechanisms.gamma_mechanism
}

_PLOT_LABELS = {
    TopKEstimationMethod.JOINT: "joint",
    TopKEstimationMethod.CDP_PEEL: "cdp peel",
    TopKEstimationMethod.PNF_PEEL: "pnf peel",
    TopKEstimationMethod.LAP: "laplace",
    TopKEstimationMethod.GAMMA: "gamma"
}

_PLOT_LINESTYLES = {
    TopKEstimationMethod.JOINT: "-",
    TopKEstimationMethod.CDP_PEEL: "--",
    TopKEstimationMethod.PNF_PEEL: ":",
    TopKEstimationMethod.LAP: "-.",
    TopKEstimationMethod.GAMMA: "-."
}

_PLOT_COLORS = {
    TopKEstimationMethod.JOINT: "palevioletred",
    TopKEstimationMethod.CDP_PEEL: "deepskyblue",
    TopKEstimationMethod.PNF_PEEL: "rebeccapurple",
    TopKEstimationMethod.LAP: "darkgreen",
    TopKEstimationMethod.GAMMA: "darkorange",
}

_PLOT_FILL_COLORS = {
    TopKEstimationMethod.JOINT: "lightpink",
    TopKEstimationMethod.CDP_PEEL: "powderblue",
    TopKEstimationMethod.PNF_PEEL: "mediumpurple",
    TopKEstimationMethod.LAP: "mediumseagreen",
    TopKEstimationMethod.GAMMA: "bisque"
}


def linf_error(true_top_k, est_top_k):
  """Computes l_inf distance between the true and estimated top k counts.

  Args:
    true_top_k: Nonincreasing sequence of counts of true top k items.
    est_top_k: Sequence of counts of estimated top k items.

  Returns:
    l_inf distance between true_top_k and sequence.
  """
  return np.linalg.norm(true_top_k - est_top_k, ord=np.inf)


def l1_error(true_top_k, est_top_k):
  """Computes l_1 distance between the true and estimated top k counts.

  Args:
    true_top_k: Nonincreasing sequence of counts of true top k items.
    est_top_k: Sequence of counts of estimated top k items.

  Returns:
    l_1 distance between true_top_k and sequence.
  """
  return np.linalg.norm(true_top_k - est_top_k, ord=1)


def k_relative_error(true_top_k, est_top_k):
  """Computes k-relative error between the true and estimated top k counts.

  Args:
    true_top_k: Nonincreasing sequence of counts of true top k items.
    est_top_k: Sequence of counts of estimated top k items.

  Returns:
    max_{i in [k]} (c_k - c'_i), where c_1, ..., c_k are the true top k counts
    and c'_1, ..., c'_k are the estimated top k counts.
  """
  return np.amax(true_top_k[-1] - est_top_k)


def linf_error_idx(true_top_k, est_top_k):
  """Computes the index with the maximum error between true and estimated top k.

  Args:
    true_top_k: Nonincreasing sequence of counts of true top k items.
    est_top_k: Sequence of counts of estimated top k items.

  Returns:
    Index i such that |c_i - c'_i| = ||c_{:k} - c'_{:k}||_infty.
  """
  return np.argmax(true_top_k - est_top_k)


class ErrorMetric(enum.Enum):
  L_INF = 1
  L_1 = 2
  K_REL = 3
  L_INF_IDX = 4


_ERROR_FUNCS = {
    ErrorMetric.L_INF: linf_error,
    ErrorMetric.L_1: l1_error,
    ErrorMetric.K_REL: k_relative_error,
    ErrorMetric.L_INF_IDX: linf_error_idx
}

_ERROR_LABELS = {
    ErrorMetric.L_INF: "$\\ell_\\infty$ error",
    ErrorMetric.L_1: "$\\ell_1$ error",
    ErrorMetric.K_REL: "$k$-relative error",
    ErrorMetric.L_INF_IDX: "$\\ell_\\infty$ error index"
}


def compare(item_counts, methods, d, k_range, epsilon, delta, num_trials,
            neighbor_type):
  """Computes 25th, 50th, and 75th percentile errors and times for each method.

  Args:
    item_counts: Array of item counts.
    methods: Available top-k estimation methods are defined in the
      TopKEstimationMethod enum.
    d: The number of counts subsampled uniformly at random from item_counts, or
      -1 to sample all counts.
    k_range: Range for k, the number of top items to estimate. For example,
      k_range = [5,10,15] will run trials estimating the top 5, 10, and 15 items
      by count.
    epsilon: Overall privacy parameter epsilon.
    delta: Overall privacy parameter delta (only used for CDP peeling
      mechanism).
    num_trials: Number of trials to run for each k in k_range.
    neighbor_type: Available neighbor types are defined in the NeighborType
      enum.

  Returns:
    Dictionary results where results["time (s)"] is a
    (# methods) x (# ks in k_range) x 3 array storing 0.25, 0.5, and 0.75
    quantile times, and for error_label in the _ERROR_LABELS enum,
    results[error_label] is a (# methods) x (# ks in k_range) x 3 array storing
    0.25, 0.5, and 0.75 quantile errors for the corresponding error metric.

  Raises:
    ValueError: Unrecognized method name: [method].
  """
  num_ks = len(k_range)
  num_methods = len(methods)
  quantiles = [0.25, 0.5, 0.75]
  num_quantiles = 3
  errors = np.empty((len(ErrorMetric), num_methods, num_ks, num_quantiles))
  times = np.empty((num_methods, num_ks, num_quantiles))
  if d == -1:
    d = len(item_counts)
  item_counts_generator = lambda: np.random.permutation(item_counts)[:d]
  method_fns = []
  for method in methods:
    method_fn = functools.partial(_PARTIAL_METHODS[method], epsilon=epsilon)
    if method == TopKEstimationMethod.JOINT:
      method_fn = functools.partial(method_fn, neighbor_type=neighbor_type)
    elif method == TopKEstimationMethod.CDP_PEEL:
      method_fn = functools.partial(method_fn, delta=delta)
    elif method == TopKEstimationMethod.LAP:
      method_fn = functools.partial(method_fn, c=d, neighbor_type=neighbor_type)
    elif method != TopKEstimationMethod.PNF_PEEL and method != TopKEstimationMethod.GAMMA:
      raise ValueError("Unrecognized method name: {}".format(method))
    method_fns.append(method_fn)
  for k_idx in range(num_ks):
    k = k_range[k_idx]
    print("running k: " + str(k))
    k_errors = np.empty((len(ErrorMetric), num_methods, num_trials))
    k_times = np.empty((num_methods, num_trials))
    for trial in range(num_trials):
      item_counts = item_counts_generator()
      true_top_k = np.sort(item_counts)[::-1][:k]
      for method_idx in range(num_methods):
        start = time.time()
        selected_items = method_fns[method_idx](item_counts=item_counts, k=k)
        end = time.time()
        k_times[method_idx][trial] = end - start
        for metric in ErrorMetric:
          k_errors[metric.value - 1][method_idx][trial] = _ERROR_FUNCS[metric](
              true_top_k, item_counts[selected_items])
    for method_idx in range(num_methods):
      times[method_idx][k_idx] = np.quantile(k_times[method_idx], quantiles)
      for metric in ErrorMetric:
        errors[metric.value - 1][method_idx][k_idx] = np.quantile(
            k_errors[metric.value - 1][method_idx], quantiles)
  results = {}
  results["time (s)"] = times
  for metric in ErrorMetric:
    results[_ERROR_LABELS[metric]] = errors[metric.value - 1]
  return results


def plot(data_source, methods, results, k_range, log_y_axis, legend):
  """Plots errors and times data generated by compare and saves plots as .png.

  Args:
    data_source: Data source used to generate input results.
    methods: Top-k estimation methods used to generate input results. Available
      top-k estimation methods are defined in the TopKEstimationMethod enum.
    results: Dictionary of error and time data generated by compare.
    k_range: Range for k, the number of top items estimated.
    log_y_axis: Boolean determining whether plot y-axis is logarithmic.
    legend: Boolean determining whether the legend appears.

  Returns:
    An error plot for each error metric and one time plot. Each error plot is
    saved as $data_source_error_metric.png where error_metric is defined in
    ErrorMetric.name, and the time plot is saved as $data_source_time.png.
  """
  for metric in ErrorMetric:
    plt.xlabel("k", fontsize=20)
    ax = plt.gca()
    ax.tick_params(labelsize=18)
    if log_y_axis:
      plt.yscale("log")
      plt.title(data_source + " " + _ERROR_LABELS[metric], fontsize=20)
    for method_idx in range(len(methods)):
      method = methods[method_idx]
      plt.plot(
          k_range,
          results[_ERROR_LABELS[metric]][method_idx, :, 1] + 1,
          linestyle=_PLOT_LINESTYLES[method],
          label=_PLOT_LABELS[method],
          color=_PLOT_COLORS[method],
          linewidth=3)
      plt.fill_between(
          k_range,
          results[_ERROR_LABELS[metric]][method_idx, :, 0] + 1,
          results[_ERROR_LABELS[metric]][method_idx, :, 2] + 1,
          color=_PLOT_FILL_COLORS[method],
          alpha=0.5)
    if legend:
      ax.legend(
          loc="lower center",
          bbox_to_anchor=(0.45, -0.4),
          ncol=3,
          frameon=False,
          fontsize=16)
    plt.ylabel(_ERROR_LABELS[metric], fontsize=20)
    plt.savefig(
        data_source + "_" + str(metric.name) + ".png", bbox_inches="tight")
    plt.close()
  plt.xlabel("k", fontsize=20)
  ax = plt.gca()
  ax.tick_params(labelsize=18)
  if log_y_axis:
    plt.yscale("log")
    plt.title(data_source + " " + _ERROR_LABELS[metric], fontsize=20)
  for method_idx in range(len(methods)):
    method = methods[method_idx]
    plt.plot(
        k_range,
        results["time (s)"][method_idx, :, 1],
        linestyle=_PLOT_LINESTYLES[method],
        label=_PLOT_LABELS[method],
        color=_PLOT_COLORS[method],
        linewidth=3)
    plt.fill_between(
        k_range,
        results["time (s)"][method_idx, :, 0],
        results["time (s)"][method_idx, :, 2],
        color=_PLOT_FILL_COLORS[method],
        alpha=0.5)
    plt.ylabel(_ERROR_LABELS[metric], fontsize=20)
    plt.title(data_source + " time", fontsize=20)
  if legend:
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.45, -0.4),
        ncol=3,
        frameon=False,
        fontsize=16)
  plt.ylabel("time (s)", fontsize=20)
  plt.savefig(data_source + "_time.png", bbox_inches="tight")
  plt.close()


def counts_histogram(item_counts, plot_title, plot_name):
  """Computes and plots histogram of item counts.

  Args:
    item_counts: Array of item counts.
    plot_title: Plot title.
    plot_name: Plot will be saved as plot_name.png.

  Returns:
    Histogram of item counts using 100 bins.
  """
  plt.title(plot_title, fontsize=20)
  plt.xlabel("item count", fontsize=20)
  plt.ylabel("# items", fontsize=20)
  plt.yscale("log")
  ax = plt.gca()
  ax.tick_params(labelsize=18)
  plt.hist(item_counts, bins=100)
  plt.savefig(plot_name + ".png")
  plt.close()


def compute_and_plot_diffs(item_counts, d, k_range, num_trials, log_y_axis,
                           plot_title, plot_name):
  """Computes and plots median diffs between top k counts.

  Args:
    item_counts: Array of item counts.
    d: Total number of items to subsample from data in each trial.
    k_range: Range for k, the number of top items estimated.
    num_trials: Number of trials to average over.
    log_y_axis: Boolean determining whether plot y-axis is logarithmic.
    plot_title: Title displayed on the plot.
    plot_name: Plot will be saved as plot_name.png.

  Returns:
    Plot of median diff between k^{th} and (k+1}^{th} sorted item count
    for each k in k_range, where each trial subsamples min(data size, d) counts.
  """
  diffs = np.zeros((num_trials, len(k_range)))
  for trial in range(num_trials):
    sample = np.sort(np.random.permutation(item_counts)[:d])[::-1]
    trial_diffs = sample[:-1] - sample[1:]
    diffs[trial] = trial_diffs[k_range]
  median_diffs = np.quantile(diffs, q=0.5, axis=0)
  if log_y_axis:
    plt.yscale("log")
  plt.xlabel("k", fontsize=20)
  plt.ylabel("count diff", fontsize=20)
  ax = plt.gca()
  ax.tick_params(labelsize=18)
  plt.title(plot_title, fontsize=20)
  plt.plot(k_range, 1 + median_diffs, linewidth=3)
  plt.savefig(plot_name + ".png")
  plt.close()
