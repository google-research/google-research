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

"""Methods for running multiquantiles experiments and plotting the results."""

import enum
import functools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dp_multiq import base
from dp_multiq import csmooth
from dp_multiq import ind_exp
from dp_multiq import joint_exp
from dp_multiq import smooth
from dp_multiq import tree


class ErrorMetric(enum.Enum):
  MISCLASSIFIED_POINTS = 1
  DISTANCE = 2


_ERROR_FUNCS = {
    ErrorMetric.MISCLASSIFIED_POINTS:
        base.misclassified_points_error,
    ErrorMetric.DISTANCE:
        lambda _, true_qs, est_qs: base.distance_error(true_qs, est_qs)
}

_ERROR_LABELS = {
    ErrorMetric.MISCLASSIFIED_POINTS: "avg # misclassified points",
    ErrorMetric.DISTANCE: "avg distance"
}


class QuantilesEstimationMethod(enum.Enum):
  JOINT_EXP = 1
  IND_EXP = 2
  APP_IND_EXP = 3
  SMOOTH = 4
  CSMOOTH = 5
  LAP_TREE = 6
  GAUSS_TREE = 7


_PARTIAL_METHODS = {
    QuantilesEstimationMethod.JOINT_EXP: joint_exp.joint_exp,
    QuantilesEstimationMethod.IND_EXP: ind_exp.ind_exp,
    QuantilesEstimationMethod.APP_IND_EXP: ind_exp.ind_exp,
    QuantilesEstimationMethod.SMOOTH: smooth.smooth,
    QuantilesEstimationMethod.CSMOOTH: csmooth.csmooth,
    QuantilesEstimationMethod.LAP_TREE: tree.tree,
    QuantilesEstimationMethod.GAUSS_TREE: tree.tree
}

_PLOT_LABELS = {
    QuantilesEstimationMethod.JOINT_EXP: "JointExp",
    QuantilesEstimationMethod.IND_EXP: "IndExp",
    QuantilesEstimationMethod.APP_IND_EXP: "AppIndExp",
    QuantilesEstimationMethod.SMOOTH: "Smooth",
    QuantilesEstimationMethod.CSMOOTH: "CSmooth",
    QuantilesEstimationMethod.LAP_TREE: "LapTree",
    QuantilesEstimationMethod.GAUSS_TREE: "GaussTree"
}

_PLOT_LINESTYLES = {
    QuantilesEstimationMethod.JOINT_EXP: "-",
    QuantilesEstimationMethod.IND_EXP: "--",
    QuantilesEstimationMethod.APP_IND_EXP: "--",
    QuantilesEstimationMethod.SMOOTH: "-.",
    QuantilesEstimationMethod.CSMOOTH: "-.",
    QuantilesEstimationMethod.LAP_TREE: ":",
    QuantilesEstimationMethod.GAUSS_TREE: ":"
}

_PLOT_COLORS = {
    QuantilesEstimationMethod.JOINT_EXP: "lightseagreen",
    QuantilesEstimationMethod.IND_EXP: "mediumpurple",
    QuantilesEstimationMethod.APP_IND_EXP: "darkorange",
    QuantilesEstimationMethod.SMOOTH: "cornflowerblue",
    QuantilesEstimationMethod.CSMOOTH: "violet",
    QuantilesEstimationMethod.LAP_TREE: "firebrick",
    QuantilesEstimationMethod.GAUSS_TREE: "peru"
}


def synthetic_comparison(methods, error_func, data_type, num_samples, data_low,
                         data_high, num_trials, num_quantiles_range, eps, delta,
                         swap, ts_matrix):
  """Returns errors and times from running experients on synthetic data.

  Args:
    methods: Array of private quantiles algorithms to test.
    error_func: Function for computing quantile estimation error.
    data_type: Type of synthetic data to use, either uniform or gaussian.
    num_samples: Number of samples to use in each trial.
    data_low: Lower bound for data, used by private quantiles algorithms.
    data_high: Upper bound for data, used by private quantiles algorithms.
    num_trials: Number of trials to average over.
    num_quantiles_range: Array of numbers of quantiles to estimate.
    eps: Privacy parameter epsilon.
    delta: Privacy parameter delta, used only by smooth.
    swap: If true, uses swap privacy definition. Otherwise uses add-remove.
    ts_matrix: Matrix of smooth sensitivity parameters passed to CSmooth, where
      ts_matrix[i,j] corresponds to quantile j+1 of num_quantiles_range[i]
      quantiles.

  Returns:
    Arrays errors and times storing, respectively, average number of
    misclassified points and time in seconds for each of the five methods and
    each num_quantiles in num_quantiles_range, for the specified synthetic data.
  """
  max_num_quantiles = len(num_quantiles_range)
  num_methods = len(methods)
  errors = np.zeros((num_methods, max_num_quantiles))
  times = np.zeros((num_methods, max_num_quantiles))
  for num_quantiles_idx in range(max_num_quantiles):
    num_quantiles = num_quantiles_range[num_quantiles_idx]
    qs = np.linspace(0, 1, num_quantiles + 2)[1:-1]
    ts = ts_matrix[num_quantiles_idx]
    errors[:, num_quantiles_idx], times[:, num_quantiles_idx] = comparison(
        methods, error_func, np.empty(0), data_type, num_samples, data_low,
        data_high, num_trials, qs, eps, delta, swap, ts)
    print("Finished num_quantiles = " + str(num_quantiles))
  return errors, times


def real_comparison(methods, error_func, data_type, num_samples, data_low,
                    data_high, num_trials, num_quantiles_range, eps, delta,
                    swap, ts_matrix):
  """Returns errors and times from running experiments on real data.

  Args:
    methods: Array of private quantiles algorithms to test.
    error_func: Function for computing quantile estimation error.
    data_type: Type of real data to use, either ratings or pages.
    num_samples: Number of samples to use in each trial.
    data_low: Lower bound for data, used by private quantiles algorithms.
    data_high: Upper bound for data, used by private quantiles algorithms.
    num_trials: Number of trials to average over.
    num_quantiles_range: Array of number of quantiles to estimate.
    eps: Privacy parameter epsilon.
    delta: Privacy parameter delta, used only by Smooth.
    swap: If true, uses swap privacy definition. Otherwise uses add-remove.
    ts_matrix: Matrix of smooth sensitivity parameters passed to CSmooth, where
      ts_matrix[i,j] corresponds to quantile j+1 of num_quantiles_range[i]
      quantiles.

  Returns:
    Arrays errors and times storing, respectively, average number of
    misclassified points and time in seconds for each of the five methods and
    each num_quantiles in num_quantiles_range, for the specified real data.
  """
  max_num_quantiles = len(num_quantiles_range)
  num_methods = len(methods)
  errors = np.zeros((num_methods, max_num_quantiles))
  times = np.zeros((num_methods, max_num_quantiles))
  if data_type == "ratings":
    data = pd.read_csv("books.csv", usecols=["average_rating"])
    data = pd.to_numeric(data["average_rating"], errors="coerce").to_numpy()
    data = data[~np.isnan(data)]
  else:
    data = pd.read_csv("books.csv", usecols=["  num_pages"])
    data = pd.to_numeric(data["  num_pages"], errors="coerce").to_numpy()
    data = data[~np.isnan(data)]
    data = data / 100

  for num_quantiles_idx in range(max_num_quantiles):
    num_quantiles = num_quantiles_range[num_quantiles_idx]
    qs = np.linspace(0, 1, num_quantiles + 2)[1:-1]
    ts = ts_matrix[num_quantiles_idx]
    errors[:, num_quantiles_idx], times[:, num_quantiles_idx] = comparison(
        methods, error_func, data, "", num_samples, data_low, data_high,
        num_trials, qs, eps, delta, swap, ts)
    print("Finished num_quantiles = " + str(num_quantiles))
  return errors, times


def comparison(methods, error_func, fixed_data, distribution, num_samples,
               data_low, data_high, num_trials, qs, eps, delta, swap, ts):
  """Helper function to run the trials set up by synthetic/real_comparison.

  Args:
    methods: Array of private quantiles algorithms to test.
    error_func: Function for computing quantile estimation error.
    fixed_data: In the case of real data, an array of data to subsample in each
      trial. In the case of synthetic data, an empty array.
    distribution: In the case of real data, an empty string. In the case of
      synthetic data, either "gaussian" or "uniform".
    num_samples: Number of samples to use in each trial.
    data_low: Lower bound for data, used by private quantiles algorithms.
    data_high: Upper bound for data, used by private quantiles algorithms.
    num_trials: Number of trials to average over.
    qs: Array of quantiles to estimate.
    eps: Privacy parameter epsilon.
    delta: Privacy parameter delta, used only by Smooth.
    swap: If true, uses swap privacy definition. Otherwise uses add-remove.
    ts: Matrix of smooth sensitivity parameters passed to CSmooth.

  Returns:
    Arrays errors and times storing, respectively, average number of
    misclassified points and time in seconds for each of the methods.

  Throws:
    ValueError if the Smooth or CSmooth method is used in conjunction with
    swap=False, or if one of the specified methods is unrecognized.
  """
  # Create an array of DP quantile functions from the array of method names.
  quant_funcs = []
  for method in methods:
    quant_func = functools.partial(
        _PARTIAL_METHODS[method], data_low=data_low, data_high=data_high, qs=qs)

    if method == QuantilesEstimationMethod.JOINT_EXP:
      quant_func = functools.partial(quant_func, eps=eps, swap=swap)
    elif method == QuantilesEstimationMethod.IND_EXP:
      quant_func = functools.partial(
          quant_func, divided_eps=eps / len(qs), swap=swap)
    elif method == QuantilesEstimationMethod.APP_IND_EXP:
      quant_func = functools.partial(
          quant_func,
          divided_eps=ind_exp.opt_comp_calculator(eps, delta, len(qs)),
          swap=swap)
    elif method == QuantilesEstimationMethod.SMOOTH:
      if not swap:
        raise ValueError("Smooth method is only implemented for swap DP.")
      quant_func = functools.partial(
          quant_func, divided_eps=eps / len(qs), divided_delta=delta / len(qs))
    elif method == QuantilesEstimationMethod.CSMOOTH:
      if not swap:
        raise ValueError("CSmooth method is only implemented for swap DP.")
      quant_func = functools.partial(
          quant_func, divided_eps=eps / np.sqrt(len(qs)), ts=ts)
    elif method == QuantilesEstimationMethod.LAP_TREE:
      quant_func = functools.partial(quant_func, eps=eps, delta=0, swap=swap)
    elif method == QuantilesEstimationMethod.GAUSS_TREE:
      quant_func = functools.partial(
          quant_func, eps=eps, delta=delta, swap=swap)
    else:
      raise ValueError("Unrecognized method name: {}".format(method))
    quant_funcs.append(quant_func)

  num_methods = len(methods)
  if len(quant_funcs) != num_methods:
    raise ValueError(
        "Quantile functions array length does not match methods array length.")

  errors = np.zeros(num_methods)
  times = np.zeros(num_methods)
  for _ in range(num_trials):
    # Sample a dataset.
    if fixed_data.size > 0:
      sampled_data = np.sort(
          np.random.choice(fixed_data, num_samples, replace=False))
    elif distribution == "gaussian":
      sampled_data = base.gen_gaussian(num_samples, 0, 5)
    elif distribution == "uniform":
      sampled_data = base.gen_uniform(num_samples, -5, 5)
    true_quantiles = base.quantiles(sampled_data, qs)

    for method_num in range(num_methods):
      quant_func = quant_funcs[method_num]
      begin = time.time()
      estimates = quant_func(sampled_data)
      end = time.time()
      times[method_num] = (end - begin) / num_trials
      errors[method_num] += error_func(sampled_data, true_quantiles,
                                       estimates) / num_trials

  return errors, times


def tune_ts_plot(eps, avg_ts, num_quantiles_range, file_name):
  """Shows the specified plot of tuned t parameters for CSmooth.

  Args:
    eps: Privacy parameter epsilon.
    avg_ts: Array of arrays of selected ts, one array for each number of number
      of quantiles.
    num_quantiles_range: Array of number of quantiles used to tune.
    file_name: File name for saving plot.

  Returns:
    Saves the specified plot as file_name.png.
  """
  for num_quantiles_idx in range(len(num_quantiles_range)):
    num_quantiles = num_quantiles_range[num_quantiles_idx]
    plt.scatter(
        np.linspace(0, 1, num_quantiles + 2)[1:-1],
        avg_ts[num_quantiles_idx],
        label=str(num_quantiles))
  plt.title("tuned t per quantile range, eps = " + str(eps))
  plt.ylabel("tuned t")
  plt.xlabel("quantile")
  plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
  plt.savefig(file_name + ".png")
  plt.close()


def plot(methods, y_label, title, x_array, y_arrays, log_scale, legend,
         plot_name):
  """Constructs and saves the specified plot as plot_name.png.

  Args:
    methods: Array of private quantiles algorithms to test.
    y_label: Label for plot's y-axis.
    title: Title to display at the top of the plot.
    x_array: Array of quantiles to use for the x-axis.
    y_arrays: len(methods) x len(x_array) array of points to plot.
    log_scale: If true, scales y-axis logarithmically.
    legend: If true, displays legend.
    plot_name: File name to use for saving plot.
  Throws: ValueError if methods and y_arrays size does not match methods and
    x_array.
  """
  num_methods = len(methods)
  if num_methods != y_arrays.shape[0]:
    raise ValueError(
        "Length of methods does not match first dimension of y_arrays.")
  if len(x_array) != y_arrays.shape[1]:
    raise ValueError(
        "Length of x_array does not match second dimension of y_arrays.")

  for index in range(num_methods):
    y_array = y_arrays[index]
    method = methods[index]
    plt.plot(
        x_array,
        y_array,
        linestyle=_PLOT_LINESTYLES[method],
        label=_PLOT_LABELS[method],
        color=_PLOT_COLORS[method],
        linewidth=3)
  plt.title(title, fontsize=18)
  plt.ylabel(y_label, fontsize=18)
  if log_scale:
    plt.yscale("log")
  plt.xlabel("# quantiles", fontsize=18)
  if legend:
    legend = plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=3,
        frameon=False,
        fontsize=16)
    plt.savefig(
        plot_name + ".png", bbox_extra_artists=(legend,), bbox_inches="tight")
  else:
    plt.savefig(plot_name + ".png", bbox_inches="tight")
  plt.close()


def experiment(methods,
               error_metric=ErrorMetric.MISCLASSIFIED_POINTS,
               data_low=-100,
               data_high=100,
               num_samples=1000,
               eps=1,
               delta=1e-6,
               swap=True,
               num_quantiles_range=range(1, 30),
               est_num_trials=10,
               ts_num_trials=2,
               ts_plot_name="eps_1_ts",
               error_plot_prefix="eps_1_error",
               time_plot_prefix="eps_1_time"):
  """Runs trials and saves relevant plots for the specified experiment.

  Args:
    methods: Array of private quantiles algorithms to test.  Available methods
      are defined in the QuantilesEstimationMethod enum.
    error_metric: Available metrics are defined in the ErrorMetric enum.
    data_low: Lower bound for data, used by private quantiles algorithms.
    data_high: Upper bound for data, used by private quantiles algorithms.
    num_samples: Number of samples for each trial.
    eps: Privacy parameter epsilon.
    delta: Privacy parameter delta.
    swap: If true, uses swap privacy definition. Otherwise uses add-remove.
    num_quantiles_range: Array of numbers of quantiles to estimate.
    est_num_trials: Number of trials to average for error and time experiments.
    ts_num_trials: Number of trials to average for tuning ts experiments.
    ts_plot_name: Name for saving the tuning ts plot.
    error_plot_prefix: File prefix for the error plots. For example, the
      Gaussian error plot will have name error_plot_prefix_gaussian.
    time_plot_prefix: File prefix for the time plots.

  Returns:
    Saves the generated ts/error/time plots.
  """
  if QuantilesEstimationMethod.CSMOOTH in methods:
    tuned_ts = csmooth.csmooth_tune_t_experiment(eps, num_samples,
                                                 ts_num_trials,
                                                 num_quantiles_range, data_low,
                                                 data_high, -2, 0, 50)
    tune_ts_plot(eps, tuned_ts, num_quantiles_range, ts_plot_name)
    print("Finished tuning t for CSmooth.")
  else:
    tuned_ts = [
        np.empty(num_quantiles) for num_quantiles in num_quantiles_range
    ]

  error_func = _ERROR_FUNCS[error_metric]
  error_label = _ERROR_LABELS[error_metric]

  for data in ["gaussian", "uniform"]:
    errors, times = synthetic_comparison(methods, error_func, data, num_samples,
                                         data_low, data_high, est_num_trials,
                                         num_quantiles_range, eps, delta, swap,
                                         tuned_ts)
    plot(methods, error_label, data + " error", num_quantiles_range, errors,
         True, True, error_plot_prefix + data)
    plot(methods, "time (secs)", data + " time", num_quantiles_range, times,
         True, True, time_plot_prefix + data)
    print("Finished " + data + " trials.")

  for data in ["ratings", "pages"]:
    errors, times = real_comparison(methods, error_func, data, num_samples,
                                    data_low, data_high, est_num_trials,
                                    num_quantiles_range, eps, delta, swap,
                                    tuned_ts)
    plot(methods, error_label, data + " error", num_quantiles_range, errors,
         True, True, error_plot_prefix + data)
    plot(methods, "time (secs)", data + " time", num_quantiles_range, times,
         True, True, time_plot_prefix + data)
    print("Finished " + data + " trials.")
