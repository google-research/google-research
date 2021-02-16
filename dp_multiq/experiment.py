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

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dp_multiq import base
from dp_multiq import csmooth
from dp_multiq import ind_exp
from dp_multiq import joint_exp
from dp_multiq import smooth


def synthetic_comparison(data_type, num_samples, data_low, data_high,
                         num_trials, num_quantiles_range, eps, delta, swap,
                         ts_matrix):
  """Returns errors and times from running experients on synthetic data.

  Args:
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
  errors = np.zeros((5, max_num_quantiles))
  times = np.zeros((5, max_num_quantiles))
  for num_quantiles_idx in range(max_num_quantiles):
    num_quantiles = num_quantiles_range[num_quantiles_idx]
    qs = np.linspace(0, 1, num_quantiles + 2)[1:-1]
    ts = ts_matrix[num_quantiles_idx]
    errors[:, num_quantiles_idx], times[:, num_quantiles_idx] = comparison(
        np.empty(0), data_type, num_samples, data_low, data_high, num_trials,
        qs, eps, delta, swap, ts)
    print("Finished num_quantiles = " + str(num_quantiles))
  return errors, times


def real_comparison(data_type, num_samples, data_low, data_high,
                    num_trials, num_quantiles_range, eps, delta, swap,
                    ts_matrix):
  """Returns errors and times from running experiments on real data.

  Args:
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
  errors = np.zeros((5, max_num_quantiles))
  times = np.zeros((5, max_num_quantiles))
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
        data, "", num_samples, data_low, data_high, num_trials, qs, eps, delta,
        swap, ts)
    print("Finished num_quantiles = " + str(num_quantiles))
  return errors, times


def comparison(fixed_data, distribution, num_samples, data_low, data_high,
               num_trials, qs, eps, delta, swap, ts):
  """Helper function to run the trials set up by synthetic/real_comparison.

  Args:
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
    misclassified points and time in seconds for each of the five methods.
  """
  errors = np.zeros(5)
  times = np.zeros(5)
  for _ in range(num_trials):
    if fixed_data.size > 0:
      sampled_data = np.sort(
          np.random.choice(fixed_data, num_samples, replace=False))
    elif distribution == "gaussian":
      sampled_data = base.gen_gaussian(num_samples, 0, 5)
    elif distribution == "uniform":
      sampled_data = base.gen_uniform(num_samples, -5, 5)
    true_quantiles = base.quantiles(sampled_data, qs)
    begin = time.time()
    joint_exp_quantiles = joint_exp.joint_exp(sampled_data, data_low, data_high,
                                              qs, eps, swap)
    end = time.time()
    errors[0] += base.quantiles_error(sampled_data, qs, true_quantiles,
                                      joint_exp_quantiles) / num_trials
    times[0] += (end - begin) / num_trials
    begin = time.time()
    ind_exp_quantiles = ind_exp.ind_exp(sampled_data, data_low, data_high, qs,
                                        swap, eps / len(qs))
    end = time.time()
    errors[1] += base.quantiles_error(sampled_data, qs, true_quantiles,
                                      ind_exp_quantiles) / num_trials
    times[1] += (end - begin) / num_trials
    app_ind_exp_eps = ind_exp.opt_comp_calculator(eps, delta, len(qs))
    begin = time.time()
    app_ind_exp_quantiles = ind_exp.ind_exp(sampled_data, data_low, data_high,
                                            qs, swap, app_ind_exp_eps)
    end = time.time()
    errors[2] += base.quantiles_error(sampled_data, qs, true_quantiles,
                                      app_ind_exp_quantiles) / num_trials
    times[2] += (end - begin) / num_trials
    begin = time.time()
    smooth_quantiles = smooth.smooth(sampled_data, data_low, data_high, qs,
                                     eps / len(qs), delta / len(qs))
    end = time.time()
    errors[3] += base.quantiles_error(sampled_data, qs, true_quantiles,
                                      smooth_quantiles) / num_trials
    times[3] += (end - begin) / num_trials
    begin = time.time()
    csmooth_quantiles = csmooth.csmooth(sampled_data, data_low, data_high, qs,
                                        eps / np.sqrt(len(qs)), ts)
    end = time.time()
    errors[4] += base.quantiles_error(sampled_data, qs, true_quantiles,
                                      csmooth_quantiles) / num_trials
    times[4] += (end - begin) / num_trials
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


def plot(plot_type, title, x_array, y_arrays, log_scale, legend, plot_name):
  """Constructs and saves the specified plot as plot_name.png.

  Args:
    plot_type: Specifies whether plot shows error or time.
    title: Title to display at the top of the plot.
    x_array: Array of quantiles to use for the x-axis.
    y_arrays: 5 x (len(x_array)) array of points to plot, one for each method.
    log_scale: If true, scales y-axis logarithmically.
    legend: If true, displays legend.
    plot_name: File name to use for saving plot.
  """
  algorithms = ["JointExp", "IndExp", "AppIndExp", "Smooth", "CSmooth"]
  linestyles = ["-", "--", "--", "-.", "-."]
  colors = [
      "lightseagreen", "mediumpurple", "darkorange", "cornflowerblue", "violet"
  ]
  for index in range(5):
    y_array = y_arrays[index]
    linestyle = linestyles[index]
    label = algorithms[index]
    color = colors[index]
    plt.plot(
        x_array,
        y_array,
        linestyle=linestyle,
        label=label,
        color=color,
        linewidth=3)
  plt.title(title, fontsize=18)
  if plot_type == "error":
    plt.ylabel("# misclassified points")
  else:
    plt.ylabel("time (s)", fontsize=18)
  if log_scale:
    plt.yscale("log")
  plt.xlabel("# quantiles", fontsize=18)
  if legend:
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.47),
        ncol=3,
        frameon=False,
        fontsize=16)
  plt.savefig(plot_name + ".png")
  plt.close()


def experiment(data_low=-100,
               data_high=100,
               num_samples=500,
               eps=1,
               delta=1e-6,
               swap=True,
               num_quantiles_range=range(1, 10),
               est_num_trials=5,
               ts_num_trials=2,
               ts_plot_name="eps_1_ts",
               error_plot_prefix="eps_1_error",
               time_plot_prefix="eps_1_time"):
  """Runs trials and saves relevant plots for the specified experiment.

  Args:
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
  tuned_ts = csmooth.csmooth_tune_t_experiment(eps, num_samples, ts_num_trials,
                                               num_quantiles_range, data_low,
                                               data_high, -2, 0, 50)
  tune_ts_plot(eps, tuned_ts, num_quantiles_range, ts_plot_name)
  print("Finished tuning t.")
  for data in ["gaussian", "uniform"]:
    errors, times = synthetic_comparison(data, num_samples, data_low, data_high,
                                         est_num_trials, num_quantiles_range,
                                         eps, delta, swap, tuned_ts)
    plot("error", data + " error, eps = " + str(eps), num_quantiles_range,
         errors, True, True, error_plot_prefix + data)
    plot("time", data + " time, eps = " + str(eps), num_quantiles_range, times,
         True, True, time_plot_prefix + data)
    print("Finished " + data + " trials.")
  for data in ["ratings", "pages"]:
    errors, times = real_comparison(data, num_samples, data_low,
                                    data_high, est_num_trials,
                                    num_quantiles_range, eps, delta, swap,
                                    tuned_ts)
    plot("error", data + " error, eps = " + str(eps), num_quantiles_range,
         errors, True, True, error_plot_prefix + data)
    plot("time", data + " time, eps = " + str(eps), num_quantiles_range, times,
         True, True, time_plot_prefix + data)
    print("Finished " + data + " trials.")
