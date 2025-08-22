# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Experiment running code for differentially private linear regression."""

import enum
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from dp_regression import baselines
from dp_regression import tukey


class RegressionMethod(enum.Enum):
  NONDP = 1
  ADASSP = 2
  DPSGD_OPT = 3
  DPSGD_SUBOPT = 4
  TUKEY = 5


_LABELS = {
    RegressionMethod.NONDP: "Non-DP",
    RegressionMethod.ADASSP: "AdaSSP",
    RegressionMethod.DPSGD_OPT: "DPSGD-Opt",
    RegressionMethod.DPSGD_SUBOPT: "DPSGD-Subopt",
    RegressionMethod.TUKEY: "Tukey"
}

_LINESTYLES = {
    RegressionMethod.NONDP: "-.",
    RegressionMethod.ADASSP: ":",
    RegressionMethod.DPSGD_OPT: "--",
    RegressionMethod.DPSGD_SUBOPT: "--",
    RegressionMethod.TUKEY: "-",
}

_COLORS = {
    RegressionMethod.NONDP: "royalblue",
    RegressionMethod.ADASSP: "palevioletred",
    RegressionMethod.DPSGD_OPT: "chocolate",
    RegressionMethod.DPSGD_SUBOPT: "purple",
    RegressionMethod.TUKEY: "seagreen"
}




def r_squared(predictions, labels):
  """Returns R^2 values for given predictions on labels.

  Args:
    predictions: Matrix where each row consists of predictions from one model.
    labels: Vector of labels.

  Returns:
    Vector of length len(predictions) containing an R^2 value for each model.
  """
  sum_squared_residuals = np.sum(np.square(predictions - labels), axis=1)
  total_sum_squares = np.sum(np.square(labels - np.mean(labels)))
  return 1 - np.divide(sum_squared_residuals, total_sum_squares)


def r_squared_from_models(models, features, labels):
  """Returns 0.25, 0.5, and 0.75 quantiles of R^2 values for given models.

  Args:
    models: Matrix where each row consists of a model.
    features: Matrix where each row consists of one data point.
    labels: Column vector of labels.
  """
  predictions = np.matmul(models, features.T)
  r2_vals = r_squared(predictions, labels)
  return np.quantile(
      r2_vals, 0.25, axis=0), np.quantile(
          r2_vals, 0.5, axis=0), np.quantile(
              r2_vals, 0.75, axis=0)


def run_nondp(features, labels):
  """Returns 0.25, 0.5, and 0.75 R^2 quantiles from num_trials non-DP models.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.

  Returns:
    0.25, 0.5, and 0.75 R^2 quantiles from num_trials non-DP models. These are
    identical here since the non-DP algorithm is deterministic.
  """
  model = baselines.nondp(features, labels)
  return r_squared_from_models(np.tile(model.T, (1, 1)), features, labels)


def run_adassp(features, labels, epsilon, delta, num_trials):
  """Returns 0.25, 0.5, and 0.75 quantiles from num_trials AdaSSP models.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Each DP model satisfies (epsilon, delta)-DP.
    delta: Each DP model satisfies (epsilon, delta)-DP.
    num_trials: Number of trials to run.
  """
  models = np.zeros((num_trials, len(features[0])))
  for trial in range(num_trials):
    models[trial, :] = baselines.adassp(features, labels, epsilon, delta)
  return r_squared_from_models(models, features, labels)


def run_dpsgd(features, labels, params, num_trials):
  """Returns 0.25, 0.5, and 0.75 quantiles from num_trials DPSGD models.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    params: Dictionary of parameters (num_epochs, clip_norm, learning_rate,
      noise_multiplier) used by DPSGD.
    num_trials: Number of trials to run.
  """
  models = np.zeros((num_trials, len(features[0])))
  for trial in range(num_trials):
    models[trial, :] = baselines.dpsgd(features, labels, params)
  return r_squared_from_models(models, features, labels)


def run_dp_tukey(features, labels, epsilon, delta, m, num_trials):
  """Returns 0.25, 0.5, and 0.75 R^2 quantiles from num_trials Tukey models.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Each DP model satisfies (epsilon, delta)-DP.
    delta: Each DP model satisfies (epsilon, delta)-DP.
    m: Number of models used by Tukey mechanism.
    num_trials: Number of trials to run.
  """
  _, d = features.shape
  final_models = np.zeros((num_trials, d))
  for trial in range(num_trials):
    models = tukey.multiple_regressions(features, labels, m)
    final_models[trial, :] = tukey.dp_tukey(models, epsilon, delta)
  return r_squared_from_models(final_models, features, labels)


def run_trials(features, labels, epsilon, delta, m_range, dpsgd_opt_params,
               dpsgd_subopt_params, num_trials):
  """Runs trials computing R^2 quantiles for each method in RegressionMethod.

  Args:
    features: Matrix where each row is a vector of features. Assumed to already
      have intercept feature.
    labels: Vector of labels.
    epsilon: Each DP model satisfies (epsilon, delta)-DP.
    delta: Each DP model satisfies (epsilon, delta)-DP.
    m_range: Range for number of models used by Tukey mechanism.
    dpsgd_opt_params: Dictionary of parameters (num_epochs, clip_norm,
      learning_rate, noise_multiplier)used by DPSGD_OPT.
    dpsgd_subopt_params: Dictionary of parameters (num_epochs, clip_norm,
      learning_rate, noise_multiplier)used by DPSGD_SUBOPT.
    num_trials: Number of trials to run for each method.

  Returns:
    Dictionaries containing, respectively, R^2 0.25, 0.5, 0.75 quantiles,
    and median time for each method in RegressionMethod, across trials.

  Raises:
    RuntimeError: [m] models requires
    [m * (len(features[0]) + 1)] points, but given features only has
    [len(features)] points.
  """
  (n, d) = features.shape
  num_m = len(m_range)
  r2_25s = {}
  r2_50s = {}
  r2_75s = {}
  times = {}
  for method in RegressionMethod:
    if method != RegressionMethod.TUKEY:
      if method == RegressionMethod.NONDP:
        begin = time.time()
        r2_25, r2_50, r2_75, = run_nondp(features, labels)
        end = time.time()
        print("finished nondp")
      elif method == RegressionMethod.ADASSP:
        begin = time.time()
        r2_25, r2_50, r2_75, = run_adassp(features, labels, epsilon, delta,
                                          num_trials)
        end = time.time()
        print("finished adassp")
      elif method == RegressionMethod.DPSGD_OPT:
        begin = time.time()
        r2_25, r2_50, r2_75, = run_dpsgd(features, labels, dpsgd_opt_params,
                                         num_trials)
        end = time.time()
        print("finished dpsgd opt")
      elif method == RegressionMethod.DPSGD_SUBOPT:
        begin = time.time()
        r2_25, r2_50, r2_75, = run_dpsgd(features, labels, dpsgd_subopt_params,
                                         num_trials)
        end = time.time()
        print("finished dpsgd subopt")
      r2_25s[_LABELS[method]] = np.repeat(r2_25, num_m)
      r2_50s[_LABELS[method]] = np.repeat(r2_50, num_m)
      r2_75s[_LABELS[method]] = np.repeat(r2_75, num_m)
      times[_LABELS[method]] = (end - begin) / num_trials
    else:
      r2_25s[_LABELS[method]] = np.zeros(num_m)
      r2_50s[_LABELS[method]] = np.zeros(num_m)
      r2_75s[_LABELS[method]] = np.zeros(num_m)
      times[_LABELS[method]] = np.zeros(num_m)
      for m_idx in range(num_m):
        m = int(m_range[m_idx])
        batch_size = int(n / m)
        if batch_size < d:
          raise RuntimeError(
              str(m) + " models requires " + str(m * d) +
              " points, but given features only has " + str(n) + " points.")
        print("finished tukey, m = " + str(m))
        begin = time.time()
        r2_25s["Tukey"][m_idx], r2_50s["Tukey"][m_idx], r2_75s["Tukey"][
            m_idx] = run_dp_tukey(features, labels, epsilon, delta, m,
                                  num_trials)
        end = time.time()
        times["Tukey"][m_idx] = (end - begin) / num_trials
  return r2_25s, r2_50s, r2_75s, times


def ptr_heuristic(d_range, m_range, epsilon, delta, num_trials):
  """Computes and plots lower bound on distance to unsafety for different d, m.

  Args:
    d_range: Range for d, the number of features in the synthetic regression
      problem.
    m_range: Range for m, the number of models used by the Tukey mechanism.
    epsilon: Privacy parameter epsilon.
    delta: Privacy parameter delta.
    num_trials: Number of trials to run for each (d, m) pair.

  Returns:
    Plot of computed distance lower bound with a line for each d, over m_range.
  """
  results = np.zeros((len(d_range), len(m_range), num_trials))
  for d_idx in range(len(d_range)):
    d = int(d_range[d_idx])
    print("d: " + str(d))
    m_idx = 0
    while m_idx < len(m_range):
      m = int(m_range[m_idx])
      trial = 0
      while trial < num_trials:
        features, labels = sklearn.datasets.make_regression(
            (d + 1) * m, n_features=d, n_informative=d, noise=10)
        models = tukey.multiple_regressions(features, labels, m)
        projections = tukey.perturb_and_sort_matrix(models.T)
        log_volumes = tukey.log_measure_geq_all_depths(projections)
        max_depth = int(m / 2)
        t = int(max_depth / 2)
        results[d_idx, m_idx,
                trial] = tukey.distance_to_unsafety(log_volumes, epsilon / 2,
                                                    delta, t, -1, t - 1)
        trial += 1
      m_idx += 1
  colors = [
      "firebrick", "hotpink", "lightcoral", "sandybrown", "darkorange",
      "olivedrab", "darkseagreen", "lightseagreen", "deepskyblue", "dodgerblue",
      "mediumblue", "mediumpurple"
  ]
  for d_idx in range(len(d_range)):
    plt.plot(
        m_range,
        np.median(results[d_idx], axis=1) + 1,
        label="d = " + str(d_range[d_idx]),
        color=colors[d_idx])
  plt.plot(
      m_range,
      np.tile(np.log(1 / (2 * delta)), len(m_range)) / (epsilon / 2),
      label="threshold",
      color=colors[-2],
      linestyle="--")
  plt.legend(
      loc="lower center", bbox_to_anchor=(0.45, -0.4), ncol=4, frameon=False)
  plt.xlabel("# models (m)")
  plt.ylabel("lower bound on distance to unsafe dataset")
  plt.savefig("num_models_plot.pdf", bbox_inches="tight")


def plot_r2(r2_25s,
            r2_50s,
            r2_75s,
            m_range,
            num_trials,
            title,
            log_scale=False):
  """Returns plot of R^2 values generated by run_trials.

  Args:
    r2_25s: Dictionary where r2_25s[method] = 0.25 quantile of R^2 of method
      over num_trials trials.
    r2_50s: Dictionary where r2_50s[method] = 0.50 quantile of R^2 of method
      over num_trials trials.
    r2_75s: Dictionary where r2_25s[method] = 0.75 quantile of R^2 of method
      over num_trials trials.
    m_range: Range for number of models used by Tukey mechanism.
    num_trials: Number of trials used to generate results.
    title: Plot title.
    log_scale: Boolean for using log_scale on plot y-axis.
  """
  for method in RegressionMethod:
    plt.plot(
        m_range,
        r2_50s[_LABELS[method]],
        color=_COLORS[method],
        label=_LABELS[method],
        linestyle=_LINESTYLES[method])
    plt.fill_between(
        m_range,
        r2_25s[_LABELS[method]],
        r2_75s[_LABELS[method]],
        color=_COLORS[method],
        alpha=0.3)
  plt.xlabel("# models $m$")
  plt.ylabel("$R^2$ from " + str(num_trials) + " trials")
  plt.legend(
      loc="lower center", bbox_to_anchor=(0.45, -0.35), ncol=3, frameon=False)
  if log_scale:
    plt.yscale("symlog")
  plt.title(title)
  plt.savefig(title + ".pdf", bbox_inches="tight")
  plt.clf()


def run_nondp_sampled(features, labels, num_trials, num_samples):
  """Returns R^2 statistics from num_trials of nondp, on num_samples each.

  Args:
    features: Matrix where each row is a feature vector. Assumed to have
      intercept feature.
    labels: Vector of labels.
    num_trials: Number of trials to run.
    num_samples: Number of samples to use for each trial.

  Returns:
    0.25, 0.5, and 0.75 R^2 quantiles from num_trials. Unlike run_nondp, here
    the sampling means this process is nondeterministic. In particular, the
    quantiles may differ.
  """
  models = np.zeros((num_trials, len(features[0])))
  for trial in range(num_trials):
    sample_indices = np.random.choice(np.arange(len(features)), num_samples)
    models[trial, :] = np.squeeze(
        baselines.nondp(features[sample_indices], labels[sample_indices]))
  return r_squared_from_models(models, features, labels)


def plot_time(times, m_range, num_trials, title, log_scale=False):
  """Returns plot of times generated by run_trials.

  Args:
    times: Dictionary where times[method] = average time in seconds taken by
      method over num_trials trialss
    m_range: Range for number of models used by Tukey mechanism.
    num_trials: Number of trials used to generate results.
    title: Plot title.
    log_scale: Boolean for using log_scale on plot y-axis.
  """
  for method in RegressionMethod:
    if method != RegressionMethod.TUKEY:
      plt.plot(
          m_range,
          np.repeat(times[_LABELS[method]], len(m_range)),
          color=_COLORS[method],
          label=_LABELS[method],
          linestyle=_LINESTYLES[method])
    else:
      plt.plot(
          m_range,
          times[_LABELS[method]],
          color=_COLORS[method],
          label=_LABELS[method],
          linestyle=_LINESTYLES[method])
  plt.xlabel("# models $m$")
  plt.ylabel("mean time (s) from " + str(num_trials) + " trials")
  plt.legend(
      loc="lower center", bbox_to_anchor=(0.45, -0.35), ncol=3, frameon=False)
  if log_scale:
    plt.yscale("symlog")
  plt.title(title)
  plt.savefig(title + ".pdf", bbox_inches="tight")
  plt.clf()


def run_and_plot_nondp_sampled_trials(features, labels, num_samples_range,
                                      num_trials, title):
  """Returns plot of R^2 for nondp regression using subsamples of the data.

  Args:
    features: Matrix where each row is a vector of features. Assumed to already
      have intercept feature.
    labels: Vector of labels.
    num_samples_range: Array of different number of samples to use.
    num_trials: Number of trials.
    title: Plot title.
  """
  r2_25 = np.zeros(len(num_samples_range))
  r2_50 = np.zeros(len(num_samples_range))
  r2_75 = np.zeros(len(num_samples_range))
  for num_samples_idx in range(len(num_samples_range)):
    num_samples = int(num_samples_range[num_samples_idx])
    r2_25[num_samples_idx], r2_50[num_samples_idx], r2_75[
        num_samples_idx] = run_nondp_sampled(features, labels, num_trials,
                                             num_samples)
  plt.plot(
      num_samples_range,
      r2_50,
      linestyle=_LINESTYLES[RegressionMethod.NONDP],
      color=_COLORS[RegressionMethod.NONDP])
  plt.fill_between(
      num_samples_range,
      r2_25,
      r2_75,
      color=_COLORS[RegressionMethod.NONDP],
      alpha=0.5)
  plt.xlabel("# random samples for nondp")
  plt.ylabel("$R^2$ from " + str(num_trials) + " trials")
  plt.title(title)
  plt.savefig(title + ".pdf", bbox_inches="tight")

def gaussian_pdf(x_values):
  range_eval = np.linspace(np.amin(x_values), np.amax(x_values), 100)
  mean = np.mean(x_values)
  stddev = np.std(x_values)
  y_values = 1 / (stddev * np.sqrt(
      2 * np.pi)) * np.exp(-(range_eval - mean)**2) / (2 * stddev**2)
  return range_eval, y_values

def run_multiple_regressions_and_plot_histograms(features, labels, num_models,
                                                 title, num_columns):
  """Returns plot of distribution of num_models models along each coordinate.

  Args:
    features: Matrix where each row is a vector of features. Assumed to already
      have intercept feature in the last column.
    labels: Vector of labels.
    num_models: Number of models to train.
    title: Plot title.
    num_columns: Number of columns for the plot.
  """
  _, d = features.shape
  residual = d % num_columns
  if residual != 0:
    multiple_files = True
  else:
    multiple_files = False

  num_rows = d // num_columns
  models = tukey.multiple_regressions(features, labels, num_models)
  _, axs = plt.subplots(
      nrows=num_rows,
      ncols=num_columns,
      figsize=(3 * num_columns, 3 * num_rows))

  for i in range(num_columns):
    vals = models[:, i]
    x_pdf, y_pdf = gaussian_pdf(vals)
    if num_rows == 1:
      _, _, _ = axs[i].hist(vals, density=True, bins=50)
      axs[i].plot(x_pdf, y_pdf, color="red")
    else:
      _, _, _ = axs[i // num_columns, i % num_columns].hist(
          vals, density=True, bins=50)
      axs[i // num_columns, i % num_columns].plot(x_pdf, y_pdf, color="red")

  plt.tight_layout()
  if not multiple_files:
    plt.savefig(title + ".pdf")
  else:
    plt.savefig(title + "1.pdf")
    _, axs = plt.subplots(
        nrows=1,
        ncols=residual,
        figsize=(3 * residual, 3 * num_rows))
    for i in range(residual):
      vals = models[:, num_columns * num_rows + i]
      x_pdf, y_pdf = gaussian_pdf(vals)
      _, _, _ = axs[i].hist(vals, density=True, bins=50)
      axs[i].plot(x_pdf, y_pdf, color="red")
    plt.tight_layout()
    plt.savefig(title + "2.pdf")