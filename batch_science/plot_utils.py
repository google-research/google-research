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

"""Helper functions for creating plots."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_science.measurement_utils import compute_steps_to_result
from batch_science.measurement_utils import get_index_values


def create_subplots(nrows, ncols, plot_width=9, subplot_aspect_ratio=8 / 7):
  """Creates a subplot grid with the specified width and aspect ratio."""
  plot_height = nrows * plot_width / ncols / subplot_aspect_ratio
  return plt.subplots(nrows, ncols, figsize=(plot_width, plot_height))


def plot_steps_to_result(ax,
                         results,
                         add_scaling=True,
                         scaling_label=None,
                         normalizing_batch_size=None):
  """Plots steps to result vs batch size.

  Args:
   ax: Instance of pyplot.axes.Axes on which to plot.
   results: DataFrame of measurements indexed by (batch_size, step) with one row
     per batch size. Or, a dictionary of such DataFrames.
   add_scaling: Whether to draw a line indicating "perfect scaling".
   scaling_label: The label in the results dictionary used to draw the "perfect
     scaling" line (provided add_scaling is True). If not specified, a separate
     line is drawn for each label.
   normalizing_batch_size: If specified, the steps to result curves are
     normalized for each label in the results dictionary by the number of steps
     at this batch size.
  """
  if isinstance(results, pd.DataFrame):
    results = {"": results}

  for label, df in results.items():
    batch_sizes = get_index_values(df, "batch_size")
    steps = get_index_values(df, "step")

    # Possibly normalize the steps.
    if normalizing_batch_size:
      normalizing_index = np.where(batch_sizes == normalizing_batch_size)[0]
      if len(normalizing_index) != 1:
        raise ValueError(
            "Expected one row with batch_size={}, but found {}".format(
                normalizing_batch_size, len(normalizing_index)))
      steps = steps.astype(np.float) / steps[normalizing_index]

    # Plot steps to result.
    ax.plot(batch_sizes, steps, "^-", label=label)

    # Possibly plot "perfect scaling".
    if add_scaling and (not scaling_label or label == scaling_label):
      if normalizing_batch_size:
        scale = steps[normalizing_index] * normalizing_batch_size
      else:
        scale = steps[0] * batch_sizes[0]
      linear_scaling = scale / batch_sizes
      ax.plot(batch_sizes, linear_scaling, "k--", label="_nolegend_")

  # Format the axes.
  ax.set_xlabel("Batch Size")
  if normalizing_batch_size:
    ylabel = "Steps / (Steps at B={})".format(normalizing_batch_size)
  else:
    ylabel = "Steps"
  ax.set_ylabel(ylabel)
  ax.set_xscale("log", basex=2)
  ax.set_yscale("log", basey=2)
  ax.grid(True)


def plot_optimal_metaparameter_values(ax, parameter_to_plot, steps_to_result,
                                      workload_metadata):
  """Plots the values of the optimal metaparameters vs batch size.

  Args:
   ax: Instance of pyplot.axes.Axes on which to plot.
   parameter_to_plot: One of ["Learning Rate", "Momentum", "Effective Learning
     Rate"].
   steps_to_result: DataFrame of measurements indexed by (batch_size, step)
     corresponding to the optimal measurements for each batch size.
   workload_metadata: A dict containing the metadata for each study.
  """
  # Get the parameters corresponding to the optimal measurements.
  batch_sizes = get_index_values(steps_to_result, "batch_size")
  trial_ids = get_index_values(steps_to_result, "trial_id")
  optimal_parameters = [
      workload_metadata[batch_size]["trials"][trial_id]["parameters"]
      for batch_size, trial_id in zip(batch_sizes, trial_ids)
  ]

  # Compute y-values for the parameter to plot.
  ylabel = parameter_to_plot
  plot_heuristics = True
  if parameter_to_plot == "Learning Rate":
    yvalues = np.array(
        [parameters["learning_rate"] for parameters in optimal_parameters])
  elif parameter_to_plot == "Momentum":
    yvalues = np.array(
        [parameters["momentum"] for parameters in optimal_parameters])
    plot_heuristics = False
  elif parameter_to_plot == "Effective Learning Rate":
    learning_rates = np.array(
        [parameters["learning_rate"] for parameters in optimal_parameters])
    momenta = np.array(
        [parameters["momentum"] for parameters in optimal_parameters])
    yvalues = learning_rates / (1 - momenta)
    ylabel = "Learning Rate / (1 - Momentum)"
  else:
    raise ValueError(
        "Unrecognized parameter_to_plot: {}".format(parameter_to_plot))

  # Plot the optimal parameter values vs batch size.
  ax.plot(batch_sizes, yvalues, "^-", label="Optimal " + parameter_to_plot)

  # Plot the "linear" and "square root" scaling heuristics for adjusting the
  # metaparameter values with increasing batch size.
  if plot_heuristics:
    linear_heuristic = [
        yvalues[0] * batch_size / batch_sizes[0] for batch_size in batch_sizes
    ]
    ax.plot(
        batch_sizes,
        linear_heuristic,
        linestyle="--",
        c="k",
        label="Linear Heuristic")

    sqrt_heuristic = [
        yvalues[0] * np.sqrt(batch_size / batch_sizes[0])
        for batch_size in batch_sizes
    ]
    ax.plot(
        batch_sizes,
        sqrt_heuristic,
        linestyle="-.",
        c="g",
        label="Square Root Heuristic")

  # Format the axes.
  ax.set_xlabel("Batch Size")
  ax.set_ylabel(ylabel)
  ax.set_xscale("log", basex=2)
  ax.set_yscale("log", basey=2)
  ax.grid(True)


def _unpack_params(params):
  """Extracts vectors of (learning_rate, one_minus_momentum) from parameters."""
  if not params:
    return [], []
  xy = [(p["learning_rate"], 1 - p["momentum"]) for p in params]
  return zip(*xy)


def plot_learning_rate_momentum_scatter(ax,
                                        objective_col_name,
                                        objective_goal,
                                        study_table,
                                        study_metadata,
                                        xlim,
                                        ylim,
                                        maximize=False):
  """Plots a categorized scatter plot of learning rate and (1 - momentum).

  Trials are categorized by those that reached the goal objective value, those
  that did not, and those that diverged during training.

  Args:
   ax: Instance of pyplot.axes.Axes on which to plot.
   objective_col_name: Column name of the objective metric.
   objective_goal: Threshold value of the objective metric indicating a
     successful trial.
   study_table: DataFrame of all measurements in the study indexed by (trial_id,
     step).
   study_metadata: A dict of study metadata.
   xlim: A pair (x_min, x_max) corresponding to the minimum and maximum learning
     rates to plot.
   ylim: A pair (y_min, y_max) corresponding to the minimum and maximum momentum
     values to plot.
    maximize: Whether the goal is to maximize (as opposed to minimize) the
      objective metric.
  """
  # Extract the parameters corresponding to each trial in 3 categories: those
  # that reached the goal objective value, those that did not, and those that
  # diverged during training.
  good_params = []
  bad_params = []
  infeasible_params = []
  comparator = operator.gt if maximize else operator.lt
  for trial_id, trial_metadata in study_metadata["trials"].items():
    params = trial_metadata["parameters"]
    if trial_metadata["status"] == "COMPLETE":
      measurements = study_table.loc[trial_id][objective_col_name]
      if np.any(comparator(measurements, objective_goal)):
        good_params.append(params)
      else:
        bad_params.append(params)
    elif trial_metadata["status"] == "INFEASIBLE":
      infeasible_params.append(params)
    else:
      raise ValueError("Unexpected status: {}".format(trial_metadata["status"]))

  # Plot all good, bad, and infeasible parameter values.
  learning_rate, one_minus_momentum = _unpack_params(good_params)
  ax.scatter(
      learning_rate,
      one_minus_momentum,
      c="b",
      marker="o",
      alpha=1.0,
      s=40,
      label="Goal Achieved")
  learning_rate, one_minus_momentum = _unpack_params(bad_params)
  ax.scatter(
      learning_rate,
      one_minus_momentum,
      c="r",
      marker="^",
      alpha=0.7,
      s=40,
      label="Goal Not Achieved")
  learning_rate, one_minus_momentum = _unpack_params(infeasible_params)
  ax.scatter(
      learning_rate,
      one_minus_momentum,
      alpha=0.7,
      marker="x",
      c="k",
      s=25,
      label="Infeasible")

  # Format the axes.
  ax.set_xlabel("Batch Size")
  ax.set_xscale("log")
  ax.set_xlim(xlim)
  ax.set_ylabel("1 - Momentum")
  ax.set_yscale("log")
  ax.set_ylim(ylim)

  # Plot contour lines.
  grid_x = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), num=50)
  grid_y = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), num=50)
  grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
  grid_z = np.log10(grid_xx / grid_yy)
  ax.contour(grid_xx, grid_yy, grid_z, 10, colors="black", alpha=0.5)

  # Plot the best measurement as a yellow star.
  str_measurement = compute_steps_to_result(study_table, objective_col_name,
                                            objective_goal, maximize, None)
  if not str_measurement.empty:
    best_trial_id = get_index_values(str_measurement, "trial_id")[0]
    best_trial_params = study_metadata["trials"][best_trial_id]["parameters"]
    learning_rate, one_minus_momentum = _unpack_params([best_trial_params])
    ax.scatter(
        learning_rate,
        one_minus_momentum,
        marker="*",
        alpha=1.0,
        s=400,
        c="yellow")


def plot_best_measurements(ax, best_measurements, objective_col_name):
  """Plots the best objective value vs batch size.

  Args:
   ax: Instance of pyplot.axes.Axes on which to plot.
   best_measurements: DataFrame of measurements indexed by batch_size with one
     row per batch size. Or, a dictionary of such DataFrames.
   objective_col_name: Column name of the objective metric.
  """
  if isinstance(best_measurements, pd.DataFrame):
    best_measurements = {"": best_measurements}

  for label, df in best_measurements.items():
    batch_sizes = get_index_values(df, "batch_size")
    best_objective_values = df[objective_col_name]
    ax.plot(batch_sizes, best_objective_values, "^-", label=label)

  # Format the axes.
  ax.set_xlabel("Batch Size")
  ax.set_xscale("log", basex=2)
  ax.grid(True)
