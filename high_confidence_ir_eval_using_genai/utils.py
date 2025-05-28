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

"""Utilities for running and evaluating methods."""

import concurrent.futures
import pathlib

import numpy as np
import tqdm


def _conf_interval(
    a, p = 0.95, seed = 0, b = 10_000
):
  """Helper to compute confidence interval using empirical bootstrapping."""
  a = np.asarray(a)
  rng = np.random.default_rng(seed)
  values = np.mean(rng.choice(a, (b, a.shape[0]), replace=True), axis=-1)
  values = np.sort(values, kind="stable")
  alpha = (1.0 - p) / 2.0
  low = values[int(np.floor(b * alpha))]
  high = values[int(np.ceil(b * (1.0 - alpha)))]
  return low, high


def run_method(
    ds_vali,
    ds_test,
    method,
    seed,
    vali_size = None,
    oracle_ratio = None,
    adversarial_ratio = None,
):
  """Runs a method on a given seed and dataset and returns coverage/width.

  This runs a specific method from methods.py on a given seed and dataset. It
  also applies the oracle and adversarial distributions if specified.

  Args:
    ds_vali: Validation set to use.
    ds_test: Test set to use.
    method: Method to run.
    seed: Seed to use.
    vali_size: Size of the validation set to use.
    oracle_ratio: Ratio of oracle examples in the validation set.
    adversarial_ratio: Ratio of adversarial examples in the validation set.

  Returns:
    The results of running the method on the data. This is a dict with the
    following keys:
    - lower: Predicted lower bound on the DCG.
    - upper: Predicted upper bound on the DCG.
    - width: Width of the predicted confidence interval (upper - lower).
    - coverage: Whether the true DCG falls within the confidence interval.
    - true_dcg: The true DCG on the test set.
    - pred_dcg: The predicted DCG on the test set.
    - per_query_true_dcg: The per-query true DCG on the test set.
    - per_query_pred_dcg: The per-query predicted DCG on the test set.
  """
  # Prepare dataset for this local run and the given seed. We choose a seed that
  # is larger than 2 to avoid collision with the global seed (1) and the seed
  # that was used for figure 4 in 'perquery.py' (2).
  ds_vali = ds_vali.with_seed(seed + 3)
  ds_test = ds_test.with_seed(seed + 3)

  # Sample validation set so it is of the given size.
  if vali_size is not None:
    ds_vali = ds_vali.sample(vali_size)

  # Apply oracle distribution.
  if oracle_ratio is not None:
    ds_vali = ds_vali.oracle_label_dist(oracle_ratio)
    ds_test = ds_test.oracle_label_dist(oracle_ratio)

  # Apply adversarial distribution.
  if adversarial_ratio is not None:
    ds_vali = ds_vali.adversarial_label_dist(adversarial_ratio)
    ds_test = ds_test.adversarial_label_dist(adversarial_ratio)

  # Compute bounds for the given method.
  lower, upper = method(ds_vali, ds_test)

  # Compute coverage (this checks for per-query bounds or whole dataset bounds).
  if isinstance(lower, float):
    coverage = lower <= ds_test.true_dcg() <= upper
  else:
    coverage = np.logical_and(
        np.less_equal(lower, ds_test.per_query_true_dcg()),
        np.greater_equal(upper, ds_test.per_query_true_dcg()),
    )

  # Return the computed results.
  return {
      "lower": lower,
      "upper": upper,
      "width": upper - lower,
      "coverage": coverage,
      "true_dcg": ds_test.true_dcg(),
      "pred_dcg": ds_test.pred_dcg(),
      "per_query_true_dcg": ds_test.per_query_true_dcg(),
      "per_query_pred_dcg": ds_test.per_query_pred_dcg(),
  }


def eval_method(
    futures,
):
  """Evaluates a method from its futures.

  Args:
    futures: List of futures that contain the results of `run_method`. This list
      represents many repetitions of a single method so we can compute
      statistics like the mean and confidence interval across many runs.

  Returns:
    The mean and 95% confidence interval of the coverage and width of the
    method. This is a dict of the form:
    {
      "coverage": {
        "mean": 0.1,
        "lower": 0.05,
        "upper": 0.15,
      },
      "width": {
        "mean": 0.2,
        "lower": 0.17,
        "upper": 0.23,
      },
    }
  """
  # Extract coverage and width from the results and flatten them so we get a
  # dict of the form:
  # {
  #   "coverage": [1.0, 0.0, 1.0, ...],
  #   "width": [0.5, 0.4, 0.6, ...],
  # }
  results = {"coverage": [], "width": []}
  for future in futures:
    results["coverage"].append(future.result()["coverage"])
    results["width"].append(future.result()["width"])

  # Extract the mean and confidence interval for each key, so we get a dict of
  # the form:
  # {
  #   "coverage": {
  #     "mean": 0.1,
  #     "lower": 0.05,
  #     "upper": 0.15,
  #   },
  #   "width": {
  #     "mean": 0.2,
  #     "lower": 0.17,
  #     "upper": 0.23,
  #   },
  # }
  output = {}
  for key, values in results.items():
    values = np.array(values)
    lower, upper = _conf_interval(values)
    output[key] = {
        "mean": float(np.mean(values)),
        "lower": lower,
        "upper": upper,
    }
  return output


def progress_bar(
    results,
    desc,
):
  """Shows a progress bar for a nested list of futures.

  This function shows a progress bar and blocks until all futures are completed.

  Args:
    results: A nested list of futures to wait for.
    desc: Description of the progress bar.
  """
  # Flatten the nested futures.
  flat_futures = []
  for _, runs in results:
    flat_futures.extend(runs)
  # Await all futures and show a progress bar that is updated as futures are
  # completed.
  for _ in tqdm.tqdm(
      concurrent.futures.as_completed(flat_futures),
      desc=desc,
      unit="run",
      total=len(flat_futures),
  ):
    pass


def write_table(
    path,
    results,
    float_fmt = "%.3f",
):
  """Writes stats to a space-separated file for the pgfplots in the paper.

  Args:
    path: Path of a .txt file to write the table to.
    results: Dict of results to write to the table. Each entry in this
      dictionary should have the same length.
    float_fmt: Format string for floats.
  """
  # Assert all results have the same length, can't create a table otherwise.
  lengths = [len(values) for values in results.values()]
  nr_rows = lengths[0]
  if not all(length == nr_rows for length in lengths):
    raise ValueError("All results must have the same length.")

  # Format floats using the precision.
  results = {
      key: [
          float_fmt % value if isinstance(value, float) else str(value)
          for value in values
      ]
      for key, values in results.items()
  }
  keys = list(results.keys())  # Materialize keys to ensure same order.

  # Compute the character width of each column in the output table. We use this
  # to right-adjust the text in the table.
  sizes = {key: max(max(map(len, results[key])), len(key)) for key in keys}

  # Write table with right-adjusted strings.
  with open(path, "wt") as f:
    f.write(" ".join([key.rjust(sizes[key]) for key in keys]) + "\n")
    for row in range(nr_rows):
      f.write(
          " ".join(results[key][row].rjust(sizes[key]) for key in keys) + "\n"
      )
