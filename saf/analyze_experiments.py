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

"""Analyze the experimental results from the logs."""

import glob

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_name", "m5",
                    "Dataset to analyze the completed experiments for.")
flags.DEFINE_integer("minimum_model_count", 10,
                     "Minimum model count for an experiment to visualize.")


def scrape_data_from_logs(log_file):
  """Scrapes the validation and test metrics data from the logs."""

  val_metrics = []
  test_metrics = []
  hyperparameters = []

  with open(log_file, "r") as myfile:
    lines = myfile.read().split("\n")
  for ind, line in enumerate(lines):
    if line.startswith("Hyperparameters"):
      line_comma_sep = lines[ind + 2].split(",")
      val_metric = float(line_comma_sep[-2].strip(" "))
      test_metric = float(line_comma_sep[-1].strip(" ").strip("]"))
      val_metrics.append(val_metric)
      test_metrics.append(test_metric)
      hyperparameters.append(lines[ind + 1])

  val_metrics = np.asarray(val_metrics)
  test_metrics = np.asarray(test_metrics)

  return val_metrics, test_metrics, hyperparameters


def display_metrics(val_metrics, test_metrics, title, performance_threshold,
                    filename):
  """Displays the metrics of the trained models so far."""

  # Remove the outliers
  val_metrics = np.asarray(val_metrics)
  test_metrics = np.asarray(test_metrics)
  test_metrics = test_metrics[val_metrics < performance_threshold]
  val_metrics = val_metrics[val_metrics < performance_threshold]

  if val_metrics.size > 0:
    plt.figure()
    plt.plot(val_metrics, test_metrics, "o")
    if val_metrics.size > 2:
      m, b = np.polyfit(val_metrics, test_metrics, 1)
      plt.plot(val_metrics, m * val_metrics + b, "k--")
    v_min = np.min([np.min(val_metrics), np.min(test_metrics)]) * 0.8
    v_max = np.max([np.max(val_metrics), np.max(test_metrics)]) * 1.0
    plt.xlabel("Validation")
    plt.ylabel("Test")
    plt.title(title)
    plt.xlim([v_min, v_max])
    plt.ylim([v_min, v_max])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(filename)


def main(args):
  """Main function to iterate over the experiments."""

  del args  # Not used.

  log_files = glob.glob("./logs/experiment_" + str(FLAGS.dataset_name) +
                        "*.log")

  for log_file in log_files:

    experiment_name = log_file.split("/")[-1]
    experiment_name = experiment_name.split(".")[0]

    val_metrics, test_metrics, hyperparameters = scrape_data_from_logs(log_file)

    if len(val_metrics) > FLAGS.minimum_model_count:
      print("------------------------------------")
      print("Experiment name:")
      print(experiment_name)
      display_metrics(val_metrics, test_metrics, hyperparameters, 1000, "")


if __name__ == "__main__":
  app.run(main)
