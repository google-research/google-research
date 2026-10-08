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

"""Script for running DP clipping experiments."""

from collections.abc import Sequence
from absl import app
import numpy as np

from dp_clip import data_lib
from dp_clip import experiment_lib
from dp_clip import plotting_lib

# ==============================================================================
# CONFIGURATION
# Edit these variables to run different experiments.
# ==============================================================================
DATASET_PATH = "path/to/dataset.csv"
DATASET_NAME = "steam"  # E.g., steam, reddit, lastfm, hn, movielens, etc.
OUTPUT_DIR = "."
BETA = 1.001
HISTOGRAM_RHO = 0.1
NUM_TRIALS = 100
GOAL_QUANTILES = [0.9, 0.95, 0.99]
CLIP_RHO_RANGE = np.linspace(0.01, 0.1, 10)
# ==============================================================================

LOADER_BY_DATASET_NAME = {
    "steam": data_lib.load_steam_dataset,
    "reddit": data_lib.load_reddit_dataset,
    "lastfm": data_lib.load_lastfm_dataset,
    "hn": data_lib.load_hn_dataset,
    "movielens": data_lib.load_movielens_dataset,
    "pantry": data_lib.load_pantry_dataset,
    "twitch": data_lib.load_twitch_dataset,
    "foursquare": data_lib.load_foursquare_dataset,
    "jester": data_lib.load_jester_dataset,
    "slashdot": data_lib.load_slashdot_dataset,
    "ansur": data_lib.load_ansur_dataset,
    "atus": data_lib.load_atus_dataset,
    "nfl": data_lib.load_nfl_dataset,
    "powerlifting": data_lib.load_powerlifting_dataset,
}

SIGNED_DATASETS = frozenset({"jester", "slashdot"})


def main(argv):
  del argv

  if DATASET_NAME not in LOADER_BY_DATASET_NAME:
    raise ValueError(f"Unknown dataset name: {DATASET_NAME}")

  print(f"Running {DATASET_NAME} experiment on {DATASET_PATH}...")
  loader = LOADER_BY_DATASET_NAME[DATASET_NAME]
  data = loader(DATASET_PATH)

  errors = {}
  estimates = {}
  times = {}
  errors[DATASET_NAME], estimates[DATASET_NAME], times[DATASET_NAME] = (
      experiment_lib.run_experiment(
          BETA,
          GOAL_QUANTILES,
          CLIP_RHO_RANGE,
          HISTOGRAM_RHO,
          NUM_TRIALS,
          data,
          nonnegative_data=(DATASET_NAME not in SIGNED_DATASETS),
      )
  )

  # Plot combined errors
  plotting_lib.plot_histogram_errors(
      [DATASET_NAME],
      clip_rho_range=CLIP_RHO_RANGE,
      all_errors=errors,
      output_dir=OUTPUT_DIR,
  )

  # Plot RC for this dataset
  optimal_clip = float(estimates[DATASET_NAME]["Optimal"][0, 0])
  plotting_lib.plot_rc(
      data,
      beta=BETA,
      rho_sum=HISTOGRAM_RHO,
      dataset_name=DATASET_NAME,
      optimal_clip=optimal_clip,
      output_dir=OUTPUT_DIR,
      log_y=True,
  )

  # Plot clip estimates
  plotting_lib.plot_clip_estimates(
      [DATASET_NAME],
      clip_rho_range=CLIP_RHO_RANGE,
      clip_estimates=estimates,
      output_dir=OUTPUT_DIR,
  )

  print("Experiment finished and plots saved.")


if __name__ == "__main__":
  app.run(main)
