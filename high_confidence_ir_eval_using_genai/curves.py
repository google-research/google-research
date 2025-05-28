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

"""Script to generate results for example curves.

This script is used to generate the data for Figure 1. Example usage:

```
# Figure 1.
python curves.py --working_dir="."
```
"""

import argparse
import logging
import os
import pathlib

import numpy as np

from high_confidence_ir_eval_using_genai import utils


PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--working_dir",
    type=str,
    required=True,
    help=(
        "The working directory. It should contain a subfolder 'data' containing"
        " the datasets. A new folder 'output' will be written containing the"
        " results."
    ),
)
PARSER.add_argument(
    "--jobs",
    type=int,
    default=os.cpu_count(),
    help="The number of jobs to run in parallel.",
)


_LABEL_VALUES = 2.0 ** np.arange(5) - 1.0


def get_upper_bound(pred_label_dist, degree):
  cum_pred = np.cumsum(pred_label_dist, axis=1)
  new_dist = pred_label_dist.copy()
  new_dist[:, 0] -= degree
  new_dist[:, 1:-1] -= np.maximum(0, degree - cum_pred[:, :-2])
  new_dist[:, :-1] = np.maximum(0, new_dist[:, :-1])
  new_dist[:, :] /= np.sum(new_dist, axis=1)[:, None]
  return np.sum(new_dist * _LABEL_VALUES[None, :], axis=1)


def get_lower_bound(pred_label_dist, degree):
  cum_pred = np.cumsum(pred_label_dist[:, ::-1], axis=1)[:, ::-1]
  new_dist = pred_label_dist.copy()
  new_dist[:, -1] -= degree
  new_dist[:, 1:-1] -= np.maximum(0, degree - cum_pred[:, 2:])
  new_dist[:, 1:] = np.maximum(0, new_dist[:, 1:])
  new_dist[:, :] /= np.sum(new_dist, axis=1)[:, None]
  return np.sum(new_dist * _LABEL_VALUES[None, :], axis=1)


def get_moved_mean(pred_label_dist, degree):
  if degree == 0:
    return np.sum(pred_label_dist * _LABEL_VALUES[None, :], axis=1)
  elif degree > 0:
    return get_upper_bound(pred_label_dist, degree)
  else:
    return get_lower_bound(pred_label_dist, -degree)


def main():
  args = PARSER.parse_args()
  working_dir = pathlib.Path(args.working_dir)
  logging.basicConfig(
      level=logging.INFO,
      format="[%(asctime)s] [%(levelname)s] %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
  )

  # Set RNG seed for deterministic results.
  np.random.seed(1)

  # Prepare the working directory.
  output_dir = working_dir / "output" / "curves"
  output_dir.mkdir(exist_ok=True, parents=True)

  # Set up synthetic data distributions.
  x = [0, 1, 2, 3, 4]
  y = [
      np.array([0.35, 0.4, 0.2, 0.05, 0.01]),
      np.array([0.01, 0.05, 0.2, 0.4, 0.35]),
      np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
  ]
  y = [list(d / np.sum(d)) for d in y]
  path = output_dir / "distributions.txt"
  logging.info("Writing results to %s", path)
  utils.write_table(path, {"x": x, "y1": y[0], "y2": y[1], "y3": y[2]}, "%.6f")

  # Compute curves.
  x = np.linspace(-1, 1, num=300)
  y_stack = np.stack(y, axis=0)
  y = np.stack([get_moved_mean(y_stack, i) for i in x], axis=-1)
  path = output_dir / "curves.txt"
  logging.info("Writing results to %s", path)
  utils.write_table(path, {"x": x, "y1": y[0], "y2": y[1], "y3": y[2]}, "%.6f")


if __name__ == "__main__":
  main()
