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

"""Script to run experiments and generate results for a sweep over a parameter.

This script is used to generate the data for Figures 2 and 3 in the paper.
Example usage:

```
# Figure 2.
python sweep.py --sweep=sizes --working_dir="."

# Top-row of Figure 3.
python sweep.py --sweep=adversarial --working_dir="."

# Bottom-row of Figure 3.
python sweep.py --sweep=oracle --working_dir="."
```
"""

import argparse
import concurrent.futures
import logging
import os
import pathlib

import numpy as np

from high_confidence_ir_eval_using_genai import datasets
from high_confidence_ir_eval_using_genai import methods
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
PARSER.add_argument(
    "--sweep",
    type=str,
    choices=["adversarial", "sizes", "oracle"],
    help="The type of experiment to run.",
)


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
  data_dir = working_dir / "data"
  output_dir = working_dir / "output" / args.sweep
  output_dir.mkdir(exist_ok=True, parents=True)

  # Read the datasets.
  ds = datasets.read_datasets(data_dir)

  # Define the methods to run the experiment on.
  methods_by_name = {
      "emp": methods.bootstrap,
      "ppi": methods.ppi,
      "crc": methods.crc_bootstrap,
  }

  # Prepare different experiment configurations.
  sweep_flag = {
      "adversarial": "adversarial_ratio",
      "oracle": "oracle_ratio",
      "sizes": "vali_size",
  }
  sweep_values = {
      "adversarial": [round(float(x), 2) for x in np.arange(0.0, 1.05, 0.05)],
      "oracle": [round(float(x), 2) for x in np.arange(0.0, 1.05, 0.05)],
      "sizes": [3] + list(range(5, 100, 5)) + [100],
  }
  format_sweep_value = lambda x: str(x) if args.sweep == "sizes" else f"{x:.2f}"

  logging.info(
      "Starting '%s' sweep with values [%s]",
      sweep_flag[args.sweep],
      ", ".join([format_sweep_value(x) for x in sweep_values[args.sweep]]),
  )
  with concurrent.futures.ProcessPoolExecutor(args.jobs) as executor:
    # Kick off all the experiments in parallel.
    futures = {
        dataset_name: {method_name: [] for method_name in methods_by_name}
        for dataset_name in ds
    }
    for dataset_name, dataset in ds.items():
      for method_name, method in methods_by_name.items():
        for sweep_value in sweep_values[args.sweep]:
          futures[dataset_name][method_name].append((
              format_sweep_value(sweep_value),
              [
                  executor.submit(
                      utils.run_method,
                      dataset["vali"],
                      dataset["test"],
                      method,
                      seed=seed,
                      **{sweep_flag[args.sweep]: sweep_value},
                  )
                  for seed in range(500)
              ],
          ))

    # Collect results for all experiments and write them to disk.
    for dataset_name in ds:
      for method_name in methods_by_name:
        utils.progress_bar(
            futures[dataset_name][method_name],
            desc=f"{dataset_name} {method_name}",
        )

        # Get coverage and widths values.
        output = {
            "coverage": {"x": [], "y": [], "lower": [], "upper": []},
            "width": {"x": [], "y": [], "lower": [], "upper": []},
        }
        for sweep_value, fs in futures[dataset_name][method_name]:
          result = utils.eval_method(fs)
          for key in output:
            output[key]["x"].append(sweep_value)
            output[key]["y"].append(result[key]["mean"])
            output[key]["lower"].append(result[key]["lower"])
            output[key]["upper"].append(result[key]["upper"])

        # Write results as tables.
        path = output_dir / f"{dataset_name}_{method_name}_coverage.txt"
        logging.info("Writing results to %s", path)
        utils.write_table(path, output["coverage"], "%.3f")
        path = output_dir / f"{dataset_name}_{method_name}_width.txt"
        logging.info("Writing results to %s", path)
        utils.write_table(path, output["width"], "%.6f")


if __name__ == "__main__":
  main()
