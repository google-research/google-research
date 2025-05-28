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

"""Script to run experiments and generate results for per-query CRC estimates.

This script is used to generate the data for Figure 4. Example usage:

```
# Figure 4.
python perquery.py --working_dir="."
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
  output_dir = working_dir / "output" / "perquery"
  output_dir.mkdir(exist_ok=True, parents=True)

  # Read the datasets.
  ds = datasets.read_datasets(data_dir)

  # Prepare different experiment configurations.
  with concurrent.futures.ProcessPoolExecutor(args.jobs) as executor:
    # Kick off all the experiments.
    futures = {dataset_name: [] for dataset_name in ds}
    for dataset_name, dataset in ds.items():
      for tau in ["0.00", "0.25", "0.50", "0.75"]:
        futures[dataset_name].append((
            tau,
            [
                executor.submit(
                    utils.run_method,
                    dataset["vali"],
                    dataset["test"],
                    methods.crc_per_query,
                    seed=2,  # Different seed than the global one.
                    oracle_ratio=float(tau),
                )
            ],
        ))

    # Collect results for all experiments and write them to disk.
    for dataset_name in ds:
      utils.progress_bar(
          futures[dataset_name],
          desc=f"{dataset_name}",
      )
      for tau, fs in futures[dataset_name]:
        # Get results.
        results = fs[0].result()

        # Sort results by the per-query true DCG.
        idxs = np.argsort(-results["per_query_true_dcg"])
        sort_keys = [
            "per_query_true_dcg",
            "per_query_pred_dcg",
            "lower",
            "upper",
            "coverage",
        ]
        results = {key: results[key][idxs] for key in sort_keys}

        # Compute the per-query data for the figure.
        results = {
            "x": np.arange(len(results["per_query_true_dcg"])),
            "true": results["per_query_true_dcg"],
            "pred": results["per_query_pred_dcg"],
            "predminus": results["per_query_pred_dcg"] - results["lower"],
            "predplus": results["upper"] - results["per_query_pred_dcg"],
            "label": [
                "y" if covered else "n" for covered in results["coverage"]
            ],
        }

        # Write to table.
        path = output_dir / f"{dataset_name}_tau_{tau}.txt"
        logging.info("Writing results to %s", path)
        utils.write_table(path, results, "%.6f")


if __name__ == "__main__":
  main()
