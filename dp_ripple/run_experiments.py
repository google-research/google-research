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

"""Script for running the ripple sum, count, and vote experiments.

To run:
bash run.sh
"""

import os

from absl import app

from dp_ripple import experiments


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Define common parameters
  num_samples = 5000
  eps_values = [0.1, 0.5, 1, 2, 3, 4, 5]
  output_dir = os.environ.get("PLOT_OUTPUT_DIR", ".")

  print("Running Sum experiment...")
  experiments.sum_plot_norm_vs_eps_for_multiple_k(
      d=20,
      num_samples=num_samples,
      k_values=[1, 3, 5, 10, 20],
      eps_values=eps_values,
      output_dir=output_dir,
  )

  print("Running Count experiment...")
  experiments.count_plot_norm_vs_eps_for_multiple_k(
      d=20,
      num_samples=num_samples,
      eps_values=eps_values,
      k_values=[1, 3, 5, 10, 20],
      output_dir=output_dir,
  )

  print("Running Vote experiment...")
  experiments.vote_plot_norm_vs_eps_multiple_d(
      d_values=[2, 5, 10, 20],
      num_samples=num_samples,
      eps_values=eps_values,
      output_dir=output_dir,
  )


if __name__ == "__main__":
  app.run(main)
