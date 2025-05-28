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

# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Generates Toy Data."""

import argparse

import numpy as np


def generate_sine_series(length):
  """Generates an example time series that can be used with the model.

  Args:
    length: The length of the time series.

  Returns:
    A tuple of the adjacency matrix and the array of time series.
  """
  adjacency_matrix = np.array([
      [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
      [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
      [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
      [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
      [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
      [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
  ])
  time_steps = np.arange(length)
  sine_series1 = np.sin(time_steps)
  sine_series2 = np.sin(time_steps * 2 + 1.0)
  sine_series3 = np.sin(time_steps * 3 + 2.0)
  series_array = np.stack((
      sine_series1,
      sine_series2,
      sine_series3,
      sine_series1 + sine_series2,
      sine_series1 + sine_series3,
      sine_series2 + sine_series3,
  ),
                          axis=1)

  return adjacency_matrix, series_array.reshape(length, 6, 1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Time series length.
  parser.add_argument("-l", "--length", type=int, default=10000)
  # The path to save the data file.
  parser.add_argument(
      "-p",
      "--path",
      type=str,
      default="./editable_graph_temporal/toy_data.npz")

  args = parser.parse_args()

  adj, time_data = generate_sine_series(args.length)
  np.savez(args.path, x=time_data, adj=adj)
