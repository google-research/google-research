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

"""Script to generate synthetic sine functions."""

import argparse
import os

import numpy as np
import yaml


def gen_sine_data(freq, ts_len, num_nodes,
                  num_feas, noise = False):
  """Generate 3D graph multivariate time serie data.


  Args:
    freq: sine frequency
    ts_len: length of the time series
    num_nodes: number of nodes in the graph
    num_feas: feature dimension
    noise: whether adding noise to the data

  Returns:
    data: (ts_len x num_nodes x num_feas)

  """

  x = np.empty((num_feas, ts_len), "int64")
  x[:] = np.array(range(ts_len)) + np.random.randint(
      -4 * freq, 4 * freq, num_feas).reshape(num_feas, 1)
  data = np.sin(x / 1.0 / freq).astype("float64")  # pylint: disable=redefined-outer-name
  data = data.T

  # Duplicate the Data to mimic 2 nodes graph (time x node x feature)
  data = np.repeat(data[:, np.newaxis, :], num_nodes, axis=1)
  bias = 0.5  #/ P

  for p in range(num_nodes - 1):
    data[:, p + 1, :] = data[:, p, :] + bias

  if noise:
    # add noise
    data = data + np.random.normal(size=(ts_len, num_nodes, num_feas))

  print("raw data generated:", data.shape)
  # plt.plot(data[:,:,0])
  return data


def gen_sine_data_freq(freq, ts_len,
                       num_nodes, num_feas):
  """Generate 2D multivariate time serie data.

  Args:
    freq:
    ts_len:
    num_nodes:
    num_feas:

  Returns:

  """
  x = np.empty((num_feas, ts_len), "int64")
  x[:] = np.array(range(ts_len)) + np.random.randint(
      -4 * freq, 4 * freq, num_feas).reshape(num_feas, 1)
  data = np.sin(x / 1.0 / freq).astype("float64")  # pylint: disable=redefined-outer-name
  data = np.expand_dims(data, 1)
  for p in range(num_nodes - 1):
    x = np.empty((num_feas, ts_len), "int64")
    x[:] = np.array(range(ts_len)) + np.random.randint(
        -4 * freq, 4 * freq, num_feas).reshape(num_feas, 1)
    omega = freq / float(p + 1)
    new_data = np.sin(x / omega).astype("float64")
    new_data = np.expand_dims(new_data, 1)
    data = np.concatenate((data, new_data), axis=1)
  data = data.T
  return data


if __name__ == "__main__":
  parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

  parser = argparse.ArgumentParser()
  # Optional argument flag which defaults to False
  parser.add_argument("-id", "--input-dim", type=int, default=3)
  parser.add_argument("-n", "--num-nodes", type=int, default=10)
  parser.add_argument(
      "-d",
      "--data-config",
      type=str,
      default=f"{parent_dir}/experiments/Sine.yaml")

  args = parser.parse_args()

  # load data config
  with open(args.data_config, "r") as stream:
    try:
      data_config = yaml.safe_load(stream)
      # overwrite args
      vars(args).update(data_config)
    except yaml.YAMLError as exc:
      print(exc)

  gen_freq = 10
  gen_ts_len = 2000

  data_path = os.path.join(parent_dir, "data")
  if not os.path.exists(data_path):
    # Create a new directory because it does not exist
    os.makedirs(data_path)
    print(f"The new directory {data_path} is created!")

  data_file = os.path.join(data_path, "sine.npy")

  data = gen_sine_data(gen_freq, gen_ts_len, args.num_nodes, args.input_dim)
  np.save(data_file, data)
