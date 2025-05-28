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

"""Download and preprocess M4 dataset."""
import subprocess
import numpy as np
import pandas as pd

# Download M4 data from https://github.com/Mcompetitions/
for freq in ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]:
  subprocess.call([
      "wget",
      "https://raw.githubusercontent.com/Mcompetitions/M4-methods/"
      + "master/Dataset/Train/"
      + freq + "-train.csv"
  ])
  subprocess.call([
      "wget",
      "https://raw.githubusercontent.com/Mcompetitions/M4-methods/"
      + "master/Dataset/Test/"
      + freq + "-test.csv"
  ])

# Load M4 csv data
train_data, test_data = {}, {}
freq_lst = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
for freq in freq_lst:
  train_data[freq] = pd.read_csv(freq + "-train.csv")
  test_data[freq] = pd.read_csv(freq + "-test.csv")

# Extract time series from pandaframe and remove NaN
train_np, test_np = {}, {}
for freq in freq_lst:
  temp_lst = []
  for i in range(train_data[freq].values.shape[0]):
    temp_lst.append(
        np.array(train_data[freq].iloc[i, 1:].dropna().values, dtype=float))
  train_np[freq] = temp_lst

  temp_lst = []
  for i in range(test_data[freq].values.shape[0]):
    temp_lst.append(
        np.array(test_data[freq].iloc[i, 1:].dropna().values, dtype=float))
  test_np[freq] = temp_lst

# Save data dictionaries into numpy files
np.save("train.npy", train_np)
np.save("test.npy", test_np)
