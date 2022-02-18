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

#!/usr/bin/python
#
# Copyright 2021 The On Combining Bags to Better Learn
# from Label Proportions Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing the raw data files for training/test set creation."""
import pathlib
import random
import numpy as np
import pandas as pd
from sklearn import *  # pylint: disable=wildcard-import

np.random.seed(22052021)

random.seed(22052021)

path_to_root_data_dir = (pathlib.Path(__file__).parent
                         / "../../Data/").resolve()

path_to_top_dir = (pathlib.Path(__file__).parent
                   / "../../").resolve()

directory_full_datasets = str(path_to_root_data_dir) + "/FullDatasets/"

directory_orig_datasets = str(path_to_top_dir) + "/OrigData/"

# ##### Heart##########

file_to_read = directory_orig_datasets + "heart.dat"

file_to_write = directory_full_datasets + "Heart-Processed.csv"

df = pd.read_csv(file_to_read, sep=" ", header=None)

no_of_cols = len(df.columns)

df.columns = ["x." + str(i) for i in range(1, no_of_cols + 1)]

last_col = "x." + str(
    no_of_cols
)  # This is the label column.
# New one with {-1,1} will be added to the front, this will be removed


def label_map(x):
  """Mapping the original labels."""
  if x == 2:
    return 1
  else:
    return -1


# pylint: disable=unnecessary-lambda
df[last_col] = df[last_col].map(lambda x: label_map(x))

# insert dummy bag column
df.insert(loc=0, column="bag", value=1)

# insert label column
df.insert(loc=0, column="label", value=df[last_col])

# drop last column
df = df.drop(last_col, axis=1)

list_of_features = []

for i in range(1, no_of_cols):
  list_of_features.append("x." + str(i))

# Add a constant column
df["constant"] = 1

# pylint: disable=undefined-variable
df[list_of_features] = preprocessing.StandardScaler().fit_transform(
    df[list_of_features]
)

df.to_csv(file_to_write, index=False)

# ##### Ionosphere##########

file_to_read = directory_orig_datasets + "ionosphere.data"

file_to_write = directory_full_datasets + "Ionosphere-Processed.csv"

df = pd.read_csv(file_to_read, header=None)

no_of_cols = len(df.columns)

df.columns = ["x." + str(i) for i in range(1, no_of_cols + 1)]

last_col = "x." + str(
    no_of_cols
)  # This is the label column.
# New one with {-1,1} will be added to the front, this will be removed


def label_map(x):  # pylint: disable=function-redefined
  """Mapping the original labels."""
  if x == "g":
    return 1
  else:
    return -1


# pylint: disable=unnecessary-lambda
df[last_col] = df[last_col].map(lambda x: label_map(x))

# insert dummy bag column
df.insert(loc=0, column="bag", value=1)

# insert label column
df.insert(loc=0, column="label", value=df[last_col])

# drop last column
df = df.drop(last_col, axis=1)

list_of_features = []

for i in range(1, no_of_cols):
  list_of_features.append("x." + str(i))

# Add a constant column
df["constant"] = 1

# pylint: disable=undefined-variable
df[list_of_features] = preprocessing.StandardScaler().fit_transform(
    df[list_of_features]
)

df.to_csv(file_to_write, index=False)

# ##### Australian##########

file_to_read = directory_orig_datasets + "australian.dat"

file_to_write = directory_full_datasets + "Australian-Processed.csv"

df = pd.read_csv(file_to_read, sep=" ", header=None)

no_of_cols = len(df.columns)

df.columns = ["x." + str(i) for i in range(1, no_of_cols + 1)]

last_col = "x." + str(
    no_of_cols
)  # This is the label column.
# New one with {-1,1} will be added to the front, this will be removed


def label_map(x):  # pylint: disable=function-redefined
  """Mapping the original labels."""
  if x == 1:
    return 1
  else:
    return -1


# pylint: disable=unnecessary-lambda
df[last_col] = df[last_col].map(lambda x: label_map(x))

# insert dummy bag column
df.insert(loc=0, column="bag", value=1)

# insert label column
df.insert(loc=0, column="label", value=df[last_col])

# drop last column
df = df.drop(last_col, axis=1)

list_of_features = []

for i in range(1, no_of_cols):
  list_of_features.append("x." + str(i))

# Add a constant column
df["constant"] = 1

# pylint: disable=undefined-variable
df[list_of_features] = preprocessing.StandardScaler().fit_transform(
    df[list_of_features]
)

df.to_csv(file_to_write, index=False)
