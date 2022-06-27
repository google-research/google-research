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
# Copyright 2021 The On Combining Bags to Better Learn from
# Label Proportions Authors. All Rights Reserved.
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

"""Creating train and test splits and bag training files."""
import pathlib
import pandas as pd
from sklearn.model_selection import KFold

data_dir = (pathlib.Path(__file__).parent / "Dataset/").resolve()

df = pd.read_csv(
    str(data_dir) + "/filtered_ratings.csv",
    usecols=["month", "date", "ts_mod5", "label", "movieId"])

print(len(df.index))

kf = KFold(n_splits=5, shuffle=True, random_state=4091)

i = 0
for train_index, test_index in kf.split(df):
  df_train_split = df.iloc[train_index]
  df_test_split = df.iloc[test_index]

  print("Train set size split " + str(i))
  print(len(df_train_split))

  print("Test set size split " + str(i))
  print(len(df_test_split))

  train_file_to_write = (
      str(data_dir) + "/Split_" + str(i) + "/train_Split_" + str(i) +
      "-filtered_ratings.csv")

  test_file_to_write = (
      str(data_dir) + "/Split_" + str(i) + "/test_Split_" + str(i) +
      "-filtered_ratings.csv")

  bags_file_to_write = (
      str(data_dir) + "/Split_" + str(i) + "/BagTrain_Split_" + str(i) +
      "-filtered_ratings.ftr")

  df_train_split.to_csv(train_file_to_write, index=False)

  print("train file " + str(i) + " written.")

  df_test_split.to_csv(test_file_to_write, index=False)

  print("test file " + str(i) + " written.")

  df_aggregated = df_train_split.groupby(["month", "date",
                                          "ts_mod5"]).agg(list).reset_index()

  # pylint: disable=unnecessary-lambda
  df_aggregated["bag_size"] = df_aggregated["label"].apply(lambda x: len(x))

  # pylint: disable=unnecessary-lambda
  df_aggregated["label_count"] = df_aggregated["label"].apply(lambda x: sum(x))

  print("df_aggregated " + str(i) + " created. Size:")
  print(len(df_aggregated.index))

  df_aggregated.to_feather(bags_file_to_write)

  print("bags train file " + str(i) + " written.")

  i = i + 1
