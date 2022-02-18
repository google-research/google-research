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

"""Creating the 5x5-fold splits."""
import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import *  # pylint: disable=wildcard-import

rng = np.random.default_rng(20210512)

names_list = ["Heart", "Ionosphere", "Australian"]

path_to_root_data_dir = (pathlib.Path(__file__).parent /
                         "../../Data/").resolve()

directory_full_datasets = str(path_to_root_data_dir) + "/FullDatasets/"

root_dir = str(path_to_root_data_dir) + "/"

for name in names_list:
  print("For " + name)
  df = pd.read_csv(directory_full_datasets + name + "-Processed.csv")
  for s in range(1, 6):  # Number of Kfold operations
    # pylint: disable=invalid-name
    Folddirectory = root_dir + name + "/" + "Fold_" + str(s) + "/"
    os.makedirs(os.path.dirname(Folddirectory), exist_ok=True)
    rand_seed = rng.integers(low=1000000, size=1)[0]
    print("Rand State For " + Folddirectory + " is", rand_seed)
    # pylint: disable=undefined-variable
    kfold = KFold(n_splits=5, random_state=rand_seed, shuffle=True)
    print("\t For Fold: ", s)
    splitnumber = 1
    for train_idx, test_idx in kfold.split(df):
      df_train = df.loc[train_idx]
      df_test = df.loc[test_idx]
      print("\t \t For Split: ", splitnumber, ": Train, Test sizes: ",
            len(df_train.index), " , ", len(df_test.index))
      splitdir = Folddirectory + "Split_" + str(splitnumber) + "/"
      os.makedirs(os.path.dirname(splitdir), exist_ok=True)
      trainfile = splitdir + name + "_" + str(s) + "_" + str(
          splitnumber) + "-train.csv"
      testfile = splitdir + name + "_" + str(s) + "_" + str(
          splitnumber) + "-test.csv"

      df_train.to_csv(trainfile, index=False)
      df_test.to_csv(testfile, index=False)

      splitnumber = splitnumber + 1
