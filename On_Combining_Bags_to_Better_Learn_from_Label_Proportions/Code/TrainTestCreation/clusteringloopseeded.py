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

"""Generating Training Bags."""
import os
import pathlib
import shutil
import bagsformethod5
import clusteringbagsmethodseeded
import numpy as np
import singlestraddlebag
import twostraddlebags

rng = np.random.default_rng(73652603)

names_list = ["Heart", "Ionosphere", "Australian"]

n_tot_features_list = [14, 35, 15]

path_to_root_data_dir = (pathlib.Path(__file__).parent /
                         "../../Data/").resolve()

root_dir = str(path_to_root_data_dir) + "/"

for clustering_bags_method in range(1, 9):
  for index_name, name in enumerate(names_list):
    for s in range(1, 6):  # Number of Kfold operations
      Folddirectory = root_dir + name + "/" + "Fold_" + str(s) + "/"
      for splitnumber in range(1, 6):
        splitdir = Folddirectory + "Split_" + str(splitnumber) + "/"
        trainfile = splitdir + name + "_" + str(s) + "_" + str(
            splitnumber) + "-train.csv"

        if clustering_bags_method == 5:
          random_seed = rng.integers(low=1000000, size=1)[0]
          numpy_seed = rng.integers(low=1000000, size=1)[0]
          continue

        cluster_dir = splitdir + "ClusterBags_" + str(
            clustering_bags_method) + "/"

        directory_to_read = splitdir + "ClusterBags_" + str(2) + "/"

        if clustering_bags_method > 5:
          cluster_dir = splitdir + "ClusterBags_" + str(clustering_bags_method -
                                                        1) + "/"

        if os.path.exists(cluster_dir):
          shutil.rmtree(cluster_dir)

        os.makedirs(os.path.dirname(cluster_dir), exist_ok=True)

        print()
        print()
        print("For ", cluster_dir, " ***************")

        if clustering_bags_method == 1:
          clusteringbagsmethodseeded.makeclusterbags(
              n_clusters=1,
              head_inclusion_prob=0.1,
              tail_inclusion_prob=0.1,
              p_law_param=-1.66,
              n_head=125,
              n_tail=125,
              cluster_bias=[1],
              trainfile=trainfile,
              cluster_dir=cluster_dir,
              n_tot_features=n_tot_features_list[index_name],
              option="normal",
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0],
              kmeans_seed=rng.integers(low=1000000, size=1)[0])

        elif clustering_bags_method == 2:
          clusteringbagsmethodseeded.makeclusterbags(
              n_clusters=3,
              head_inclusion_prob=0.9,
              tail_inclusion_prob=0.1,
              p_law_param=-1.66,
              n_head=40,
              n_tail=40,
              cluster_bias=[1, 1, 1],
              trainfile=trainfile,
              cluster_dir=cluster_dir,
              n_tot_features=n_tot_features_list[index_name],
              option="normal",
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0],
              kmeans_seed=rng.integers(low=1000000, size=1)[0])

        elif clustering_bags_method == 3:
          clusteringbagsmethodseeded.makeclusterbags(
              n_clusters=3,
              head_inclusion_prob=0.9,
              tail_inclusion_prob=0.1,
              p_law_param=-1.66,
              n_head=15,
              n_tail=15,
              cluster_bias=[1, 3, 5],
              trainfile=trainfile,
              cluster_dir=cluster_dir,
              n_tot_features=n_tot_features_list[index_name],
              option="normal",
              directory_to_read=directory_to_read,
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0],
              kmeans_seed=rng.integers(low=1000000, size=1)[0])

        elif clustering_bags_method == 4:
          clusteringbagsmethodseeded.makeclusterbags(
              n_clusters=3,
              head_inclusion_prob=-0.9,
              tail_inclusion_prob=-0.1,
              p_law_param=1.66,
              n_head=15,
              n_tail=15,
              cluster_bias=[1, 3, 5],
              trainfile=trainfile,
              cluster_dir=cluster_dir,
              n_tot_features=n_tot_features_list[index_name],
              option="powerlaw",
              directory_to_read=directory_to_read,
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0],
              kmeans_seed=rng.integers(low=1000000, size=1)[0])

        elif clustering_bags_method == 5:
          bagsformethod5.makeonlybags(
              n_clusters=3,
              head_inclusion_powerlaw=[1.7, 1.6, 1.9],
              tail_inclusion_powerlaw=[1.1, 1.2, 1.01],
              p_law_param=-1.66,
              n_head=15,
              n_tail=15,
              cluster_bias=[1, 3, 5],
              trainfile=trainfile,
              cluster_dir=cluster_dir,
              n_tot_features=n_tot_features_list[index_name],
              option="powerlaw",
              directory_to_read=directory_to_read,
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0])

        elif clustering_bags_method == 6:
          singlestraddlebag.makeonlybagswithstraddle(
              n_clusters=3,
              straddle_inclusion=[0.2, 0.2, 0.2],
              tail_inclusion=[0.6, 0.6, 0.6],
              p_law_param=-1.66,
              trainfile=trainfile,
              n_tail=60,
              n_straddle=60,
              cluster_dir=cluster_dir,
              option="powerlaw",
              directory_to_read=directory_to_read,
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0])

        elif clustering_bags_method == 7:
          singlestraddlebag.makeonlybagswithstraddle(
              n_clusters=3,
              straddle_inclusion=[0.4, 0.8, 0.8],
              tail_inclusion=[0.2, 0.2, 0.2],
              p_law_param=-1.66,
              trainfile=trainfile,
              n_tail=60,
              n_straddle=60,
              cluster_dir=cluster_dir,
              option="powerlaw",
              directory_to_read=directory_to_read,
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0])

        elif clustering_bags_method == 8:
          twostraddlebags.makeonlybagswithtwostraddle(
              n_clusters=3,
              straddle_inclusion_first=[0.2, 0.2],
              straddle_inclusion_second=[0.6, 0.6],
              tail_inclusion=[0.2, 0.2, 0.2],
              p_law_param=-1.66,
              trainfile=trainfile,
              n_tail=50,
              n_straddle=50,
              cluster_dir=cluster_dir,
              option="powerlaw",
              directory_to_read=directory_to_read,
              random_seed=rng.integers(low=1000000, size=1)[0],
              numpy_seed=rng.integers(low=1000000, size=1)[0])
