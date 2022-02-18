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

"""Generating Training Bags for two straddle bags."""
import pickle
import random
import numpy as np
import pandas as pd


def makeonlybagswithtwostraddle(n_clusters,
                                straddle_inclusion_first,
                                straddle_inclusion_second,
                                tail_inclusion,
                                p_law_param,
                                n_straddle,
                                n_tail,
                                trainfile,
                                cluster_dir,
                                option,
                                directory_to_read,
                                random_seed=None,
                                numpy_seed=None):
  # pylint: disable=unused-argument
  """Generating Training Bags for two straddle bags."""

  random.seed(random_seed)

  np.random.seed(numpy_seed)

  train_df = pd.read_csv(trainfile)

  train_df.reset_index(drop=True, inplace=True)

  # ###Reading instead of creating
  cluster_to_indices_list = pickle.load(
      open(directory_to_read + "cluster_indices", "rb"))

  for cluster_label in range(n_clusters):
    print("size of cluster ", cluster_label, " is ",
          len(cluster_to_indices_list[cluster_label]))

  # All Bags
  all_bags_list = []

  # #create the first straddle bags
  straddle_bags_first = []

  for _ in range(n_straddle):

    this_bag = []
    for cluster_label in range(n_clusters - 1):
      no_of_indices = len(cluster_to_indices_list[cluster_label])

      no_of_sampled_indices = np.random.binomial(
          n=no_of_indices, p=straddle_inclusion_first[cluster_label])
      this_bag = this_bag + random.sample(
          cluster_to_indices_list[cluster_label], no_of_sampled_indices)
    straddle_bags_first.append(this_bag)
    print("A straddle bag created")
    all_bags_list.append(this_bag)

  straddle_bags_file = cluster_dir + "straddle_bags_first"
  with open(straddle_bags_file, "wb") as writing_to_straddle_bags_file:
    pickle.dump(straddle_bags_first, writing_to_straddle_bags_file)

  # #create the second straddle bags
  straddle_bags_second = []

  for _ in range(n_straddle):

    this_bag = []
    for cluster_label in range(1, n_clusters):
      no_of_indices = len(cluster_to_indices_list[cluster_label])

      no_of_sampled_indices = np.random.binomial(
          n=no_of_indices, p=straddle_inclusion_second[cluster_label - 1])
      this_bag = this_bag + random.sample(
          cluster_to_indices_list[cluster_label], no_of_sampled_indices)
    straddle_bags_second.append(this_bag)
    print("A straddle bag created")
    all_bags_list.append(this_bag)

  straddle_bags_file = cluster_dir + "straddle_bags_second"
  with open(straddle_bags_file, "wb") as writing_to_straddle_bags_file:
    pickle.dump(straddle_bags_second, writing_to_straddle_bags_file)

  # create the tail bags
  cluster_label_to_tail_bags_list = []

  for cluster_label in range(n_clusters):
    this_cluster_tail_bags = []
    no_of_indices = len(cluster_to_indices_list[cluster_label])
    for _ in range(n_tail):
      no_of_sampled_indices = np.random.binomial(
          n=no_of_indices, p=tail_inclusion[cluster_label])
      this_bag = random.sample(cluster_to_indices_list[cluster_label],
                               no_of_sampled_indices)
      this_bag.sort()
      this_cluster_tail_bags.append(this_bag)
      all_bags_list.append(this_bag)
    cluster_label_to_tail_bags_list.append(this_cluster_tail_bags)
    tail_bags_file = cluster_dir + "tail_bags_" + str(cluster_label)
    with open(tail_bags_file, "wb") as writing_to_tail_bags_file:
      pickle.dump(this_cluster_tail_bags, writing_to_tail_bags_file)

  # write all bags
  all_bags_file = cluster_dir + "all_bags"
  with open(all_bags_file, "wb") as writing_to_all_bags_file:
    pickle.dump(all_bags_list, writing_to_all_bags_file)

  # create the raw training set using all bags

  new_train_df = pd.DataFrame()

  bag_no = 1

  for bag_list in all_bags_list:
    if not bag_list:
      continue
    this_bag_df = train_df.iloc[bag_list].copy()
    this_bag_df["bag"] = bag_no
    new_train_df = new_train_df.append(this_bag_df, ignore_index=True)
    bag_no = bag_no + 1

  new_train_df = new_train_df.sample(frac=1)
  new_train_df.to_csv(cluster_dir + "full_train.csv", index=False)
