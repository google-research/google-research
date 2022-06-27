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

"""Generating Training Bags for head and tail."""
import pickle
import random
import numpy as np
import pandas as pd
import scipy
from sklearn.cluster import *  # pylint: disable=wildcard-import


def makeclusterbags(n_clusters,
                    head_inclusion_prob,
                    tail_inclusion_prob,
                    p_law_param,
                    n_head,
                    n_tail,
                    cluster_bias,
                    trainfile,
                    cluster_dir,
                    n_tot_features,
                    option,
                    random_seed=None,
                    numpy_seed=None,
                    kmeans_seed=None,
                    directory_to_read=None):
  # pylint: disable=unused-argument
  """Generating Training Bags for head and tail."""

  random.seed(random_seed)

  np.random.seed(numpy_seed)

  if kmeans_seed is None:
    print("kmeans seed = None")
  else:
    print("kmeans seed = ", kmeans_seed)

  train_df = pd.read_csv(trainfile)

  train_df.reset_index(drop=True, inplace=True)

  cluster_indices_filename = ""

  if directory_to_read is None:

    cluster_indices_filename = cluster_dir + "cluster_indices"

    list_of_features = []

    for i in range(1, n_tot_features):
      list_of_features.append("x." + str(i))

    list_of_features.append("constant")

    # pylint: disable=undefined-variable
    cluster_algo = KMeans(
        n_clusters=n_clusters, init="k-means++", random_state=kmeans_seed)

    feature_matrix = train_df[list_of_features].to_numpy()

    cluster_algo.fit(feature_matrix)

    cluster_labels = cluster_algo.labels_.tolist()

    cluster_to_indices_list = []

    def indicesofelements(list_of_elements, element):
      """Return the indices of an element from a list."""
      return [i for i, ele in enumerate(list_of_elements) if ele == element]

    for cluster_label in range(n_clusters):
      cluster_to_indices_list.append(
          indicesofelements(cluster_labels, cluster_label))

    # for cluster_label in range(n_clusters):
    #     print("size of cluster ", cluster_label, " is ",
    #     len(cluster_to_indices_list[cluster_label]))

    writing_to_cluster_indices_filename = open(cluster_indices_filename, "wb")

    pickle.dump(cluster_to_indices_list, writing_to_cluster_indices_filename)

    writing_to_cluster_indices_filename.close()

  else:
    # ###Reading instead of creating
    cluster_indices_filename = directory_to_read + "cluster_indices"
    cluster_to_indices_list = pickle.load(open(cluster_indices_filename, "rb"))

  for cluster_label in range(n_clusters):
    print("size of cluster ", cluster_label, " is ",
          len(cluster_to_indices_list[cluster_label]))

  # #create the head bags
  cluster_label_to_head_bags_list = []

  # All Bags
  all_bags_list = []

  for cluster_label in range(n_clusters):
    this_cluster_head_bags = []
    no_of_indices = len(cluster_to_indices_list[cluster_label])
    for i in range(n_head * cluster_bias[cluster_label]):

      if option == "powerlaw":
        head_inclusion_prob = scipy.stats.powerlaw.rvs(p_law_param, size=1)[0]

      no_of_sampled_indices = np.random.binomial(
          n=no_of_indices, p=head_inclusion_prob)
      this_bag = random.sample(cluster_to_indices_list[cluster_label],
                               no_of_sampled_indices)
      this_bag.sort()
      this_cluster_head_bags.append(this_bag)
      all_bags_list.append(this_bag)
    cluster_label_to_head_bags_list.append(this_cluster_head_bags)
    head_bags_file = cluster_dir + "head_bags_" + str(cluster_label)
    with open(head_bags_file, "wb") as writing_to_head_bags_file:
      pickle.dump(this_cluster_head_bags, writing_to_head_bags_file)

  # create the tail bags
  cluster_label_to_tail_bags_list = []

  for cluster_label in range(n_clusters):
    this_cluster_tail_bags = []
    no_of_indices = len(cluster_to_indices_list[cluster_label])
    for i in range(n_tail * cluster_bias[cluster_label]):

      if option == "powerlaw":
        tail_inclusion_prob = scipy.stats.powerlaw.rvs(p_law_param, size=1)[0]
      no_of_sampled_indices = np.random.binomial(
          n=no_of_indices, p=tail_inclusion_prob)
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
