# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Group-wise Evaluation functions supporting Movielens examples.
"""

import array
import collections

from typing import Dict, List, Text, Tuple

import numpy as np
import tensorflow as tf


def evaluate_sparse_dataset(
    user_model,
    movie_model,
    test,
    movies,
    user_features,
    item_features,
    cutoffs,
    groups = None,
    train = None,
):
  """Evaluates a Movielens model on the supplied (SparseTensor) datasets.

  Args:
    user_model: User representation model.
    movie_model: Movie representation model.
    test: Test dataset, in batch format.
    movies: Dataset of movies.
    user_features: A list of user features (used by models consider user
      features).
    item_features: A list of item features (used by models consider item
      features).
    cutoffs: A list of cutoff values at which to compute precision and recall.
    groups: A list of strings for group-wise evaluations. Assume group is among
      input features now.
    train: Training dataset. If supplied, recommendations for training watches
      will be removed.

  Returns:
   Dictionary of metrics.
  """

  # One batch data of `movies` contains all the candidates.
  for i in movies.take(1):
    movie_candidates = i
  movies = movie_candidates["item_id"].values.numpy()

  movie_vocabulary = dict(zip(movies.tolist(), range(len(movies))))

  train_user_to_movie_ids = collections.defaultdict(lambda: array.array("i"))
  test_user_to_movies_ids = collections.defaultdict(lambda: array.array("i"))
  # map user ids to features
  test_user_to_features = collections.defaultdict(dict)
  # map item ids to features
  test_item_to_features = collections.defaultdict(dict)

  def tensor_to_numpy(tensor):
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
    return tensor.numpy().squeeze()

  if train is not None:
    for batch in train:
      users = tensor_to_numpy(batch["user"])
      movies = tensor_to_numpy(batch["item_id"])
      for user, movie in zip(users, movies):
        train_user_to_movie_ids[user].append(movie_vocabulary[movie])

  for batch in test:
    users = tensor_to_numpy(batch["user"])  # shape (batch_size, 1)
    movies = tensor_to_numpy(batch["item_id"])
    user_feature_dict = {
        feature: tensor_to_numpy(batch[feature]) for feature in user_features
    }
    item_features_dict = {
        feature: tensor_to_numpy(batch[feature]) for feature in item_features
    }
    for i, (user, movie) in enumerate(zip(users, movies)):
      item = movie_vocabulary[movie]
      test_user_to_movies_ids[user].append(item)

      if user not in test_user_to_features:
        test_user_to_features[user] = {
            k: v[i] for k, v in user_feature_dict.items()
        }
      if item not in test_item_to_features:
        test_item_to_features[item] = {
            k: v[i] for k, v in item_features_dict.items()
        }

  movie_embeddings = movie_model(movie_candidates).numpy()

  precision_values = collections.defaultdict(list)
  recall_values = collections.defaultdict(list)
  if groups is not None:
    groups_wise_precision_values = {}
    groups_wise_recall_values = {}
    for group in groups:
      groups_wise_precision_values[group] = collections.defaultdict(
          lambda: collections.defaultdict(list))
      groups_wise_recall_values[group] = collections.defaultdict(
          lambda: collections.defaultdict(list))

  for (user, test_movies) in test_user_to_movies_ids.items():
    user_embedding = user_model(test_user_to_features[user]).numpy()
    scores = (user_embedding @ movie_embeddings.T).flatten()
    test_movies = np.frombuffer(test_movies, dtype=np.int32)
    if train is not None:
      train_movies = np.frombuffer(
          train_user_to_movie_ids[user], dtype=np.int32)
      scores[train_movies] = -1e6

    top_ranks = np.argsort(-scores)
    for k in cutoffs:
      top_movies = top_ranks[:k]
      num_test_movies_in_k = sum(x in top_movies for x in test_movies)
      sample_precision = num_test_movies_in_k / k
      sample_recall = num_test_movies_in_k / len(test_movies)
      precision_values[k].append(sample_precision)
      recall_values[k].append(sample_recall)

      # Add group metrics.
      if groups is not None:
        for group in groups:
          if group.startswith("item:"):
            # Parse item-side subgroup features.
            subgroup = test_item_to_features[test_movies][group.strip("item:")]
          else:
            # Parse user-side subgroup features.
            subgroup = test_user_to_features[user][group]
          groups_wise_precision_values[group][subgroup][k].append(
              sample_precision)
          groups_wise_recall_values[group][subgroup][k].append(sample_recall)

  # Logging for averaged performance.
  number_of_samples = len(precision_values[cutoffs[0]])
  precision_results = {
      f"precision_at_{k}": np.mean(precision_values[k]) for k in cutoffs
  }
  recall_results = {
      f"recall_at_{k}": np.mean(recall_values[k]) for k in cutoffs
  }
  print(f"\nAveraged (Num of samples):{number_of_samples}")
  print(precision_results, "\n", recall_results)

  # Logging for group-wise performance.
  if groups is not None:
    for group in groups:
      print("\nGroup:", group)
      target_group_precision = groups_wise_precision_values[group]
      target_group_recall = groups_wise_recall_values[group]
      for subgroup in sorted(list(target_group_precision.keys())):
        print(
            f"Subgroup={subgroup}(Num of samples:{len(target_group_precision[subgroup][cutoffs[0]])}"
        )
        print({
            f"precision_at_{k}": np.mean(target_group_precision[subgroup][k])
            for k in cutoffs
        })
        print({
            f"recall_at_{k}": np.mean(target_group_recall[subgroup][k])
            for k in cutoffs
        })

  return recall_results, precision_results
