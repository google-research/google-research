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

"""Revisiting the Performance of IALS on Item Recommendation Benchmarks."""

import concurrent.futures
import numpy as np


class DataSet():
  """A class holding the train and test data."""

  def __init__(self, train_by_user, train_by_item, test, num_batches):
    """Creates a DataSet and batches it.

    Args:
      train_by_user: list of (user, items)
      train_by_item: list of (item, users)
      test: list of (user, history_items, target_items)
      num_batches: partitions each set using this many batches.
    """
    self.train_by_user = train_by_user
    self.train_by_item = train_by_item
    self.test = test
    self.num_users = len(train_by_user)
    self.num_items = len(train_by_item)
    self.user_batches = self._batch(train_by_user, num_batches)
    self.item_batches = self._batch(train_by_item, num_batches)
    self.test_batches = self._batch(test, num_batches)

  def _batch(self, xs, num_batches):
    batches = [[] for _ in range(num_batches)]
    for i, x in enumerate(xs):
      batches[i % num_batches].append(x)
    return batches


def map_parallel(fn, xs, *args):
  """Applies a function to a list, equivalent to [fn(x, *args) for x in xs]."""
  if len(xs) == 1:
    return [fn(xs[0], *args)]

  num_threads = len(xs)
  executor = concurrent.futures.ProcessPoolExecutor(num_threads)
  futures = [executor.submit(fn, x, *args) for x in xs]
  concurrent.futures.wait(futures)
  results = [future.result() for future in futures]
  return results


class Recommender():
  """A Recommender class used to evaluate a recommendation algorithm.

  Inheriting classes must implement the score() method.
  """

  def _evaluate_user(self, user_history, ground_truth, exclude):
    """Evaluates one user.

    Args:
      user_history: list of items to use in the projection.
      ground_truth: list of target items.
      exclude: list of items to exclude, usually the same as ground_truth.
    Returns:
      A tuple of (Recall@20, Recall@50 and nDCG@100).
    """
    scores = self.score(user_history)
    scores[exclude] = -np.infty
    topk = np.argsort(scores)[::-1]

    def recall(k, gt_set, topk):
      result = 0.0
      for i in range(k):
        if topk[i] in gt_set:
          result += 1
      return result / min(k, len(gt_set))

    def ndcg(k, gt_set, topk):
      result = 0.0
      norm = 0.0
      for i in range(k):
        if topk[i] in gt_set:
          result += 1.0/np.log2(i+2)
      for i in range(min(k, len(gt_set))):
        norm += 1.0/np.log2(i+2)
      return result / norm

    gt_set = ground_truth
    return np.array([
        recall(20, gt_set, topk), recall(50, gt_set, topk),
        ndcg(100, gt_set, topk)
        ])

  def _evaluate_users(self, users):
    """Evaluates a set of users.

    Args:
      users: a list of users, where each user is a tuple
        (id, history, ground truth).
    Returns:
      A dict mapping user id to a tuple of (Recall@20, Recall@50, nDCG@100).
    """
    metrics = {}
    for user_id, ground_truth, history in users:
      if set(ground_truth) & set(history):
        raise ValueError("The history and ground_truth must be disjoint.")
      metrics[user_id] = self._evaluate_user(history, ground_truth, history)
    return metrics

  def evaluate(self, users_batches):
    results = map_parallel(self._evaluate_users, users_batches)
    all_metrics = []
    for r in results:
      all_metrics.extend(list(r.values()))
    return np.mean(all_metrics, axis=0)


class IALS(Recommender):
  """iALS solver."""

  def __init__(self, num_users, num_items, embedding_dim, reg,
               unobserved_weight, stddev):
    self.embedding_dim = embedding_dim
    self.reg = reg
    self.unobserved_weight = unobserved_weight
    self.user_embedding = np.random.normal(
        0, stddev, (num_users, embedding_dim))
    self.item_embedding = np.random.normal(
        0, stddev, (num_items, embedding_dim))
    self._update_user_gramian()
    self._update_item_gramian()

  def _update_user_gramian(self):
    self.user_gramian = np.matmul(self.user_embedding.T, self.user_embedding)

  def _update_item_gramian(self):
    self.item_gramian = np.matmul(self.item_embedding.T, self.item_embedding)

  def score(self, user_history):
    user_emb = project(
        user_history, self.item_embedding, self.item_gramian, self.reg,
        self.unobserved_weight)
    result = np.dot(user_emb, self.item_embedding.T)
    return result

  def train(self, ds):
    """Runs one iteration of the IALS algorithm.

    Args:
      ds: a DataSet object.
    """
    # Solve for the user embeddings
    self._solve(ds.user_batches, is_user=True)
    self._update_user_gramian()
    # Solve for the item embeddings
    self._solve(ds.item_batches, is_user=False)
    self._update_item_gramian()

  def _solve(self, batches, is_user):
    """Solves one side of the matrix."""
    if is_user:
      embedding = self.user_embedding
      args = (self.item_embedding, self.item_gramian, self.reg,
              self.unobserved_weight)
    else:
      embedding = self.item_embedding
      args = (self.user_embedding, self.user_gramian, self.reg,
              self.unobserved_weight)
    results = map_parallel(solve, batches, *args)
    for r in results:
      for user, emb in r.items():
        embedding[user, :] = emb


def project(user_history, item_embedding, item_gramian, reg, unobserved_weight):
  """Solves one iteration of the iALS algorithm."""
  if not user_history:
    raise ValueError("empty user history in projection")
  emb_dim = np.shape(item_embedding)[1]
  lhs = np.zeros([emb_dim, emb_dim])
  rhs = np.zeros([emb_dim])
  for item in user_history:
    item_emb = item_embedding[item]
    lhs += np.outer(item_emb, item_emb)
    rhs += item_emb

  lhs += unobserved_weight * item_gramian
  lhs = lhs + np.identity(emb_dim) * reg
  return np.linalg.solve(lhs, rhs)


def solve(data_by_user, item_embedding, item_gramian, global_reg,
          unobserved_weight):
  user_embedding = {}
  for user, items in data_by_user:
    reg = global_reg *(len(items) + unobserved_weight * item_embedding.shape[0])
    user_embedding[user] = project(
        items, item_embedding, item_gramian, reg, unobserved_weight)
  return user_embedding


