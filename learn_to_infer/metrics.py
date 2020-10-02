# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Functions for computing performance metrics of various inference methods.
"""
import collections
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as onp
import sklearn
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture


def accuracy(preds, labels, unused_num_modes):
  return onp.mean(preds == labels)


@functools.partial(onp.vectorize, signature="(n)->(m)")
def to_pairwise(x):
  pairwise = x[onp.newaxis, :] == x[:, onp.newaxis]
  return pairwise[onp.tril_indices_from(pairwise, k=-1)]


def pairwise_accuracy(preds, labels, unused_num_modes):
  preds = to_pairwise(preds)
  labels = to_pairwise(labels)
  return jnp.mean(preds == labels)


def permutation_invariant_accuracy(preds, labels, num_modes):
  permutations = jnp.array(list(itertools.permutations(range(num_modes))))
  permuted_labels = jax.lax.map(lambda p: p[labels], permutations)
  acc = jnp.max(
      jax.lax.map(lambda ls: jnp.mean(ls == preds), permuted_labels))
  return acc


def pairwise_f1(preds, labels, unused_num_modes):
  preds = to_pairwise(preds)
  labels = to_pairwise(labels)
  return binary_f1(preds, labels, unused_num_modes)


def pairwise_micro_f1(preds, labels, unused_num_modes):
  preds = to_pairwise(preds)
  labels = to_pairwise(labels)
  return micro_f1(preds, labels, unused_num_modes)


def pairwise_macro_f1(preds, labels, unused_num_modes):
  preds = to_pairwise(preds)
  labels = to_pairwise(labels)
  return macro_f1(preds, labels, unused_num_modes)


def binary_f1(preds, labels, unused_num_modes):
  return sklearn.metrics.f1_score(labels, preds, average="binary")


def macro_f1(preds, labels, unused_num_modes):
  return sklearn.metrics.f1_score(labels, preds, average="macro")


def micro_f1(preds, labels, unused_num_modes):
  return sklearn.metrics.f1_score(labels, preds, average="micro")


def permutation_invariant_binary_f1(preds, labels, unused_num_modes):
  f1_pos = binary_f1(preds, labels, unused_num_modes)
  permuted_predictions = onp.array([1, 0])[preds]
  f1_neg = binary_f1(permuted_predictions, labels, unused_num_modes)
  return onp.maximum(onp.mean(f1_pos), onp.mean(f1_neg))

METRIC_FNS = {
    "accuracy": accuracy,
    "pairwise_accuracy": pairwise_accuracy,
    "permutation_invariant_accuracy": permutation_invariant_accuracy,
    "binary_f1": binary_f1,
    "permutation_invariant_binary_f1": permutation_invariant_binary_f1,
    "pairwise_f1": pairwise_f1,
    "micro_f1": micro_f1,
    "macro_f1": macro_f1,
    "pairwise_micro_f1": pairwise_micro_f1,
    "pairwise_macro_f1": pairwise_macro_f1
}


def em_fit_and_predict(xs, num_modes):
  return sklearn.mixture.GaussianMixture(
      n_components=num_modes,
      covariance_type="full",
      init_params="kmeans",
      n_init=3).fit_predict(xs)


def spectral_rbf_fit_and_predict(xs, num_modes):
  return sklearn.cluster.SpectralClustering(
      n_clusters=num_modes,
      n_init=3,
      affinity="rbf").fit_predict(xs)


def agglomerative_fit_and_predict(xs, num_modes):
  return sklearn.cluster.AgglomerativeClustering(
      n_clusters=num_modes,
      affinity="euclidean").fit_predict(xs)


METHODS = {"em": em_fit_and_predict,
           "spectral_rbf": spectral_rbf_fit_and_predict,
           "agglomerative": agglomerative_fit_and_predict}


def compute_baseline_metrics(xs, cs,
                             num_modes,
                             predict_fns=METHODS,
                             metrics=[
                                 "pairwise_accuracy", "pairwise_f1",
                                 "pairwise_micro_f1", "pairwise_macro_f1"
                             ]):
  batch_size = xs.shape[0]
  metric_lists = collections.defaultdict(lambda: collections.defaultdict(list))
  for i in range(batch_size):
    for name, predict_fn in predict_fns.items():
      predicted_cs = predict_fn(xs[i], num_modes)
      for metric_name in metrics:
        m = METRIC_FNS[metric_name](predicted_cs, cs[i], num_modes)
        metric_lists[name][metric_name].append(m)

  avg_metrics = collections.defaultdict(dict)
  for method_name, metric_dict in metric_lists.items():
    for metric_name, metric_list in metric_dict.items():
      avg_metrics[method_name][metric_name] = onp.mean(metric_list)
  return avg_metrics


def compute_metrics(cs, pred_cs, num_modes,
                    metrics=["pairwise_accuracy", "pairwise_f1",
                             "pairwise_micro_f1", "pairwise_macro_f1"]):
  batch_size = cs.shape[0]
  metric_lists = collections.defaultdict(list)
  for i in range(batch_size):
    for metric_name in metrics:
      m = METRIC_FNS[metric_name](pred_cs[i], cs[i], num_modes)
      metric_lists[metric_name].append(m)

  avg_metrics = {}
  for metric_name, metric_list in metric_lists.items():
    avg_metrics[metric_name] = onp.mean(metric_list)
  return avg_metrics


def compute_masked_metrics(cs, pred_cs, num_modes, num_points,
                           metrics=["pairwise_accuracy", "pairwise_f1",
                                    "pairwise_micro_f1", "pairwise_macro_f1"]):
  batch_size = cs.shape[0]
  metric_lists = collections.defaultdict(list)
  for i in range(batch_size):
    for metric_name in metrics:
      m = METRIC_FNS[metric_name](pred_cs[i, :num_points[i]],
                                  cs[i, :num_points[i]], num_modes[i])
      metric_lists[metric_name].append(m)

  avg_metrics = {}
  for metric_name, metric_list in metric_lists.items():
    avg_metrics[metric_name] = onp.mean(metric_list)
  return avg_metrics


def compute_masked_baseline_metrics(xs, cs, num_modes, num_points,
                                    predict_fns=METHODS,
                                    metrics=[
                                        "pairwise_accuracy", "pairwise_f1",
                                        "pairwise_micro_f1", "pairwise_macro_f1"
                                    ]):
  batch_size = xs.shape[0]
  metric_lists = collections.defaultdict(lambda: collections.defaultdict(list))
  for i in range(batch_size):
    for name, predict_fn in predict_fns.items():
      predicted_cs = predict_fn(xs[i, :num_points[i]], num_modes[i])
      for metric_name in metrics:
        m = METRIC_FNS[metric_name](predicted_cs,
                                    cs[i, :num_points[i]],
                                    num_modes[i])
        metric_lists[name][metric_name].append(m)

  avg_metrics = collections.defaultdict(dict)
  for method_name, metric_dict in metric_lists.items():
    for metric_name, metric_list in metric_dict.items():
      avg_metrics[method_name][metric_name] = onp.mean(metric_list)
  return avg_metrics
