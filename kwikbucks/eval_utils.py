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

"""Evaluation utils clustering with weak and strong signals."""


class PrecRecallF1:
  """Computes Precision, Recall, and F1 score."""

  def __init__(self):
    self.tp = 0.0
    self.tn = 0.0
    self.fp = 0.0
    self.fn = 0.0

  def update_confusion_matrix(self, pred, label):
    if pred and label:
      self.tp += 1
    if pred and not label:
      self.fp += 1
    if not pred and label:
      self.fn += 1
    if not pred and not label:
      self.tn += 1

  def compute_prec_recall_f1(self, eps=0.00001):
    self.prec = self.tp / (self.tp + self.fp + eps)
    self.recall = self.tp / (self.tp + self.fn + eps)
    self.f1 = (2 * self.prec * self.recall) / (self.prec + self.recall + eps)

  def print_(self):
    print(f'Precision: {round(self.prec, 3)}')
    print(f'Recall: {round(self.recall, 3)}')
    print(f'F1 Score: {round(self.f1, 3)}')

  def compute_and_print(self):
    self.compute_prec_recall_f1()
    self.print_()


class Evaluator:
  """Evaluates clusterings based on labels."""

  def __init__(self):
    self.prec_recall_f1 = PrecRecallF1()

  def compute_prec_recall_f1(self, clusters, dataset):
    """Calculates correlation clustering objective and precision / recall of positive edges given clusters."""
    for ex_id1, ex_id2, same_label in dataset.pair_same_cluster_iterator():
      same_cluster = clusters.same_cluster(ex_id1, ex_id2)
      self.prec_recall_f1.update_confusion_matrix(same_cluster, same_label)
    self.prec_recall_f1.compute_prec_recall_f1()
    self.cc_objective = self.prec_recall_f1.fp + self.prec_recall_f1.fn

  def compute_prec_recall_f1_graph(self, clusters, dataset):
    """Above function specialized for graph datasets."""
    # Keep track of positive edges within and across clusters.
    num_positive_edges_across_clusters = 0
    num_positive_edges_within_clusters = 0
    # Get edges of graph.
    rows, columns = dataset.strong_signal.nonzero()
    # Loop over edges.
    for row, col in zip(rows, columns):
      if row > col:
        assignment_row = clusters.assignments[row]
        assignment_col = clusters.assignments[col]
        # Increment quantities depending on assignments of edge vertices.
        if assignment_row == assignment_col:
          num_positive_edges_within_clusters += 1
        else:
          num_positive_edges_across_clusters += 1
    # Now calculate number of negative eddges within clusters.
    num_intra_cluster_edges = 0
    c_id_to_ex_id_dict = clusters.c_id_to_ex_id()
    # Get size of each cluster.
    clusters_to_size = {
        c_id: len(c_id_to_ex_id_dict[c_id]) for c_id in c_id_to_ex_id_dict
    }
    # Get the total number of inter cluster edges and subtract out
    # the number of positive edges within to get number of negative
    # edges within.
    for c_id in clusters_to_size:
      num_current_internal_edges = (
          clusters_to_size[c_id] * (clusters_to_size[c_id] - 1) / 2
      )
      num_intra_cluster_edges += num_current_internal_edges
    num_negative_edges_within_clusters = (
        num_intra_cluster_edges - num_positive_edges_within_clusters
    )
    num_all_pairwise_edges = (
        dataset.num_examples * (dataset.num_examples - 1) / 2
    )
    self.cc_objective = (
        num_positive_edges_across_clusters + num_negative_edges_within_clusters
    )
    self.prec_recall_f1.tp = num_positive_edges_within_clusters
    self.prec_recall_f1.fp = num_negative_edges_within_clusters
    self.prec_recall_f1.fn = num_positive_edges_across_clusters
    self.prec_recall_f1.tn = num_all_pairwise_edges - (
        self.prec_recall_f1.tp + self.prec_recall_f1.fp + self.prec_recall_f1.fn
    )
    self.prec_recall_f1.compute_prec_recall_f1()

  def print_cluster_stats(self, clusters):
    print(f'Largest cluster size: {clusters.largest_cluster_size()}')
    print(f'Correlation clustering objective: {self.cc_objective}')
    print(f'Positive edges recall: {self.prec_recall_f1.recall}')
    print(f'Positive edges precision: {self.prec_recall_f1.prec}')

  def evaluate(self, clusters, dataset):
    if dataset.is_graph or dataset.is_sparse:
      self.compute_prec_recall_f1_graph(clusters, dataset)
    else:
      self.compute_prec_recall_f1(clusters, dataset)
    self.print_cluster_stats(clusters)
    return (
        self.cc_objective,
        self.prec_recall_f1.recall,
        self.prec_recall_f1.prec,
    )
