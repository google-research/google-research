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

"""Clustering with weak and strong signals."""

import os
import sys

from absl import app
from absl import flags
import gin
import numpy as np

from kwikbucks import data_utils
from kwikbucks import eval_utils
from kwikbucks import model_utils

_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', 'configs/config.gin', 'Gin files.'
)
_GIN_PARAM = flags.DEFINE_multi_string(
    'gin_param', None, 'Gin parameter bindings.'
)
_DATASET_PATH = flags.DEFINE_string('dataset_path', '', 'Dataset path.')
_DATASET_NAME = flags.DEFINE_string('dataset_name', 'cora', 'Dataset name.')
_CORRELATION_CLUSTERING_ALGORITHM = flags.DEFINE_string(
    'correlation_clustering_algorithm',
    'qwick_cluster_using_ordering',
    'Algorithm to use for correlation clustering.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', 'CorrelationClustering', 'Model to use.'
)
_NUM_PIVOTS = flags.DEFINE_integer(
    'num_pivots', 200, 'Number of pivots to use.'
)
_QUERY_BUDGET_FOR_ALGORITHM = flags.DEFINE_float(
    'query_budget_for_algorithm',
    10000,
    'Query budget for clustering algorithm.',
)
_QUERY_BUDGET_FOR_MERGE = flags.DEFINE_float(
    'query_budget_for_merge', 1000, 'Query budget for merging post processing.'
)
_CLUSTER_USE_MERGE_POST_PROCESSING = flags.DEFINE_bool(
    'cluster_use_merge_post_processing',
    True,
    'Variable for if we want to use merge post processing.',
)
_BOOST_BY_NEIGHBORS_ASSIGNMENT = flags.DEFINE_bool(
    'boost_by_neighbors_assignment',
    True,
    'Using neighborhood statistics for weak signal ordering.',
)
_ORDERING_STYLE = flags.DEFINE_string(
    'ordering_style',
    'weak_signal',
    'Rank all possible edge queries using weak signal ordering',
)
_SPECTRAL_CLUSTERING_N_CLUSTERS = flags.DEFINE_integer(
    'spectral_clustering_n_clusters',
    5,
    'Number of clusters to use in spectral clustering.',
)
_IS_SPARSE = flags.DEFINE_bool(
    'is_sparse', False, 'If the weak signal of dataset is a sparse csr matrix.'
)
_MAX_NEIGHBORS_TO_QUERY = flags.DEFINE_integer(
    'max_neighbors_to_query',
    100,
    'Maximum number of strong signal queries to make.',
)
_ORDER_NON_PIVOTS = flags.DEFINE_bool(
    'order_non_pivots',
    True,
    'If we want to order non-pivots using weak signal similarities to pivots.',
)


def set_gin_params():
  """Set gin parameter values from flag values."""
  dataset_path = (
      _DATASET_PATH.value
      if _DATASET_PATH.value
      else f'{os.getcwd()}/kwikbucks/data'
  )
  with gin.unlock_config():
    gin.bind_parameter('Dataset.path', dataset_path)
    gin.bind_parameter('Dataset.name', _DATASET_NAME.value)
    gin.bind_parameter('create_model.name', _MODEL_NAME.value)
    if _MODEL_NAME.value == 'CorrelationClustering':
      gin.bind_parameter(
          'correlation_clustering.cluster.algorithm',
          _CORRELATION_CLUSTERING_ALGORITHM.value,
      )
      gin.bind_parameter('set_num_pivots.num_pivots', _NUM_PIVOTS.value)
      gin.bind_parameter(
          'correlation_clustering.cluster.budget_algorithm',
          _QUERY_BUDGET_FOR_ALGORITHM.value,
      )
      gin.bind_parameter(
          'correlation_clustering.cluster.budget_merge',
          _QUERY_BUDGET_FOR_MERGE.value,
      )
      gin.bind_parameter(
          'correlation_clustering.cluster.use_merge_post_processing',
          _CLUSTER_USE_MERGE_POST_PROCESSING.value,
      )
      gin.bind_parameter(
          'qwick_cluster_using_ordering.ordering_style', _ORDERING_STYLE.value
      )
      gin.bind_parameter(
          'assign_to_pivots_given_ordering.boost_by_neighbors_assignment',
          _BOOST_BY_NEIGHBORS_ASSIGNMENT.value,
      )
      gin.bind_parameter(
          'qwick_cluster_using_ordering.max_neighbors_to_query',
          _MAX_NEIGHBORS_TO_QUERY.value,
      )
      gin.bind_parameter(
          'qwick_cluster_using_ordering.order_non_pivots',
          _ORDER_NON_PIVOTS.value,
      )
    elif _MODEL_NAME.value == 'QwickClusterSampled':
      gin.bind_parameter(
          'qwick_cluster_sampled.cluster.budget_algorithm',
          _QUERY_BUDGET_FOR_ALGORITHM.value,
      )
    elif _MODEL_NAME.value == 'QwickClusterKnn':
      gin.bind_parameter(
          'qwick_cluster_knn.cluster.budget_algorithm',
          _QUERY_BUDGET_FOR_ALGORITHM.value,
      )
    elif _MODEL_NAME.value == 'SpectralClusteringKnn':
      gin.bind_parameter(
          'spectral_clustering_knn.cluster.budget_algorithm',
          _QUERY_BUDGET_FOR_ALGORITHM.value,
      )
      gin.bind_parameter(
          'spectral_clustering_knn.cluster.n_clusters',
          _SPECTRAL_CLUSTERING_N_CLUSTERS.value,
      )
    else:
      print('model_name must be defined.')
      exit()


def run_experiment(num_trials):
  """Running experiment."""
  dataset = data_utils.Dataset()
  if _IS_SPARSE.value:
    dataset.is_sparse = True
  cc_objective_values = np.zeros(num_trials)
  recall_values = np.zeros(num_trials)
  precall_values = np.zeros(num_trials)

  for trial in range(num_trials):
    with gin.unlock_config():
      gin.bind_parameter('set_random_seed.seed', trial)
    model = model_utils.create_model()
    clusters = model.cluster(dataset)
    evaluator = eval_utils.Evaluator()
    cc_objective_trial, recall_trial, prec_trial = evaluator.evaluate(
        clusters, dataset
    )
    cc_objective_values[trial] = cc_objective_trial
    recall_values[trial] = recall_trial
    precall_values[trial] = prec_trial
  return cc_objective_values, recall_values, precall_values


def main(unused_argv):
  sys.setrecursionlimit(20000)
  gin.add_config_file_search_path(f'{os.getcwd()}/kwikbucks/')
  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_PARAM.value)
  set_gin_params()
  run_experiment(15)


if __name__ == '__main__':
  app.run(main)
