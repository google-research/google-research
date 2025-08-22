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

"""Example DP PPR computation."""

from absl import app
from absl import flags
import networkx as nx
import numpy as np
import scipy.sparse as sps

from private_personalized_pagerank import io
from private_personalized_pagerank import metrics
from private_personalized_pagerank import ppr
from private_personalized_pagerank import privacy

_PPR_ALPHA = flags.DEFINE_float('ppr_alpha', 0.15, 'Input PPR alpha')
_PPR_NUM_ITERATIONS = flags.DEFINE_integer(
    'ppr_num_iterations',
    100,
    'Number of push iterations for PPR approximation.',
)
_DP_EPSILON = flags.DEFINE_float('dp_epsilon', 1, 'DP epsilon.')
_SIGMA = flags.DEFINE_float('sigma', 1e-6, 'DP PPR sigma parameter.')
_TOP_K = flags.DEFINE_integer('top_k', 10, 'Report recall@k for top_k results.')

FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv
  # We use the karate club graph as an example.
  # It is very sparse - not a very good fit for DP PPR accuracy-wise.
  # Nevertheless, we use it as an example.
  # Note that the in the paper we ran experiments in parallel on many machines.
  graph = nx.karate_club_graph()
  adjacency = sps.csr_matrix(nx.adjacency_matrix(graph))
  adjacency = io.preprocess_adjacency(adjacency)
  top_k = _TOP_K.value
  recalls = []
  for node in range(graph.number_of_nodes()):
    print(f'Computing DP PPR for node {node}...')
    dp_ppr = privacy.compute_dp_ppr(
        adjacency,
        node,
        epsilon=_DP_EPSILON.value,
        sigma=_SIGMA.value,
        num_iter=_PPR_NUM_ITERATIONS.value,
        alpha=_PPR_ALPHA.value,
        is_joint_dp=True,
    )
    non_dp_ppr = ppr.ppr_power_iteration(
        adjacency,
        node,
        num_iter=_PPR_NUM_ITERATIONS.value,
        alpha=_PPR_ALPHA.value,
    )
    results = metrics.score_approximation(dp_ppr, non_dp_ppr, ks=(top_k,))
    recalls.append(results.get(f'Recall@{top_k}', np.nan))
  print(f'Average Recall@{top_k}: {np.mean(recalls):.3f}')


if __name__ == '__main__':
  app.run(main)
