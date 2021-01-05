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

r"""Implementation of the Fair Correlation Clustering AISTATS2020 Paper.

===============================

This is the implementation accompanying the AISTATS 2020 paper,
[_Fair Correlation Clustering_](https://arxiv.org/abs/2002.02274).


Citing
------
If you find _Fair Correlation Clustering_ useful in your research, we ask that
you cite the following paper:

> Ahmadian, S., Epasto, A., Mahdian, M., Kumar, R. (2020).
> Fair Correlation Clustering.
> In _AISTATS_.

    @inproceedings{ahmadian2020fair,
     author={Ahmadian, Epasto, Mahdian, Kumar}
     title={Fair Correlation Clustering},
     booktitle = {AISTATS},
     year = {2020},
    }

Example execution
------
python3 -m fair_correlation_clustering.correlation_clustering \
  --input_graph=${graph} \
  --input_color_mapping=${color_mapping} \
  --output_results=${results}

Where ${graph} is the path to a text file containing the graph,
${color_mapping} is the path to the color mapping and ${results} is the path
to the output file with the results.


As in fair clustering problems nodes are labeled with a color. We assume colors
to be intergers from 0 to c-1, for c number of colors. The node ids must be
consecutive integers in 0 ... n-1.

The input of the color mapping is a file where for each node there is row with
id color
e.g.
0 0
1 1
2 2
3 2
represents that node 0 has color 0, node 1 has color 1 and node 2 has color 2
and 3 has color 2. All nodes must have one color associated.


In this library, we assume that the graph is a _complete_ unweighted signed
graph.
That means that between each pair of nodes there is an edge which is either
positive or negative.
In the input we represent the graph by listing the positive edges only,
with the assumption that all missing edges are negative.
The graph input format is a text file containing one (positive)
edge per row represented as its pair of node ids. The graph is supposed to
be undirected. The node ids must be consecutive integers in 0 ... n-1.
For instance the file:
0 1
1 2
2 0
represents the positive triangle 0, 1, 2 and contains the implicit negative
edges 0 3, 1 3, 2 3  (since the graph has 4 nodes).

Notice that as we assume that the graph is complete, the code not optimized for
sparse graphs and it uses n^2 memory to store explictly all positive and
negative edges.

The output results format is a json file containing a pandas dataframe with the
results of the experiment.
"""

import collections
import datetime
import random
from absl import app
from absl import flags
from absl import logging
from correlation_clustering.baselines import BaselineAllTogether
from correlation_clustering.baselines import BaselineRandomFairEqual
from correlation_clustering.baselines import BaselineRandomFairOneHalf
from correlation_clustering.correlation_clustering_solver import LocalSearchAlgorithm
from correlation_clustering.correlation_clustering_solver import PivotAlgorithm
from correlation_clustering.utils import ClusterIdMap
from correlation_clustering.utils import CorrelationClusteringError
from correlation_clustering.utils import FractionalColorImbalance
from correlation_clustering.utils import PairwiseFairletCosts
import networkx as nx
import pandas as pd


flags.DEFINE_string(
    'input_graph', None,
    'The input graph path as a text file containing one (positive) edge per '
    'row, as the two node ids u v of the edge separated by a whitespace. '
    'The graph is assumed to be undirected. For example the '
    'file:\n0 1\n1 2\n0 2\n represents the triangle 0 1 2.')

flags.DEFINE_string(
    'input_color_mapping', None,
    'The input color mapping path as a text file containing one node color pair'
    ' per row, separated by a whitespace. For example the '
    'file:0 0\n 1 1\n 2 2\n 3 2\n represents the mapping 0 -> 0, 1 -> 1, '
    '2 -> 2, 3 -> 2. Colors must be intergers in [0, c-1].')

flags.DEFINE_string('output_results', None,
                    'output path for the json file containing the results.')

flags.DEFINE_integer('seed', 12838123, 'seed for random number generator.')

flags.DEFINE_integer('tries', 2, 'number of times the algorithms are run.')

FLAGS = flags.FLAGS


def ReadInput(input_file_graph, input_file_colors):
  """Read the graph and the color map.

  Read the graph file and color mapping and output a nx.Graph.

  Args:
    input_file_graph: text file, each line representing the positive edges as
    tab separated pairs of ints u v.
    input_file_colors: a text file, each line representing the color mapping as
    tab separated pairs of ints id color.

  Returns:
    The graph in nx.Graph format where nodes are consecutive integers and all
    pairs of nodes have an edge and the 'color' attribute is set of a each node.
    This is the standard format assumed in the library.
  """
  node_id = collections.defaultdict(int)
  graph = nx.Graph()
  with open(input_file_colors, 'r') as in_file:
    for row in in_file:
      row = row.split('\t')
      u = row[0].strip()
      c = int(row[1].strip())
      if u not in node_id:
        node_id[u] = len(node_id)
        graph.add_node(node_id[u])
      graph.nodes[node_id[u]]['color'] = c

  with open(input_file_graph, 'r') as in_file:
    for row in in_file:
      row = row.split('\t')
      u = row[0].strip()
      v = row[1].strip()
      if u == v: continue
      assert u in node_id
      assert v in node_id
      graph.add_edge(node_id[u], node_id[v], weight=+1)

  for n1 in graph.nodes():
    for n2 in graph.nodes():
      if n1 < n2 and not graph.has_edge(n1, n2):
        graph.add_edge(n1, n2, weight=-1)
  return graph


def CorrelationClusteringOneHalfAlgorithm(graph):
  """Algorithm for correlation clustering with fairness constraint, alpha = 1/2.

  The algorithm solves approximately correlation clustering with fairness
  constraints, for the special case of alpha=1/2 balancedness (i.e., each color
  is at most 1/2 of the cluster).
  For simplicity we use the matching based algorithm which assumes that a
  feasible solution consisting of fairlets of size 2 is possible
  (i.e. no color is >1/2 of nodes and nodes are even).

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'.

  Returns:
    A list of lists represeting the clusters. Each entry in the list is a
    cluster.
  """

  # First, the fairlet decomposition problem is solved.
  fairlets = FairletDecompositionOneHalf(graph)
  # Then, we obtain the compressed graph.
  compressed_graph, fairlet_dict = CompressGraph(graph, fairlets)
  # Then, we run local search heuristic algorithm over compressed graph.
  solution = LocalSearchAlgorithm(compressed_graph)
  # Finally, the solution of the compressed graph is lifted to the original
  # problem.
  return UnpackSolution(solution, fairlet_dict)


def CorrelationClusteringEqualRepresentation(graph):
  """Algorithm for correlation clustering with fairness constraint, alpha = 1/c.

  The algorithm solves approximately correlation clustering with fairness
  constraints, for the special case of equal color representation.
  It is assumed that in the graph all colors are balanced.

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'. Only positive weight edges are represented. All missing
      edges are assumed to be negative.

  Returns:
    A list of lists represeting the clusters. Each entry in the list is a
    cluster.
  """
  # First, the fairlet decomposition problem is solved.
  fairlets = FairletDecompositionEqualRepresentation(graph)
  # Then, we obtain the compressed graph.
  compressed_graph, fairlet_dict = CompressGraph(graph, fairlets)
  # Then, we run local search heuristic algorithm over compressed graph.
  solution = LocalSearchAlgorithm(compressed_graph)
  # Finally, the solution of the compressed graph is lifted to the original
  # problem.
  return UnpackSolution(solution, fairlet_dict)


def FairletDecompositionOneHalf(graph):
  """Algorithm for fairlet decomposition for the alpha = 1/2 case.

  The algorithm solves the fairlet decomposition problem for the 1/2 case.
  We assume there is a feasible solution with fairlets of size 2.

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'. Only positive weight edges are represented. All missing
      edges are assumed to be negative.

  Returns:
    A list of lists represeting the fairlets. Each entry in the list is a
    farilet.
  """
  # Compute a matrix with the fairlet cost for each pair of  nodes.
  distance_matrix = PairwiseFairletCosts(graph)
  # Create a graph with edges between different color nodes with the
  # fairlet cost as weight.
  matching_graph = nx.Graph()
  # Notice that internally we assume the graph has consecutive integers.
  for i in range(len(distance_matrix)):
    for j in range(i + 1, len(distance_matrix)):
      if graph.nodes[i]['color'] != graph.nodes[j]['color']:
        # using -distance for weight as the algorithm used finds the max
        # weight cardinality matching.
        matching_graph.add_edge(i, j, weight=-distance_matrix[i][j])
  matching = nx.max_weight_matching(matching_graph, maxcardinality=True)
  return [list(m) for m in matching]


def FairletDecompositionEqualRepresentation(graph):
  """Algorithm for fairlet decomposition for the equal representation case.

  The algorithm solves the fairlet decomposition problem for the 1/c case.
  We assume there is a feasible solution with fairlets of size C.

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'. Only positive weight edges are represented. All missing
      edges are assumed to be negative.

  Returns:
    A list of lists represeting the fairlets. Each entry in the list is a
    farilet.
  """

  # First we check that the problem is feasible (equal representation of each
  # color in the dataset).
  # Map between color and nodes of that colors.
  color_nodes = collections.defaultdict(list)
  for u, d in graph.nodes(data=True):
    color_nodes[d['color']].append(u)
  sizes = [len(l) for l in color_nodes.values()]
  assert max(sizes) == min(sizes)
  assert len(sizes) >= 2

  # Computes the pairwise fairlet cost matrix.
  distance_matrix = PairwiseFairletCosts(graph)
  # Pick random ordering of colors for the computing a repeated matching
  # between pairs of nodes of different colors.
  color_nodes = list(color_nodes.values())
  random.shuffle(color_nodes)
  # Initializes one fairlet for each node of color in position 0.
  color_a_nodes = set(color_nodes[0])
  fairlets = collections.defaultdict(list)
  node_to_fairlet_id = {}
  for node_a in color_a_nodes:
    fairlets[node_a].append(node_a)
    node_to_fairlet_id[node_a] = node_a

  # Matches nodes of color color_id with nodes of color color_id - 1.
  for color_id  in range(1, len(color_nodes)):
    matching_graph = nx.Graph()
    previous_assigned = set(color_nodes[color_id - 1])
    color_b_nodes = color_nodes[color_id]
    for i in previous_assigned:
      for j in color_b_nodes:
        matching_graph.add_edge(i, j, weight=-distance_matrix[i][j])
    matching = nx.max_weight_matching(matching_graph, maxcardinality=True)
    assert len(matching) == len(color_a_nodes)
    for a, b in matching:
      assigned_node = a if a in previous_assigned else b
      unassigned_node = b if a in previous_assigned else a
      fairlets[node_to_fairlet_id[assigned_node]].append(unassigned_node)
      node_to_fairlet_id[unassigned_node] = node_to_fairlet_id[assigned_node]

  return [nodes for nodes in fairlets.values()]


def CompressGraph(graph, fairlets):
  """Algorithm for graph compression algorithm based on fairlets.

  The algorithm compress the graph representing each fairlet a single node as
  described in the paper.

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'. Only positive weight edges are represented. All missing
      edges are assumed to be negative.
    fairlets: The fairlets as a list of lists.

  Returns:
    A graph in networkx format and the mapping of node ids in the new graph to
      fairlet ids.
  """
  fairlet_dict = {}
  for i, nodes in enumerate(fairlets):
    for node in nodes:
      fairlet_dict[node] = i
  # Computes the positive - negatives weights between the two fairlets
  positives = collections.defaultdict(int)
  negatives = collections.defaultdict(int)
  for u, v, d in graph.edges(data=True):
    if fairlet_dict[u] != fairlet_dict[v]:
      m1 = fairlet_dict[
          u] if fairlet_dict[u] < fairlet_dict[v] else fairlet_dict[v]
      m2 = fairlet_dict[
          v] if fairlet_dict[u] < fairlet_dict[v] else fairlet_dict[u]
      if d['weight'] > 0:
        positives[(m1, m2)] += 1
      elif d['weight'] < 0:
        negatives[(m1, m2)] += 1
  compressed_graph = nx.Graph()
  pairs = set(positives.keys()) | set(negatives.keys())
  for u, v in pairs:
    if positives[(u, v)] >= negatives[(u, v)]:
      compressed_graph.add_edge(u, v, weight=positives[(u, v)])
    else:
      compressed_graph.add_edge(u, v, weight=-negatives[(u, v)])
  assert len(compressed_graph.nodes()) == len(fairlets)
  return compressed_graph, fairlet_dict


def UnpackSolution(compressed_solution, fairlet_dict):
  """Algorithm for unpacking the compressed graph solution.

  Given a solution to the compressed graph and the fairlet_id dictionary, unpack
  the soluton by associating all nodes in the fairlet with the cluster to which
  their id belongs.
  Args:
    compressed_solution: the solution of the compressed graph.
    fairlet_dict: a map from fairlet id to original node

  Returns:
    A solultion for the original graph.
  """
  compressed_clust_assignment = ClusterIdMap(compressed_solution)
  new_clusters = collections.defaultdict(list)
  for node, fairlet_id in fairlet_dict.items():
    assert fairlet_id in compressed_clust_assignment
    clust_id = compressed_clust_assignment[fairlet_id]
    new_clusters[clust_id].append(node)
  return list(new_clusters.values())


def RunEval(graph, num_colors, algorithm, algo_label, seed):
  """Run the evalution of a given correlation clustering algorithm.

  Runs the function algorithm over graph to obtain a solution and then evaluates
  the correlation clustering error, the imbalance for equal representation and
  for majority representation
  Args:
    graph: the graph
    num_colors: number of colors in the graph
    algorithm: the algorithm to call
    algo_label: a label for the algorithm
    seed: a seed used for randomness

  Returns:
    A dictionary with the results of the evaluation.
  """

  logging.info('[RUNNING] algorithm %s with seed %d', algo_label, seed)
  result = {'algo': algo_label, 'colors': num_colors, 'seed': seed}
  random.seed(seed)
  start_time = datetime.datetime.now()
  solution = algorithm(graph)
  end_time = datetime.datetime.now()
  delta = end_time - start_time
  result['time'] = delta.total_seconds()
  all_elems = set()
  for clus in solution:
    for c in clus:
      all_elems.add(c)
  assert all_elems == set(graph.nodes())
  assert sum(len(clust) for clust in solution) == graph.number_of_nodes()
  result['error'] = CorrelationClusteringError(graph, solution)
  result['onehalf_imbalance'] = FractionalColorImbalance(graph, solution, 0.5)
  if num_colors > 2:
    result['equal_imbalance'] = FractionalColorImbalance(
        graph, solution, 1.0 / num_colors)
  logging.info('[DONE] results: %s', result)
  return result


def RunAnalysis(graph, num_colors, seed, tries, outfile):
  """Run all algorithms over a given graph and output the results.

  Runs all the algorithms over a graph evaluates them.
  Args:
    graph: the graph
    num_colors: number of colors in the graph
    seed: a seed used for randomness
    tries: how many times to call the algorithms
    outfile: path to a file.

  Returns:
    A dictionary with the results of the evaluation.
  """
  results = []
  algorithms = [(PivotAlgorithm, 'pivot'),
                (LocalSearchAlgorithm, 'local_search'),
                (CorrelationClusteringOneHalfAlgorithm, 'onehalf_fair'),
                (BaselineAllTogether, 'one_cluster'),
                (BaselineRandomFairOneHalf, 'random_onehalf_fair')]
  if num_colors > 2:
    algorithms.extend([(CorrelationClusteringEqualRepresentation, 'equal_fair'),
                       (BaselineRandomFairEqual, 'random_equal_fair')])

  for t in range(tries):
    for algo, algo_label in algorithms:
      seed_for_run = seed + t
      results.append(RunEval(graph, num_colors, algo, algo_label, seed_for_run))

  df = pd.DataFrame(results)
  with open(outfile, 'w') as out_file:
    df.to_json(out_file)


def main(argv=()):
  del argv  # Unused.
  graph = ReadInput(FLAGS.input_graph, FLAGS.input_color_mapping)
  num_colors = len(set([graph.nodes[n]['color'] for n in graph.nodes()]))
  RunAnalysis(graph, num_colors, FLAGS.seed, FLAGS.tries, FLAGS.output_results)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_graph')
  flags.mark_flag_as_required('input_color_mapping')
  flags.mark_flag_as_required('output_results')
  app.run(main)
