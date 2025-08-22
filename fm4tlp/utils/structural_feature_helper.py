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

"""Helper functions for generating structural features for structural augmentation and structural mapping of memeory embeddings..

For each batch in train, validation, test, or warmstart datasets, we generate
structural features for each node. Thes tructural features include degree, betweeness centrality, clustering coefficient, positional encodings, and orbit counts.
These features are used to generate structural augmentation and structural mapping.
"""

import collections
import os
import types

import graph_tool as gt
from graph_tool import generation as gt_generation
from graph_tool import topology as gt_topology
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import torch
import torch_geometric
from torch_geometric.utils import remove_self_loops


def chunker(df, size):
  r"""Splits a dataframe into chunks of a certain size.

  Parameters:

    df: the dataframe to be split
    size: the size of each chunk
  Returns:
    df_list: a list of dataframes, each of size `size`
  """
  df_list = []
  for pos in range(0, len(df), size):
    df_list.append(df.iloc[pos:pos + size])
  return df_list


def generate_graph_features(G):
  r"""Generates simple structural graph features for each node in a graph.

  Parameters:

    G: the graph to be featurized
  Returns:
    feature_dict: a dictionary of features for each node in the graph
  """
  bet_centr_dict = nx.betweenness_centrality(G)
  feature_dict = {}
  for node in G.nodes():
    deg = G.degree(node)
    closeness_centrality = nx.closeness_centrality(G,node)
    betw_centr = bet_centr_dict[node]
    clustering = nx.clustering(G,node)
    feat = np.array([deg, closeness_centrality, betw_centr, clustering])
    feature_dict[node] = feat
  return feature_dict


def lap_positional_encoding(g, pos_enc_dim):
  r"""Graph positional encoding v/ Laplacian eigenvectors

  Parameters:
    g: the graph to be featurized
    pos_enc_dim: the number of positional encoding dimensions
  Returns:
    pos_enc_dict: a dictionary of positional encodings for each node in the graph
  """
  # Laplacian
  A = nx.adjacency_matrix(g).astype(np.float32)
  degrees = list(dict(g.degree()).values())
  N = scipy.sparse.diags(np.array(degrees) ** -0.5, dtype=float)
  L = scipy.sparse.eye(g.number_of_nodes()) - N * A * N

  # Eigenvectors with numpy
  EigVal, EigVec = np.linalg.eig(L.toarray())
  idx = EigVal.argsort()
  _, EigVec = EigVal[idx], np.real(EigVec[:, idx])

  pos_enc_emb = EigVec[:, 1:pos_enc_dim + 1]
  if pos_enc_emb.shape[-1] < pos_enc_dim:
    offset = pos_enc_dim - pos_enc_emb.shape[-1]
    pos_enc_emb = np.concatenate((pos_enc_emb, np.zeros((pos_enc_emb.shape[0], offset))), axis=-1)

  pos_enc_dict = {}
  for i, node in enumerate(g.nodes()):
    pos_enc_dict[node] = pos_enc_emb[i]

  return pos_enc_dict


def init_positional_encoding(g, pos_enc_dim):
  r"""Initializing positional encoding with RWPE

  Parameters:
    g: the graph to be featurized
    pos_enc_dim: the number of positional encoding dimensions
  Returns:
    PE_dict: a dictionary of positional encodings for each node in the graph
  """
  # Geometric diffusion features with Random Walk
  A = nx.adjacency_matrix(g)
  degrees = list(dict(g.degree()).values())
  Dinv = scipy.sparse.diags(np.array(degrees).clip(1) ** -1.0, dtype=float)
  RW = A * Dinv
  M = RW

  # Iterate
  nb_pos_enc = pos_enc_dim
  PE = [M.diagonal()]
  M_power = M
  for _ in range(nb_pos_enc - 1):
    M_power = M_power * M
    PE.append(M_power.diagonal())
  PE = np.stack(PE, axis=-1)

  PE_dict = {}
  for i, node in enumerate(g.nodes()):
    PE_dict[node] = PE[i]

  return PE_dict


def get_custom_edge_list(ks, substructure_type=None, filename=None):
  r"""Instantiates a list of `edge_list`s representing substructures

  of type `substructure_type` with sizes specified by `ks`.
  Parameters:
    ks: a list of integers specifying the sizes of the substructures to be
      generated
    substructure_type: the type of substructure to be generated. If None,
      then the substructures will be read from the file specified by
      `filename`.
    filename: the name of the file from which to read the substructures. If
      None, then the substructures will be generated using the
      `substructure_type`.
  Returns:
    edge_lists: a list of edge lists representing the substructures.
  """
  if substructure_type is None and filename is None:
    raise ValueError(
        'You must specify either a type or a filename where to read'
        ' substructures from.'
    )
  edge_lists = []
  for k in ks:
    if substructure_type is not None:
      graphs_nx = getattr(nx, substructure_type)(k)
    else:
      graphs_nx = nx.read_graph6(
          os.path.join(filename, 'graph{}c.g6'.format(k))
      )
    if isinstance(graphs_nx, list) or isinstance(
        graphs_nx, types.GeneratorType
    ):
      edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
    else:
      edge_lists.append(list(graphs_nx.edges))
  return edge_lists


def automorphism_orbits(edge_list, print_msgs=False, **kwargs):
  r"""Computes the automorphism group of a given substructure and the orbit

  partition of the vertices of the substructure.
  Parameters:
    edge_list: the edge list of the substructure
    print_msgs: whether to print messages during the computation
    **kwargs: additional arguments to be passed to the `subgraph_isomorphism`
      function
  Returns:
    graph: the graph representing the substructure
    orbit_partition: the orbit partition of the vertices of the substructure
    orbit_membership: the orbit membership of the vertices of the substructure
  """
  ##### vertex automorphism orbits #####

  directed = kwargs['directed'] if 'directed' in kwargs else False

  graph = gt.Graph(directed=directed)
  graph.add_edge_list(edge_list)
  gt_generation.remove_self_loops(graph)
  gt_generation.remove_parallel_edges(graph)

  # compute the vertex automorphism group
  aut_group = gt_topology.subgraph_isomorphism(
      graph, graph, induced=False, subgraph=True, generator=False
  )

  orbit_membership = {}
  for v in graph.get_vertices():
    orbit_membership[v] = v

  # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
  for aut in aut_group:
    for original, vertex in enumerate(aut):
      role = min(original, orbit_membership[vertex])
      orbit_membership[vertex] = role

  orbit_membership_list = [[], []]
  for vertex, om_curr in orbit_membership.items():
    orbit_membership_list[0].append(vertex)
    orbit_membership_list[1].append(om_curr)

  # make orbit list contiguous (i.e. 0,1,2,...O)
  _, contiguous_orbit_membership = np.unique(
      orbit_membership_list[1], return_inverse=True
  )
  orbit_membership = {
      vertex: contiguous_orbit_membership[i]
      for i, vertex in enumerate(orbit_membership_list[0])
  }

  orbit_partition = {}
  for vertex, orbit in orbit_membership.items():
    orbit_partition[orbit] = (
        [vertex]
        if orbit not in orbit_partition
        else orbit_partition[orbit] + [vertex]
    )

  aut_count = len(aut_group)

  if print_msgs:
    print('Orbit partition of given substructure: {}'.format(orbit_partition))
    print('Number of orbits: {}'.format(len(orbit_partition)))
    print('Automorphism count: {}'.format(aut_count))

  return graph, orbit_partition, orbit_membership, aut_count


def subgraph_isomorphism_vertex_counts(edge_index, **kwargs):
  r"""Computes the number of subgraph isomorphisms for each vertex in a graph.

    Parameters:
    edge_index: the edge index of the graph
    **kwargs: additional arguments to be passed to the `subgraph_isomorphism`
      function
  Returns:
    counts: a numpy array of counts for each vertex in the graph
  """
  ##### vertex structural identifiers #####

  subgraph_dict, induced, num_nodes = (
      kwargs['subgraph_dict'],
      kwargs['induced'],
      kwargs['num_nodes'],
  )
  directed = kwargs['directed'] if 'directed' in kwargs else False

  G_gt = gt.Graph(directed=directed)
  G_gt.add_edge_list(list(edge_index.transpose(1, 0).cpu().numpy()))
  gt_generation.remove_self_loops(G_gt)
  gt_generation.remove_parallel_edges(G_gt)

  # compute all subgraph isomorphisms
  sub_iso = gt_topology.subgraph_isomorphism(
      subgraph_dict['subgraph'],
      G_gt,
      induced=induced,
      subgraph=True,
      generator=True,
  )

  ## num_nodes should be explicitly set for the following edge case:
  ## when there is an isolated vertex whose index is larger
  ## than the maximum available index in the edge_index

  counts = np.zeros((num_nodes, len(subgraph_dict['orbit_partition'])))
  for sub_iso_curr in sub_iso:
    for i, node in enumerate(sub_iso_curr):
      # increase the count for each orbit
      counts[node, subgraph_dict['orbit_membership'][i]] += 1
  counts = counts / subgraph_dict['aut_count']

  return counts


class VertexAutomorphismCounter:
  r"""Class for computing the number of vertex automorphism counts for a given graph."""

  def __init__(
      self,
      k=3,
      id_types=[
          'cycle_graph',
          'path_graph',
          'complete_graph',
          'binomial_tree',
          'star_graph',
          'nonisomorphic_trees',
      ],
      directed=False,
      directed_orbits=False,
      induced=True,
  ):

    self._subgraph_dicts = collections.defaultdict(list)
    self._orbit_partition_sizes = collections.defaultdict(list)
    self._directed = directed
    self._directed_orbits = directed_orbits
    self._induced = induced
    self._k = k
    self._id_types = id_types

    for id_type in self._id_types:
      subgraph_edge_lists = get_custom_edge_list([self._k], id_type)
      for edge_list in subgraph_edge_lists:
        subgraph, orbit_partition, orbit_membership, aut_count = (
            automorphism_orbits(
                edge_list=edge_list,
                directed=self._directed,
                directed_orbits=self._directed_orbits,
            )
        )
        self._subgraph_dicts[id_type].append({
            'subgraph': subgraph,
            'orbit_partition': orbit_partition,
            'orbit_membership': orbit_membership,
            'aut_count': aut_count,
        })
        self._orbit_partition_sizes[id_type].append(len(orbit_partition))

  def get_automorphism_counts(self, torch_geo_graph, id_type):
    r"""Computes the number of vertex automorphism counts for a given graph.

    Parameters:

      torch_geo_graph: the graph to be featurized
      id_type: the type of the subgraph to be used for computing the counts
    Returns:
      counts: a numpy array of counts for each vertex in the graph
    """
    if torch_geo_graph.edge_index.shape[1] == 0:
      setattr(
          torch_geo_graph, 'degrees', torch.zeros((torch_geo_graph.num_nodes,))
      )
    else:
      setattr(
          torch_geo_graph,
          'degrees',
          torch_geometric.utils.degree(torch_geo_graph.edge_index[0]),
      )

    edge_index = remove_self_loops(torch_geo_graph.edge_index)[0]

    count_signals = []
    for subgraph_dict in self._subgraph_dicts[id_type]:
      kwargs = {
          'subgraph_dict': subgraph_dict,
          'induced': self._induced,
          'num_nodes': torch_geo_graph.num_nodes,
          'directed': self._directed,
      }
      counts = subgraph_isomorphism_vertex_counts(edge_index, **kwargs)
      count_signals.append(counts)
    return count_signals

  def plot_orbits_of_node(
      self,
      target_node,
      torch_geo_graph,
      nx_graph,
      id_type,
      auto_index=0
  ):
    r"""Plots the orbits of a given node in a graph.

    Parameters:

      target_node: the node to be plotted
      torch_geo_graph: the graph to be featurized
      nx_graph: the networkx graph to be featurized
      id_type: the type of the subgraph to be used for computing the counts
      auto_index: the index of the automorphism to be used for plotting
    """
    subgraph_dicts = self._subgraph_dicts[id_type]
    induced = self._induced
    directed = self._directed
    edge_index = remove_self_loops(torch_geo_graph.edge_index)[0]

    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index.transpose(1,0).cpu().numpy()))
    gt_generation.remove_self_loops(G_gt)
    gt_generation.remove_parallel_edges(G_gt)

    G_pos = nx.spring_layout(nx_graph, seed=12345)

    for subgraph_dict in subgraph_dicts:
      print(list(subgraph_dict['subgraph'].edges()))

      # compute all subgraph isomorphisms
      sub_iso = gt_topology.subgraph_isomorphism(
          subgraph_dict['subgraph'], G_gt,
          induced=induced, subgraph=True, generator=True)

      for sub_iso_curr in sub_iso:
        nodes = list(sub_iso_curr)
        color_map = []
        if nodes[auto_index] != target_node:
          continue
        for node in G_pos:
          if node in nodes:
            color_map.append('pink')
          else:
            color_map.append('yellow')
        nx.draw(nx_graph, node_color=color_map, pos=G_pos, with_labels=True)
        plt.show()

  def get_all_automorphism_counts(self, nx_graph):
    r"""Computes the number of vertex automorphism counts for a given graph.

    Parameters:

      nx_graph: the graph to be featurized
    Returns:
      automorphism_count_dict: a dictionary of counts for each vertex in the graph
    """
    torch_geo_graph = torch_geometric.utils.from_networkx(nx_graph)
    all_counts = []
    for id_type in self._id_types:
      counts_list = self.get_automorphism_counts(torch_geo_graph, id_type)
      for counts in counts_list:
        all_counts.append(counts)

    automorphism_count_dict = {}
    for i, node in enumerate(nx_graph.nodes()):
      automorphism_count_dict[node] = np.concatenate([all_counts[j][i] for j in range(len(all_counts))], axis=0)

    return automorphism_count_dict


def get_timebound_subgraph(
    G,
    nodes,
    current_ts,
    window = 365 * 86400
):
  r"""Gets a subgraph of G that contains all nodes in nodes and all edges that are within window of current_ts.

  Parameters:

    G: the graph to be featurized
    nodes: the nodes to be included in the subgraph
    current_ts: the current timestamp
    window: the window of time to be included in the subgraph
  Returns:
    G_sub: the subgraph of G that contains all nodes in nodes and all edges
    that are within window of current_ts.
  """
  G_sub = nx.Graph()
  G_sub.add_edges_from(
      list(set(
          [(u, v) for u, v, ts_dict in G.edges(nodes, data=True) if ts_dict['ts'] + window > current_ts]
      )))
  return G_sub
