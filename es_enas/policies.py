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

"""Contains special policies which use topologies to define neural networks."""
import abc
import collections
import math
import time
from typing import List
import numpy as np
import pyglove as pg
import tensorflow as tf

from es_optimization.policies import Policy


class TopologyPolicy(Policy):
  """Basic class for all policies using special topologies."""

  def __init__(self, state_dimensionality, action_dimensionality):
    self.state_dimensionality = state_dimensionality
    self.action_dimensionality = action_dimensionality

  @abc.abstractmethod
  def update_weights(self, vectorized_parameters):
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def update_topology(self, topology_str):
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def make_search_space(self):
    raise NotImplementedError("Abstract method")

  def update(self, vectorized_parameters):
    self.update_weights(vectorized_parameters)

  @property
  def dna_spec(self):
    """Contains the search space definition for the network architecture."""
    raise NotImplementedError("Abstract method")


class NumpyTopologyPolicy(TopologyPolicy):
  """Parent class for numpy-based policies."""

  def __init__(self, state_dimensionality, action_dimensionality,
               hidden_layers, **kwargs):
    super().__init__(
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality)
    self.hidden_layers = hidden_layers
    self.total_nb_nodes = sum(
        hidden_layers) + state_dimensionality + action_dimensionality
    self.all_layer_sizes = [self.state_dimensionality
                           ] + hidden_layers + [self.action_dimensionality]

    self.total_weight_parameters = self.total_nb_nodes**2
    self.total_bias_parameters = self.total_nb_nodes
    self.total_nb_parameters = self.total_weight_parameters + self.total_bias_parameters

    np.random.seed(0)
    self.weight_list = np.random.uniform(
        low=-1.0, high=1.0, size=(self.total_nb_nodes, self.total_nb_nodes))
    self.bias_list = np.random.uniform(
        low=-1.0, high=1.0, size=self.total_nb_nodes)

  def get_action(self, state):
    values = [0.0] * self.total_nb_nodes
    for i in range(self.state_dimensionality):
      values[i] = state[i]
    for i in range(self.total_nb_nodes):
      if ((i > self.state_dimensionality) and
          (i < self.total_nb_nodes - self.action_dimensionality)):
        values[i] = np.tanh(values[i] + self.bias_list[i])
      if i in self.edge_dict:
        j_list = self.edge_dict[i]
        for j in j_list:
          t = self.weight_list[i][j]
          values[j] += t * values[i]

    action = np.reshape(
        values[len(values) - self.action_dimensionality:len(values)],
        (self.action_dimensionality, 1))
    action = np.tanh(action)
    return action

  @property
  def dna_spec(self):
    return self.template.dna_spec()


class NumpyEdgeSparsityPolicy(NumpyTopologyPolicy):
  """This policy prunes edges in the neural network."""

  def __init__(self, state_dimensionality, action_dimensionality, hidden_layers,
               hidden_layer_edge_num, edge_policy_sample_mode, **kwargs):

    self.edge_policy_sample_mode = edge_policy_sample_mode
    self.hidden_layer_edge_num = hidden_layer_edge_num
    super().__init__(state_dimensionality, action_dimensionality, hidden_layers,
                     **kwargs)
    self.make_all_possible_edges()
    self.make_search_space()
    self.init_topology()

  def update_weights(self, vectorized_parameters):
    weight_variables = np.reshape(
        vectorized_parameters[:self.total_weight_parameters],
        (self.total_nb_nodes, self.total_nb_nodes))
    bias_variables = vectorized_parameters[self.total_weight_parameters:]
    self.weight_list = weight_variables
    self.bias_list = bias_variables

  def init_topology(self):
    """Sets the edge_dict (needed for parent get_action function) to be complete."""
    init_dna = self.template.encode(next(pg.random_sample(self.search_space)))
    init_topology_str = pg.to_json(init_dna)
    self.update_topology(init_topology_str)

  def update_topology(self, topology_str):
    dna = pg.from_json(topology_str)
    decoded = self.template.decode(dna)

    if self.edge_policy_sample_mode == "independent":
      list_of_edges = []
      for sector_list_of_edges in decoded:
        list_of_edges.extend(sector_list_of_edges)
    else:
      list_of_edges = decoded

    self.edge_dict = self.list_to_edge_dict(list_of_edges)

  def get_initial(self):
    initial_params = np.concatenate(
        (self.weight_list.flatten(), self.bias_list.flatten()))
    return initial_params

  def get_total_num_parameters(self):
    return self.total_nb_parameters

  def make_all_possible_edges(self):
    if self.edge_policy_sample_mode == "aggregate_edges":
      self.all_possible_edges = []
      chunk_index = 0
      for i in range(len(self.all_layer_sizes) - 1):
        sector_before = list(
            range(chunk_index, chunk_index + self.all_layer_sizes[i]))
        sector_after = list(
            range(
                chunk_index + self.all_layer_sizes[i], chunk_index +
                self.all_layer_sizes[i] + self.all_layer_sizes[i + 1]))
        for a in sector_before:
          for b in sector_after:
            self.all_possible_edges.append((a, b))
        chunk_index += self.all_layer_sizes[i]

    elif self.edge_policy_sample_mode == "independent_edges":  # check if right
      self.ssd_list = []
      chunk_index = 0

      self.all_possible_edges = []
      for i in range(len(self.all_layer_sizes) - 1):
        k = self.hidden_layer_edge_num[i]
        layer_possible_edges = []
        sector_before = list(
            range(chunk_index, chunk_index + self.all_layer_sizes[i]))
        sector_after = list(
            range(
                chunk_index + self.all_layer_sizes[i], chunk_index +
                self.all_layer_sizes[i] + self.all_layer_sizes[i + 1]))
        for a in sector_before:
          for b in sector_after:
            layer_possible_edges.append((a, b))
            self.all_possible_edges.append((a, b))
        chunk_index += self.all_layer_sizes[i]
        ssd_i = pg.sublist_of(
            k,
            candidates=layer_possible_edges,
            choices_sorted=False,
            choices_distinct=True)

        self.ssd_list.append(ssd_i)

    elif self.edge_policy_sample_mode == "residual_edges":
      self.all_possible_edges = []

      for i in range(len(self.all_layer_sizes) - 1):
        for j in range(i + 1, len(self.all_layer_sizes)):
          sector_before = list(
              range(
                  sum(self.all_layer_sizes[0:i]),
                  sum(self.all_layer_sizes[0:i + 1])))
          sector_after = list(
              range(
                  sum(self.all_layer_sizes[0:j]),
                  sum(self.all_layer_sizes[0:j + 1])))

          for a in sector_before:
            for b in sector_after:
              self.all_possible_edges.append((a, b))

  def make_search_space(self):
    if self.edge_policy_sample_mode == "aggregate_edges":
      total_number_k = sum(self.hidden_layer_edge_num)

      self.search_space = pg.sublist_of(
          total_number_k,
          candidates=self.all_possible_edges,
          choices_sorted=False,
          choices_distinct=True)

    elif self.edge_policy_sample_mode == "independent_edges":  # check if right
      self.search_space = pg.List(self.ssd_list)

    elif self.edge_policy_sample_mode == "residual_edges":
      total_number_k = sum(self.hidden_layer_edge_num)
      self.search_space = pg.sublist_of(
          total_number_k,
          candidates=self.all_possible_edges,
          choices_sorted=False,
          choices_distinct=True)

    self.template = pg.template(self.search_space)

  def list_to_edge_dict(self, list_of_edges):
    """Performs necessary conversion for RPC format."""
    temp_dict = collections.defaultdict(list)
    for edge_pair in list_of_edges:
      small_vertex = min(edge_pair[0], edge_pair[1])
      large_vertex = max(edge_pair[0], edge_pair[1])
      temp_dict[small_vertex].append(large_vertex)
    return temp_dict


SAFE_RECIPROCAL_MAX = 10.0


def safe_reciprocal(x):
  if math.isclose(x, 0.0):
    return SAFE_RECIPROCAL_MAX

  return tf.math.reciprocal(x)


def step_function(x):
  """Classic step function. Outputs 1 if x is nonnegative, 0 otherwise."""
  return 1.0 - max(0.0, tf.math.sign(-x))


class OpEdgePolicy(NumpyEdgeSparsityPolicy):
  """Varying nonlinearities allowed for each node, while using the residual sampling mode from NumpyEdgeSparsityPolicy.

  Very similar to WANNs (https://weightagnostic.github.io/).
  """

  def __init__(self, state_dimensionality, action_dimensionality, hidden_layers,
               num_edges, **kwargs):

    self.nonlinearity_ops = [
        tf.nn.tanh, tf.nn.relu, tf.math.exp, tf.identity, tf.math.sin,
        tf.nn.sigmoid, tf.math.abs, tf.math.cos, tf.math.square, step_function,
        safe_reciprocal, step_function
    ]
    self.num_edges = num_edges
    super().__init__(
        state_dimensionality,
        action_dimensionality,
        hidden_layers,
        hidden_layer_edge_num=None,
        edge_policy_sample_mode=None,
        **kwargs)
    self.make_search_space()
    self.init_topology()

  def update_topology(self, topology_str):
    dna = pg.from_json(topology_str)
    decoded_dna = self.template.decode(dna)
    self.edge_dict = self.list_to_edge_dict(decoded_dna["edge_search_space"])
    self.node_op_dict = decoded_dna["op_search_space"]

  def get_action(self, state):
    values = [0.0] * self.total_nb_nodes
    for i in range(self.state_dimensionality):
      values[i] = state[i]
    for i in range(self.total_nb_nodes):
      if ((i > self.state_dimensionality) and
          (i < self.total_nb_nodes - self.action_dimensionality)):
        nonlinearity_fn = self.node_op_dict[i]
        values[i] = nonlinearity_fn(values[i] + self.bias_list[i])
      if i in self.edge_dict:
        j_list = self.edge_dict[i]
        for j in j_list:
          t = self.weight_list[i][j]
          values[j] += t * values[i]

    action = np.reshape(
        values[len(values) - self.action_dimensionality:len(values)],
        (self.action_dimensionality, 1))
    action = np.tanh(action)

    return action

  def make_search_space(self):
    """Creates both the edge and operation search space definitions."""
    self.make_all_possible_edges()
    self.edge_search_space = pg.sublist_of(
        self.num_edges,
        candidates=self.all_possible_edges,
        choices_sorted=False,
        choices_distinct=True)
    # Every node is allowed an operation over it, after matrix multiplication.
    self.op_search_space = pg.List(
        [pg.one_of(self.nonlinearity_ops) for i in range(self.total_nb_nodes)])

    self.search_space = pg.Dict(
        edge_search_space=self.edge_search_space,
        op_search_space=self.op_search_space)

    self.template = pg.template(self.search_space)

  def make_all_possible_edges(self):
    """This is a copied variant of the "residual" sampling method from its parent class NumpyEdgeSparsityPolicy.

    However, edges are now allowed to connect directly from the state to the
    action tensors.
    """
    self.all_possible_edges = []
    # For loops between all pairs of "chunks".
    # Each chunk represents a tensor layer.
    for i in range(len(self.all_layer_sizes) - 1):
      for j in range(i + 1, len(self.all_layer_sizes)):
        sector_before = list(
            range(
                sum(self.all_layer_sizes[0:i]),
                sum(self.all_layer_sizes[0:i + 1])))
        sector_after = list(
            range(
                sum(self.all_layer_sizes[0:j]),
                sum(self.all_layer_sizes[0:j + 1])))
        # Enumerates all possible edges between the two "chunks".
        for a in sector_before:
          for b in sector_after:
            self.all_possible_edges.append((a, b))


class EfficientNetPolicy(NumpyEdgeSparsityPolicy):
  """Search space is boolean on an edge rather than choice over aggregate set of edges."""

  def update_topology(self, topology_str):
    dna = pg.from_json(topology_str)
    decoded_dna = self.template.decode(dna)
    list_of_edges = self.mask_to_edge_list(decoded_dna["boolean_mask"])
    self.edge_dict = self.list_to_edge_dict(list_of_edges)

  def make_search_space(self):
    """Creates both the edge and operation search space definitions."""
    self.make_all_possible_edges()
    self.boolean_mask = pg.List(
        [pg.oneof([0, 1]) for edge in self.all_possible_edges])
    self.search_space = pg.Dict(boolean_mask=self.boolean_mask)
    self.template = pg.template(self.search_space)

  def mask_to_edge_list(self, boolean_mask):
    edge_list = []
    for i, mask_val in enumerate(boolean_mask):
      if mask_val:
        edge_list.append(self.all_possible_edges[i])
    return edge_list

  def compute_flops_multiplication(self):
    multiplications = 0
    for v in self.edge_dict:
      multiplications += len(self.edge_dict[v])
    return multiplications

  def compute_inference_time(self, samples=30):
    state_samples = np.random.normal(size=(samples, self.state_dimensionality))
    inference_times = []
    for state_sample in state_samples:
      start_time = time.time()
      _ = self.get_action(state_sample)
      end_time = time.time()

      inference_times.append(end_time - start_time)

    return np.mean(inference_times)

  def compute_pareto_score_multiplications(self,
                                           reward,
                                           target_num=64,
                                           weight_factor_alpha=0.0,
                                           weight_factor_beta=-1.0):
    multiplications = self.compute_flops_multiplication()
    ratio = float(multiplications) / float(target_num)

    if ratio <= 1.0:
      return (reward + 1000.0) * (ratio**weight_factor_alpha)
    else:
      return (reward + 1000.0) * (ratio**weight_factor_beta)


def convert_to_color_to_edge(edge_to_color_dict):
  temp_dict = collections.defaultdict(list)
  for edge_key in edge_to_color_dict:
    partition_number = edge_to_color_dict[edge_key]
    temp_dict[partition_number].append(eval(edge_key))  # pylint: disable=eval-used
  return temp_dict


class NumpyWeightSharingPolicy(NumpyTopologyPolicy):
  """This policy takes in a partition to share weights across the NN."""

  def __init__(self, state_dimensionality, action_dimensionality, hidden_layers,
               num_partitions, **kwargs):

    self.num_partitions = num_partitions
    super().__init__(state_dimensionality, action_dimensionality, hidden_layers,
                     **kwargs)
    self.true_weights = np.random.normal(
        loc=0.0, scale=1.0, size=self.num_partitions)
    self.make_sectors()
    self.make_search_space()
    self.init_topology()
    self.partition_weights_to_network_weights(self.true_weights)
    self.total_nb_parameters = self.num_partitions + self.total_nb_nodes

  def init_topology(self):
    """Sets the edge_dict (needed for parent get_action function) to be complete."""
    self.edge_dict = collections.defaultdict(list)
    for sector_pair in self.sectors:
      sector_before, sector_after = sector_pair
      for a in sector_before:
        for b in sector_after:
          self.edge_dict[a].append(b)

    # Initializes random partition to propagate true weights.
    self.partition = convert_to_color_to_edge(
        next(pg.random_sample(self.search_space)))

  def partition_weights_to_network_weights(self, new_true_weights):
    self.true_weights = new_true_weights
    for partition_index in self.partition:
      for edge in self.partition[partition_index]:
        v = edge[0]
        w = edge[1]
        self.weight_list[v][w] = self.true_weights[partition_index]

  def update_weights(self, vectorized_parameters):
    self.partition_weights_to_network_weights(
        vectorized_parameters[:self.num_partitions])
    self.bias_list = vectorized_parameters[self.num_partitions:]

  def update_topology(self, topology_str):
    """Partition is dict: color_index -> List[edges]."""
    self.partition = convert_to_color_to_edge(
        self.template.decode(pg.from_json(topology_str)))
    self.partition_weights_to_network_weights(self.true_weights)

  def get_initial(self):
    initial_params = np.concatenate(
        (self.true_weights.flatten(), self.bias_list.flatten()))
    return initial_params

  def get_total_num_parameters(self):
    return self.total_nb_parameters

  def make_sectors(self):
    self.sectors = []
    chunk_index = 0
    # format: [[sector_before, sector_after], [sector_before, sector_after],...]
    for i in range(len(self.all_layer_sizes) - 1):
      sector_before = list(
          range(chunk_index, chunk_index + self.all_layer_sizes[i]))
      sector_after = list(
          range(
              chunk_index + self.all_layer_sizes[i], chunk_index +
              self.all_layer_sizes[i] + self.all_layer_sizes[i + 1]))

      self.sectors.append([sector_before, sector_after])
      chunk_index += self.all_layer_sizes[i]

  def make_search_space(self):
    self.search_space = pg.Dict()

    for sector_pair in self.sectors:
      sector_before, sector_after = sector_pair
      for a in sector_before:
        for b in sector_after:
          edge_key = str((a, b))
          self.search_space[edge_key] = pg.one_of(
              list(range(self.num_partitions)))

    self.template = pg.template(self.search_space)
