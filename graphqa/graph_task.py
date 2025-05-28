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

"""The graph tasks to be tried with LLMs."""

import random

import networkx as nx
import numpy as np

from graphqa import graph_text_encoder


class GraphTask:
  """The parent class for all the graph tasks."""

  def __init__(self):
    self.name = 'default'
    self.maximum_nnodes_cot_graph = 10

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    raise NotImplementedError()

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    raise NotImplementedError()


class CycleCheck(GraphTask):
  """The graph task to check if there is at least one cycle or not."""

  def __init__(self):
    super().__init__()
    self.name = 'cycle_check'
    self._task_description = 'Q: Is there a cycle in this graph?\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = (
          graph_text_encoder.encode_graph(graph, encoding_method)
          + self._task_description
      )
      try:
        nx.find_cycle(graph)
        answer = 'Yes, there is a cycle.'
      except nx.NetworkXNoCycle:
        answer = 'No, there is no cycle.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': self._task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    """Create a few shot example w or w/o cot for the graph graph."""
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = (
        graph_text_encoder.encode_graph(graph, encoding_method)
        + self._task_description
    )
    try:
      cycle = nx.find_cycle(graph)
      cycle_text = ''
      answer = 'Yes, there is a cycle. '
      if cot:
        for pair in cycle:
          cycle_text += (
              name_dict[pair[0]]
              + ' is connected to '
              + name_dict[pair[1]]
              + ', '
          )
        cycle_cot = 'The cycle is: %s.' % cycle_text[:-2]
        answer += cycle_cot
    except nx.NetworkXNoCycle:
      answer = 'No, there is no cycle.'
    return question + answer

  def choose_few_shot_examples(
      self,
      few_shots_dict,
      encoding_method,
      k = 2,
  ):
    """Choose few shot examples for each algorithm."""
    pos_cycle_algorithms = ['er', 'ba', 'sbm', 'sfn', 'complete']
    neg_cycle_algorithms = ['star', 'path']
    few_shots_str = ''
    # choose k-1 shots for pos algorithms and one negative.
    positive_algorithms = random.choices(pos_cycle_algorithms, k=k - 1)
    for positive_algorithm in positive_algorithms:
      example_list = few_shots_dict[(positive_algorithm, encoding_method)]
      few_shots_str += 'Example: ' + random.choice(example_list) + '\n'
    negative_algorithm = random.choice(neg_cycle_algorithms)
    example_list = few_shots_dict[(negative_algorithm, encoding_method)]
    few_shots_str += 'Example: ' + random.choice(example_list) + '\n'
    return few_shots_str


class EdgeExistence(GraphTask):
  """The graph task to check if an edge exist in a graph or not."""

  def __init__(self):
    super().__init__()
    self.name = 'edge_existence'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = 'Q: Is node %s connected to node %s?\nA: ' % (
          name_dict[source],
          name_dict[target],
      )
      question += task_description
      if ((source, target) in graph.edges()) or (
          (target, source) in graph.edges()
      ):
        answer = 'Yes.'
      else:
        answer = 'No.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    source, target = random.sample(list(graph.nodes()), k=2)
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    question += 'Q: Is node %s connected to node %s?\nA: ' % (
        name_dict[source],
        name_dict[target],
    )
    if ((source, target) in graph.edges()) or (
        (target, source) in graph.edges()
    ):
      answer = 'Yes.'
      if cot:
        answer += (
            ' Because, there is an edge from %s to %s in the graph description.'
            % (name_dict[source], name_dict[target])
        )
    else:
      answer = 'No.'
      if cot:
        answer += (
            ' Because, there is no edge from %s to %s in the graph description.'
            % (name_dict[source], name_dict[target])
        )
    return question + answer


class NodeCount(GraphTask):
  """The graph task for finding number of nodes in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'node_count'
    self._task_description = 'Q: How many nodes are in this graph?\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      question += self._task_description
      answer = ' %d.' % len(graph.nodes())
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': self._task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict

  def get_nodes_string(self, name_dict, nnodes):
    node_string = ''
    for i in range(nnodes - 1):
      node_string += name_dict[i] + ', '
    node_string += 'and ' + name_dict[nnodes - 1]
    return node_string

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    question += self._task_description
    answer = '%d.' % len(graph.nodes())
    if cot:
      answer += ' The nodes are %s.' % self.get_nodes_string(
          name_dict, len(graph.nodes())
      )

    return question + answer


class NodeDegree(GraphTask):
  """The graph task for finding degree of a node in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'node_degree'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      source_node = random.sample(list(graph.nodes()), k=1)[0]
      task_description = (
          'Q: What is the degree of node %s?\nA: ' % name_dict[source_node]
      )
      question += task_description
      answer = '%d.' % graph.degree[source_node]
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source_node],
      }
    return examples_dict

  def get_edge_string(
      self, name_dict, graph, source_node
  ):
    """Gets a string identifying the edges a given node is connected to."""
    edge_string = ''
    target_edges = graph.edges(source_node)
    target_nodes = []
    for edge in target_edges:
      target_nodes.append(edge[1])
    if target_nodes:
      for i in range(len(target_nodes) - 1):
        edge_string += name_dict[target_nodes[i]] + ', '
      edge_string += 'and ' + name_dict[target_nodes[-1]]
    else:
      edge_string = 'no nodes'
    return edge_string

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    source_node = random.sample(list(graph.nodes()), k=1)[0]
    question += (
        'Q: What is the degree of node %s?\nA: ' % name_dict[source_node]
    )
    answer = '%d.' % graph.degree[source_node]
    if cot:
      answer += ' This is because %s is connected to %s.' % (
          name_dict[source_node],
          self.get_edge_string(name_dict, graph, source_node),
      )
    return question + answer


class EdgeCount(GraphTask):
  """The graph task for finding number of edges in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'edge_count'
    self._task_description = 'Q: How many edges are in this graph?\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      question += self._task_description
      answer = ' %d.' % len(graph.edges())
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': self._task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict

  def get_edges_string(
      self, name_dict, edges
  ):
    edges_string = ''
    for edge in edges:
      edges_string += (
          '(' + name_dict[edge[0]] + ', ' + name_dict[edge[1]] + '), '
      )
    return edges_string.strip()[:-1]

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    question += self._task_description
    answer = '%d.' % len(graph.edges())
    if cot:
      answer += ' The edges are %s.' % self.get_edges_string(
          name_dict, list(graph.edges())
      )
    return question + answer


class ConnectedNodes(GraphTask):
  """The graph task for finding connected nodes to a given node in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'connected_nodes'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      source_node = random.sample(list(graph.nodes()), k=1)[0]
      task_description = (
          'Q: List all the nodes connected to %s in alphabetical order.\nA: '
          % name_dict[source_node]
      )
      question += task_description
      outgoing_edges = list(graph.edges(source_node))
      if outgoing_edges:
        answer = self.get_connected_nodes(outgoing_edges, name_dict) + '.'
      else:
        answer = ' No nodes.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source_node],
      }
    return examples_dict

  def get_connected_nodes(
      self, edges, name_dict
  ):
    """Gets a string including all the nodes that are connected to source."""
    connected_nodes = []
    for edge in edges:
      connected_nodes.append(name_dict[edge[1]])
    connected_nodes_string = ''
    if connected_nodes:
      try:
        int(connected_nodes[0])
        connected_nodes_string = ', '.join(map(str, connected_nodes))
      except ValueError:
        # Check if these are not integers, sort
        connected_nodes_string = ', '.join(map(str, sorted(connected_nodes)))
    return connected_nodes_string

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    source_node = random.sample(list(graph.nodes()), k=1)[0]
    question += (
        'Q: List all the nodes connected to %s in alphabetical order.\nA: '
        % name_dict[source_node]
    )
    outgoing_edges = list(graph.edges(source_node))
    answer = ''
    if outgoing_edges:
      answer = self.get_connected_nodes(outgoing_edges, name_dict) + '.'
      if cot:
        answer += ' This is because there is an edge from %s to %s.' % (
            name_dict[source_node],
            answer,
        )
      else:
        answer = 'No nodes.'
        if cot:
          answer += (
              ' This is because %s is not connected to any node.'
              % name_dict[source_node]
          )
    return question + answer


class DisconnectedNodes(GraphTask):
  """The task for finding disconnected nodes for a given node in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'disconnected_nodes'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      source_node = random.sample(list(graph.nodes()), k=1)[0]
      task_description = (
          'Q: List all the nodes that are not connected to %s in alphabetical'
          ' order.\nA: '
          % name_dict[source_node]
      )
      question += task_description
      outgoing_edges = list(graph.edges(source_node))
      answer = self.get_disconnected_nodes(
          source_node, outgoing_edges, name_dict, list(graph.nodes())
      )
      if not answer:
        answer = 'No nodes'

      answer += '.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source_node],
      }
    return examples_dict

  def get_disconnected_nodes(
      self,
      source,
      edges,
      name_dict,
      all_nodes,
  ):
    """Gets a string with all the nodes that are not connected to source."""
    for edge in edges:
      if edge[1] in all_nodes:
        all_nodes.remove(edge[1])
    if source in all_nodes:
      all_nodes.remove(source)
    all_nodes_names = []
    for node in all_nodes:
      all_nodes_names.append(name_dict[node])
    # sorted operation should be different for integers vs strings.
    if all_nodes_names:
      try:
        int(all_nodes_names[0])
        for ind, value in enumerate(all_nodes_names):
          all_nodes_names[ind] = int(value)
        all_nodes_names = sorted(all_nodes_names)
        for ind, value in enumerate(all_nodes_names):
          all_nodes_names[ind] = str(value)
      except ValueError:
        pass
    return ', '.join(map(str, sorted(all_nodes_names)))

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    source_node = random.sample(list(graph.nodes()), k=1)[0]
    question += (
        'Q: List all the nodes that are not connected to %s in alphabetical'
        ' order.\nA: '
        % name_dict[source_node]
    )
    outgoing_edges = list(graph.edges(source_node))
    answer = ''
    disconnected_nodes_string = self.get_disconnected_nodes(
        source_node, outgoing_edges, name_dict, list(graph.nodes())
    )
    if outgoing_edges:
      if not disconnected_nodes_string:
        disconnected_nodes_string = 'No nodes'
      answer = disconnected_nodes_string + '.'
      if cot:
        answer += ' This is because there is not an edge from %s to %s.' % (
            name_dict[source_node],
            answer,
        )
      else:
        answer = ' No nodes.'
        if cot:
          answer += (
              ' This is because %s is connected to all nodes.'
              % name_dict[source_node]
          )
    return question + answer


class Reachability(GraphTask):
  """The graph task to check if there is a path from a source to target."""

  def __init__(self):
    super().__init__()
    self.name = 'reachability'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = 'Q: Is there a path from node %s to node %s?\nA: ' % (
          name_dict[source],
          name_dict[target],
      )
      question += task_description
      if nx.has_path(graph, source, target):
        answer = 'Yes.'
      else:
        answer = 'No.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    source, target = random.sample(list(graph.nodes()), k=2)
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    question += 'Q: Is there a path from node %s to node %s?\nA: ' % (
        name_dict[source],
        name_dict[target],
    )
    if nx.has_path(graph, source, target):
      answer = 'Yes.'
      if cot:
        path = nx.shortest_path(graph, source, target)
        explanation = ' Because'
        for i in range(len(path) - 1):
          # The only edge or the non-last edges in the path.
          if len(path) == 2 or i < len(path) - 2:
            sep = ','
          # The last edge in a path with more than one edge.
          else:
            sep = ', and'
          explanation += '%s there is an edge from node %d to node %d' % (
              sep,
              path[i],
              path[i + 1],
          )
        explanation += ' .'
        answer += explanation
    else:
      answer = 'No.'
      if cot:
        answer += (
            ' Because, there is no path connecting node %s to node %s based on'
            ' the graph description.' % (name_dict[source], name_dict[target])
        )
    return question + answer


class ShortestPath(GraphTask):
  """The graph task to check if there is a path from a source to target."""

  def __init__(self):
    super().__init__()
    self.name = 'shortest_path'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = (
          'Q: What is the length of the shortest path from node %s to node'
          ' %s?\nA: '
          % (
              name_dict[source],
              name_dict[target],
          )
      )
      question += task_description
      try:
        path = nx.shortest_path(graph, source, target)
        answer = str(len(path) - 1) + '.'
      except nx.NetworkXNoPath:
        answer = 'There is no path from node %s to node %s.' % (
            name_dict[source],
            name_dict[target],
        )
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    source, target = random.sample(list(graph.nodes()), k=2)
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    question += (
        'Q: What is the length of the shortest path from node %s to node'
        ' %s?\nA: '
        % (
            name_dict[source],
            name_dict[target],
        )
    )
    if nx.has_path(graph, source, target):
      path = nx.shortest_path(graph, source, target)
      answer = str(len(path) - 1) + '.'
      if cot:
        explanation = ' Because'
        for i in range(len(path) - 1):
          # The only edge or the non-last edges in the path.
          if len(path) == 2 or i < len(path) - 2:
            sep = ','
          # The last edge in a path with more than one edge.
          else:
            sep = ', and'
          explanation += '%s there is an edge from node %d to node %d' % (
              sep,
              path[i],
              path[i + 1],
          )
        explanation += ' .'
        answer += explanation
    else:
      answer = 'There is no path from node %s to node %s.' % (
          name_dict[source],
          name_dict[target],
      )
      if cot:
        answer += (
            ' Because, there is no path connecting node %s to node %s based on'
            ' the graph description.' % (name_dict[source], name_dict[target])
        )
    return question + answer


class TriangleCounting(GraphTask):
  """The graph task to count the number of triangles in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'triangle_counting'
    self._task_description = 'Q: How many triangles are in this graph?\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = (
          graph_text_encoder.encode_graph(graph, encoding_method)
          + self._task_description
      )
      ntriangles = int(np.sum(list(nx.triangles(graph).values())) / 3)

      answer = '%i.' % ntriangles
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': self._task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    """Create a few shot example w or w/o cot for the graph graph."""
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = (
        graph_text_encoder.encode_graph(graph, encoding_method)
        + self._task_description
    )
    triangles_dict = nx.triangles(graph)
    ntriangles = int(np.sum(list(triangles_dict.values())) / 3)

    if ntriangles > 0:
      answer = '%i.' % ntriangles
      if cot:
        ntriangles_cot = ''
        for key, value in triangles_dict.items():
          if value > 0:
            if value == 1:
              ntriangles_cot += (
                  'There is %i triangle including node %s as a vertex.\n'
                  % (value, name_dict[key])
              )
            else:
              ntriangles_cot += (
                  'There are %i triangles including node %s as a vertex.\n'
                  % (value, name_dict[key])
              )
        ntriangles_cot += (
            'Summing the number of triangles for all nodes and dividing them by'
            ' three gives us %i triangles in total.' % ntriangles
        )
        answer += ntriangles_cot
    else:
      answer = '0.'
      if cot:
        ntriangles_cot = 'No three nodes form a triangle of edges.'
        answer += ntriangles_cot
    return question + answer


class MaximumFlow(GraphTask):
  """The graph task to compute the maximum flow from a source to a target."""

  def __init__(self):
    super().__init__()
    self.name = 'maximum_flow'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      graph = add_edge_weight(graph)
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = (
          'Q: What is the maximum capacity of the flow from node %s to node'
          ' %s?\nA: ' % (name_dict[source], name_dict[target])
      )
      question += task_description
      maximum_flow_value = nx.maximum_flow(
          graph, source, target, capacity='weight'
      )[0]
      answer = str(maximum_flow_value) + '.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(len(graph.nodes())),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    graph = add_edge_weight(graph)
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    source, target = random.sample(list(graph.nodes()), k=2)
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    question += (
        'Q: What is the maximum capacity of the flow from node %s to'
        ' node %s?\nA: ' % (name_dict[source], name_dict[target])
    )
    flow_value, flow_dict = nx.maximum_flow(
        graph, source, target, capacity='weight'
    )
    answer = str(flow_value) + '.'
    if flow_value > 0:
      if cot:
        explanation = ' This is because of the following edges: '
        for edge, capacity in flow_dict.items():
          for key, value in capacity.items():
            if value > 0:
              explanation += (
                  'the edge from node %i to node %i with capacity %i, '
                  % (
                      edge,
                      key,
                      value,
                  )
              )
        explanation = explanation.strip()[:-1] + '.'
        answer += explanation
    else:
      if cot:
        answer += (
            ' Because, there is no path connecting node %s to node %s based on'
            ' the graph description.' % (name_dict[source], name_dict[target])
        )
    return question + answer


def has_edge_weights(graph):
  for _, _, data in graph.edges(data=True):
    if 'weight' not in data:
      return False
  return True


def add_edge_weight(graph):
  if has_edge_weights(graph):
    return graph
  else:
    for edge in graph.edges():
      graph[edge[0]][edge[1]]['weight'] = random.randint(1, 10)
    return graph


class NodeClassification(GraphTask):
  """The graph task to classify a given node in the graph."""

  def __init__(self):
    super().__init__()
    self.name = 'node_classification'
    self.classes = [
        'soccer',
        'baseball',
        'tennis',
        'golf',
        'football',
        'surfing',
    ]

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    classes = random.sample(list(self.classes), k=2)
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      nnodes = len(graph.nodes())
      # Sampling nnodes // 2 + 1 nodes.
      sampled_nodes = random.sample(
          list(graph.nodes(data=True)), k=nnodes // 2 + 1
      )
      # Adding the class of half of the nodes.
      for node_data in sampled_nodes[:-1]:
        node_class = classes[node_data[1]['block']]
        question += (
            'Node ' + name_dict[node_data[0]] + ' likes ' + node_class + '.\n'
        )
      # Reserving the last sampled node for the question.
      task_description = 'Q: Does node %s like %s or %s?\nA: ' % (
          name_dict[sampled_nodes[-1][0]],
          classes[0],
          classes[1],
      )
      question += task_description
      answer = classes[sampled_nodes[-1][1]['block']]

      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(nnodes),
          'nedges': str(len(graph.edges())),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          # id of the last samples node
          'node_ids': [sampled_nodes[-1][0]],
      }

    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    classes = random.sample(list(self.classes), k=2)
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    question = graph_text_encoder.encode_graph(graph, encoding_method)
    nnodes = len(graph.nodes())
    sampled_nodes = random.sample(
        list(graph.nodes(data=True)), k=nnodes // 2 + 1
    )
    for node_data in sampled_nodes[:-1]:
      node_class = classes[node_data[1]['block']]
      question += (
          'Node ' + name_dict[node_data[0]] + ' likes ' + node_class + '.\n'
      )
    task_description = 'Q: Does node %s like %s or %s?\nA: ' % (
        name_dict[sampled_nodes[-1][0]],
        classes[0],
        classes[1],
    )
    question += task_description
    answer = classes[sampled_nodes[-1][1]['block']]

    if cot:
      explanation = (
          ' This is because most of the nodes that are connected to node %s'
          ' likes %s.'
          % (sampled_nodes[-1][0], classes[sampled_nodes[-1][1]['block']])
      )
      answer += explanation
    return question + answer
