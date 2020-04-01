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

r"""Implementation of Splitter, a method for learning node representations that capture multiple contexts.

===============================

This is part of the implementation accompanying the WWW 2019 paper, [_Is a
Single Embedding Enough? Learning Node Representations that Capture Multiple
Social Contexts_](https://ai.google/research/pubs/pub46238).

Citing
------
If you find _Splitter_ or the associated resources useful in your research,
we ask that you cite the following paper:
> Epasto, A., Perozzi, B., (2019).
> Is a Single Embedding Enough? Learning Node Representations that Capture
Multiple Social Contexts.
> In _The Web Conference_.

Example execution
------
python3 -m graph_embedding.persona.splitter
  --input_graph=${graph} \
  --output_embedding=${embedding_output}

Where ${graph} is the path to a text file containing the graph and
${embedding_output} is the path to the output embedding.

The graph input format is a text file containing one edge per row represented
as its pair of node ids. The graph is supposed to be undirected.
For instance the file:
1 2
2 3
represents the triangle 1, 2, 3.

The output embedding format is a text file containing for each row one
(overlapping) cluster represented as the space-separted list of node ids in the
cluster.

For the persona decomposition, a number of different local clustering
algorithms can be used.  Supported out of the box are"

connected_components: the standard connected component algorithm.
label_prop: a label propagation based algorithm
            (nx.label_prop.label_propagation_communities).
modularity: an algorithm optimizing modularity
            (nx.modularity.greedy_modularity_communities).
"""
#pylint: skip-file
from __future__ import print_function

import os.path
import random

from . import persona
from absl import app
from absl import flags
from gensim.models import Word2Vec
import networkx as nx
import numpy
from six.moves import xrange
from .third_party import persona2vec

flags.DEFINE_string('output_persona_embedding', None, 'model.')

flags.DEFINE_string('output_embedding_prior', None, 'model.')

flags.DEFINE_integer('embedding_dim', 128, 'embedding_dim.')

flags.DEFINE_integer('walk_length', 10, 'walk_length.')

flags.DEFINE_integer('num_walks_node', 40, 'num_walks_node.')

flags.DEFINE_integer('iterations', 10, 'iterations.')

flags.DEFINE_float('constraint_learning_rate_scaling_factor', 0.1,
                   'learning rate constraint.')

flags.DEFINE_integer('seed', 1, 'seed.')

flags.DEFINE_integer('window_size', 5, 'window size over random walk.')

FLAGS = flags.FLAGS


def Splitter(graph,
             embedding_dim=128,
             walk_length=40,
             num_walks_node=10,
             constraint_learning_rate_scaling_factor=0.1,
             iterations=10,
             seed=None,
             window_size=5,
             local_clustering_fn=persona._CLUSTERING_FN['label_prop']):
  """This function runs the Splitter algorithm.

  Given a graph, it decomposes the nodes into personas.  It then embeds the
  original graph, and the persona graph to learn a representation that has
  multiple senses.

  Args:
    graph: Undirected graph represented as a dictionary of lists that maps each
      node id its list of neighbor ids;
    embedding_dim: The dimensionality of the embedding to use.
    walk_length: The length of the random walks to generate from each node.
    num_walks_node: The number of walks to start at each node.
    constraint_learning_rate_scaling_factor: Strength of the constraint that
      personas predict their original node.
    iterations: Number of iterations to run for.
    seed: Initial seed to use.
    window_size: Size of the window around the source node in the random walk.
    local_clustering_fn: A non-overlapping clustering algorithm function that
      takes in input a nx.Graph and outputs the a clustering. The output format
      is a list containing each partition as element. Each partition is in turn
      represented as a list of node ids. The default function is the networkx
      label_propagation_communities clustering algorithm.

  Returns:
    A pair of (graph, mapping) where "graph" is an nx.Graph instance of the
    persona graph (which contains different nodes from the original graph) and
    "mapping" is a dict of the new node ids to the node ids in the original
    graph.The persona graph as nx.Graph, and the mapping of persona nodes to
    original node ids.
  """
  to_return = {}

  print('Running persona decomposition...')
  # perform persona decomposition
  persona_graph, persona_id_mapping = persona.CreatePersonaGraph(
      graph, local_clustering_fn, persona_start_id=graph.number_of_nodes() + 1)

  # make sure ids don't collide between persona graph & input
  persona_id_set = set()
  graph_id_set = set()

  for x in graph:
    for y in graph[x]:
      graph_id_set.add(str(y))

  for x in persona_graph:
    for y in persona_graph[x]:
      persona_id_set.add(str(y))

  assert len(graph_id_set & persona_id_set
            ) == 0, 'intersection between graph ids and persona ids is non-zero'

  to_return['persona_graph'] = persona_graph
  to_return['persona_id_mapping'] = persona_id_mapping

  # generate random walks
  print('Generating persona random walks...')
  sentences_persona = list(
      GenerateRandomWalks(
          persona_graph, walks_per_node=num_walks_node,
          walk_length=walk_length))
  random.shuffle(sentences_persona)

  print('Generating regular random walks...')
  sentences_regular = list(
      GenerateRandomWalks(
          graph, walks_per_node=num_walks_node, walk_length=walk_length))
  random.shuffle(sentences_regular)

  # initial embedding for prior
  regular_model = RunDeepWalk(
      sentences_regular, embedding_dim, window_size, iterations, seed=seed)

  # persona embedding
  persona_model = RunPersona2Vec(
      persona_id_mapping,
      sentences_persona,
      embedding_dim,
      window_size,
      iterations,
      constraint_learning_rate_scaling_factor,
      prior_model=regular_model,
      seed=seed)

  to_return['regular_model'] = regular_model
  to_return['persona_model'] = persona_model

  return to_return


def SampleNextNode(graph, node):
  d = graph[node]
  v_list = list(d.keys())
  num = len(v_list)
  if num > 0:
    random_value = numpy.random.choice(num)
    return v_list[random_value]
  else:
    return node


def GenerateRandomWalks(graph, walks_per_node, walk_length):
  for node in graph:
    for _ in xrange(walks_per_node):
      walk = [node]
      for _ in xrange(walk_length):
        walk.append(SampleNextNode(graph, walk[-1]))
      yield walk


def RunPersona2Vec(persona_id_mapping,
                   sentences,
                   embedding_dim,
                   window_size,
                   iterations,
                   constraint_learning_rate_scaling_factor,
                   seed=0,
                   prior_model=None):
  """Runs Persona2Vec implementation."""
  persona_map = {}
  for p, node in persona_id_mapping.items():
    if node not in persona_map:
      persona_map[node] = []
    persona_map[node].append(p)

  node_init_cnt = 0
  persona_init_cnt = 0

  initialization_map = {}
  if prior_model:
    for node in persona_map:
      initialization_map[node] = prior_model[node]
      node_init_cnt += 1
      for p in persona_map[node]:
        initialization_map[p] = prior_model[node]
        persona_init_cnt += 1

  print('Initialized %d nodes' % node_init_cnt)
  print('Initialized %d personas' % persona_init_cnt)

  model = persona2vec.Persona2Vec(
      initial_weight_map=initialization_map,
      extra_constraint_map=persona_map,
      constraint_learning_rate_scaling_factor=constraint_learning_rate_scaling_factor,
      sentences=sentences,
      min_count=0,
      sg=1,
      hs=1,
      negative=0,
      size=embedding_dim,
      seed=seed,
      sample=0,
      workers=12,
      window=window_size,
      iter=iterations)

  return model


def RunDeepWalk(sentences, embedding_dim, window_size, iterations, seed=0):
  """Runs standard DeepWalk model."""
  model = Word2Vec(
      sentences=sentences,
      min_count=0,
      sg=1,
      hs=1,
      negative=0,
      size=embedding_dim,
      seed=seed,
      sample=0,
      workers=12,
      window=window_size,
      iter=iterations)

  return model


def main(argv=()):
  del argv  # Unused.

  # confirm output paths exist
  assert os.path.exists(os.path.dirname(FLAGS.output_persona_embedding))
  if FLAGS.output_embedding_prior:
    assert os.path.exists(os.path.dirname(FLAGS.output_embedding_prior))
  if FLAGS.output_persona_graph:
    assert os.path.exists(os.path.dirname(FLAGS.output_persona_graph))
  if FLAGS.output_persona_graph_mapping:
    assert os.path.exists(os.path.dirname(FLAGS.output_persona_graph_mapping))

  print('Loading graph...')
  graph = nx.read_edgelist(FLAGS.input_graph, create_using=nx.Graph)

  # read persona args
  local_clustering_fn = persona._CLUSTERING_FN[
      FLAGS.local_clustering_method]

  print('Running splitter...')
  splitter = Splitter(
      graph,
      embedding_dim=FLAGS.embedding_dim,
      walk_length=FLAGS.walk_length,
      num_walks_node=FLAGS.num_walks_node,
      constraint_learning_rate_scaling_factor=FLAGS
      .constraint_learning_rate_scaling_factor,
      iterations=FLAGS.iterations,
      seed=FLAGS.seed,
      local_clustering_fn=local_clustering_fn)

  # output embeddings
  splitter['persona_model'].save_word2vec_format(
      open(FLAGS.output_persona_embedding, 'wb'))

  # optional output
  if FLAGS.output_embedding_prior is not None:
    splitter['regular_model'].save_word2vec_format(
        open(FLAGS.output_embedding_prior, 'wb'))

  if FLAGS.output_persona_graph is not None:
    nx.write_edgelist(splitter['persona_graph'], FLAGS.output_persona_graph)

  if FLAGS.output_persona_graph_mapping is not None:
    with open(FLAGS.output_persona_graph_mapping, 'w') as outfile:
      for persona_node, original_node in splitter['persona_id_mapping'].items():
        outfile.write('{} {}\n'.format(persona_node, original_node))

  return 0


if __name__ == '__main__':
  flags.mark_flag_as_required('input_graph')
  flags.mark_flag_as_required('output_persona_embedding')
  flags.mark_flag_as_required('output_persona_graph_mapping')
  app.run(main)
