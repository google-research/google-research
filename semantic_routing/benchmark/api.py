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

"""Tensorflow wrappers for edgelist prediction datasets."""

import random

import tensorflow as tf
import tensorflow_datasets as tfds

from semantic_routing.benchmark import config
from semantic_routing.benchmark import utils
from semantic_routing.benchmark.datasets import routing_dataset
from semantic_routing.benchmark.datasets import touring_dataset
from semantic_routing.benchmark.graphs import city_graph
from semantic_routing.benchmark.graphs import grid_graph
from semantic_routing.benchmark.query_engines import basic_query_engines
from semantic_routing.benchmark.query_engines import labeled_query_engines
from semantic_routing.tokenization import tokenization


MAX_DATAPOINT_ATTEMPTS = 50
MAX_SEED = int(1e8)
QUERY_SEED = 0
QUERY_SPLITS = (0.95, 0, 0.05)


class EdgeListPredictionDatapointGenerator(object):
  """TF-compatible generator class for edgelist prediction datasets.

  This class provides a TensorFlow-compatible data generator for edgelist
  prediction tasks, yielding featurized datapoints based on configurable
  graph types, query engines, and tasks (touring or routing). It handles the
  generation of underlying road graphs, creation of datasets based on the
  specified task, and sampling of individual datapoints.

  The generator can be used directly in TensorFlow training pipelines via the
  `as_dataset` method, which returns a `tf.data.Dataset` yielding batches
  of encoded datapoints.
  """

  def __init__(
      self,
      seed,
      query_engine_type,
      graph_type,
      task,
      num_nodes,
      receptive_field_size,
      poi_receptive_field_size,
      poi_prob,
      use_fresh,
      test,
      prefix_len,
      poi_specs,
      **kargs,
  ):
    """Initializes the EdgeListPredictionDatapointGenerator.

    Args:
      seed: Random seed for data generation. Affects graph and datapoint
        generation within a generator instance.
      query_engine_type: Type of query engine to use. Options are: "basic": Uses
        a simple template-based query engine. "labeled": Uses a human-labeled
        query dataset.
      graph_type: Type of graph to create. Options are: "grid": Creates a
        `grid_graph.GridGraph`. "city": Creates a `city_graph.CityGraph`.
        "simplecity": Creates a `city_graph.CityGraph` with edge contractions.
      task: The prediction task. Options are: "touring": Generates datapoints
        for a touring task. "routing": Generates datapoints for a routing task.
      num_nodes: Number of nodes in the graph.
      receptive_field_size: Receptive field size for the model.
      poi_receptive_field_size: Receptive field size for POIs.
      poi_prob: Probability of a node being a POI.
      use_fresh: If True, a new datapoint is generated for each request, without
        caching.
      test: If True, uses test splits for data generation.
      prefix_len: Length of the input prefix during datapoint generation.
      poi_specs: Specifications for generating POIs.
      **kargs: Additional keyword arguments passed to the dataset constructor.
        These can include: `max_segments` (int): Maximum number of segments for
        'city' and 'simplecity' graphs (default: 600). `cand_len` (int):
        Candidate length for 'city' and 'simplecity' graphs (default: 20).
        `auto_simplify_datapoint` (bool): Automatically contract edges for
        'simplecity' graphs (default: True).
    """

    self.base_seed = seed
    self.use_fresh = use_fresh
    self.prefix_len = prefix_len
    self.test = test
    self.rng = random.Random(seed)
    self.poi_specs = poi_specs

    if query_engine_type == "basic":
      if task == "touring":
        self.engine = basic_query_engines.POIBasedTouringQueryEngine(
            poi_specs=self.poi_specs, splits=QUERY_SPLITS, seed=QUERY_SEED
        )
      elif task == "routing":
        self.engine = basic_query_engines.POIBasedRoutingQueryEngine(
            poi_specs=self.poi_specs, splits=QUERY_SPLITS, seed=QUERY_SEED
        )
      else:
        raise ValueError("Unsupported task: %s" % task)
    elif query_engine_type == "labeled":
      self.engine = labeled_query_engines.HumanLabeledQueryEngine(
          poi_specs=self.poi_specs, splits=(0.98, 0, 0.02), seed=QUERY_SEED
      )
    else:
      raise ValueError("Unsupported query engine type: %s" % query_engine_type)
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=config.DEFAULT_BERT_VOCAB
    )
    self.num_nodes = num_nodes
    self.receptive_field_size = receptive_field_size
    self.poi_receptive_field_size = poi_receptive_field_size
    self.poi_prob = poi_prob
    self.graph_type = graph_type
    self.task = task

    self.dataset_kargs = kargs

    if (
        self.graph_type in ("city", "simplecity")
        and "max_segments" not in self.dataset_kargs
    ):
      self.dataset_kargs["max_segments"] = 600
    if (
        self.graph_type in ("city", "simplecity")
        and "cand_len" not in self.dataset_kargs
    ):
      self.dataset_kargs["cand_len"] = 20

    if (
        self.graph_type == "simplecity"
        and "auto_simplify_datapoint" not in self.dataset_kargs
    ):
      self.dataset_kargs["auto_simplify_datapoint"] = True

    graph = self.create_graph(seed)
    self.aux_embedding_dim = (graph.embedding_dim,)

  def signature(self):
    """Signature of an encoded datapoint."""
    return {
        "token_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "aux_embeddings": tf.TensorSpec(
            shape=(None, *self.aux_embedding_dim), dtype=tf.float32
        ),
        "type_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "input_mask": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "candidate_token_ids": tf.TensorSpec(
            shape=None,
            dtype=tf.int32,
        ),
        "candidate_type_ids": tf.TensorSpec(
            shape=None,
            dtype=tf.int32,
        ),
        "candidates": tf.TensorSpec(
            shape=None,
            dtype=tf.int32,
        ),
        "candidate_aux_embeddings": tf.TensorSpec(
            shape=(None, *self.aux_embedding_dim),
            dtype=tf.float32,
        ),
        "num_candidates": tf.TensorSpec(shape=(), dtype=tf.int32),
        "position_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }

  def create_graph(self, seed):
    """Creates road graph."""
    if self.test:
      splits = (0, 0, 1)
    else:
      splits = (1, 0, 0)
    if self.graph_type == "grid":
      return grid_graph.GridGraph(
          poi_specs=self.poi_specs,
          num_nodes=self.num_nodes,
          poi_prob=self.poi_prob,
          seed=seed,
          splits=splits,
      )
    elif self.graph_type in ("city", "simplecity"):
      return city_graph.CityGraph(
          poi_specs=self.poi_specs,
          num_nodes=self.num_nodes,
          seed=seed,
          city_group_seed=self.base_seed,
          splits=splits,
          use_test_city=self.test,
      )
    else:
      raise ValueError("Unsupported graph type: %s" % self.graph_type)

  def create_dataset(self, seed, graph):
    """Creates dataset."""
    if self.task == "touring":
      return touring_dataset.TouringDataset(
          self.tokenizer,
          graph,
          self.engine,
          self.poi_specs,
          seed,
          self.receptive_field_size,
          self.poi_receptive_field_size,
          **self.dataset_kargs,
      )
    elif self.task == "routing":
      return routing_dataset.RoutingDataset(
          self.tokenizer,
          graph,
          self.engine,
          self.poi_specs,
          seed,
          self.receptive_field_size,
          self.poi_receptive_field_size,
          **self.dataset_kargs,
      )
    else:
      raise ValueError("Unsupported task: %s" % self.task)

  def __call__(self):
    """Generator that featurized datapoints."""
    datapoint_iter = iter(self.generate_datapoints())
    while True:
      datapoint = next(datapoint_iter)
      yield datapoint["parent"].featurize_datapoint(datapoint, pad=True)

  def generate_datapoints(self):
    """Generator that yields datapoints."""
    while True:
      datapoint = None
      for _ in range(MAX_DATAPOINT_ATTEMPTS):
        seed = self.rng.randint(0, MAX_SEED)
        graph = self.create_graph(seed)
        data = self.create_dataset(seed, graph)
        try:
          done = False
          while not done:
            datapoint = data.sample_datapoint(
                True,
                2 if self.test else 0,
                prefix_len=self.prefix_len,
                use_fresh=self.use_fresh,
            )
            yield datapoint
            done = not data.datapoint_cached
            assert not self.use_fresh or done
          break
        except TimeoutError:
          continue
      if datapoint is None:
        raise ValueError("Maximum attempts reached.")

  def as_dataset(self, batch_size):
    """Return a tf.data.Dataset that yields encoded datapoints.

    Args:
      batch_size: batch size for datapoints yielded by TF generator.

    Returns:
      Tensorflow dataset object.
    """
    ds = tf.data.Dataset.from_generator(self, output_signature=self.signature())
    return tfds.as_numpy(ds.batch(batch_size, drop_remainder=True))
