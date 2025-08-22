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

"""I/O routines for HUGE-TPU OSS project."""

from typing import Callable, Iterator, List, Optional, Tuple

from absl import logging
import tensorflow as tf


class PositiveExampleParser:
  """Callable for parsing positive graph sample examples.

  Clients will probably not use this directly and instead should prefer the free
  function LoadPositiveSamplesDataset.
  """

  _SOURCE_ID = "S"
  _DESTINATION_ID = "D"
  _FEATURES = "W"

  def __init__(self, walk_length):
    """Initialize the positive samples parser.

    Args:
      walk_length: The length of each random walk or equivalently the size of
        the co-occurence feature count.
    """
    self.feature_spec = {
        self._SOURCE_ID: tf.io.FixedLenFeature([], dtype=tf.int64),
        self._DESTINATION_ID: tf.io.FixedLenFeature([], dtype=tf.int64),
        self._FEATURES: tf.io.FixedLenFeature([walk_length], dtype=tf.float32),
    }

  def __call__(self, serialized):
    td = tf.io.parse_example(serialized, self.feature_spec)
    return (td[self._SOURCE_ID], td[self._DESTINATION_ID], td[self._FEATURES])


def load_positive_dataset(
    filenames, positive_batch_size, walk_length
):
  """Loads a positive samples dataset.

  Args:
    filenames: A list of tfrecord formatted file shards containing positive
      co-occurence counts. If dataset is only a single file pass in [filename],
      else use os.glob or similar to build the file list.
    positive_batch_size: The desired positive example batch size.
    walk_length: The number of steps per random-walk simulation.

  Returns:
    An iterable tf.data.Dataset instance.
  """
  positive_example_parser = PositiveExampleParser(walk_length)

  ds = tf.data.Dataset.from_tensor_slices(filenames)
  ds = ds.shuffle(len(filenames))
  ds = ds.interleave(
      tf.data.TFRecordDataset,
      cycle_length=tf.data.AUTOTUNE,
      deterministic=False,
      num_parallel_calls=tf.data.AUTOTUNE,
  )

  # shuffle buffer size (4*batch_size) is small and picked at random. However,
  # this may be good enough or not needed if the data is already shuffled on the
  # sampler side prior to saving to disk on top of the global shuffle on input
  # shard file names.
  ds = (
      ds.shuffle(4 * positive_batch_size)
      .repeat()
      .batch(positive_batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
  )

  ds = ds.map(positive_example_parser, num_parallel_calls=tf.data.AUTOTUNE)

  return ds


class RandomUniformNegativeSampler:
  """Callable for augmenting a positive sampled tf.data.Dataset.

  For use with `AddRandomNegatives` to inject random negative sampling into the
  tf.data.Dataset pipeline.
  """

  def __init__(self, num_nodes, num_neg_per_pos):
    self.num_nodes = num_nodes
    self.num_neg_per_pos = num_neg_per_pos

  def __call__(
      self, src, dst, feat, *unused_tensors
  ):
    """Return tensors positive and negative samples.

    Args:
      src: tf.Tensor with shape [batch_size] of positive source node IDs.
      dst: tf.Tensor with shape [batch_size] of positive destination node IDs.
      feat: tf.Tensor with shape [batch_size, walk_len] of co-occurence counts.
      *unused_tensors: Any extra tensors not used by this function.

    Returns:
      A tuple of tf.Tensors.
    """
    neg_src = tf.repeat(src, self.num_neg_per_pos)
    random_offset = tf.random.uniform(
        minval=1,
        maxval=self.num_nodes,
        shape=tf.shape(neg_src),
        dtype=neg_src.dtype,
    )

    neg_dst = tf.math.floormod(
        tf.math.add(neg_src, random_offset), self.num_nodes
    )

    src = tf.concat([src, neg_src], axis=0, name="combine_src")
    dst = tf.concat([dst, neg_dst], axis=0, name="combine_dst")

    return (src, dst, feat, *unused_tensors)


def add_uniform_random_negatives(
    ds,
    num_nodes,
    num_negs_per_pos,
):
  """Inject random uniform negative sampling into the tf.data.Data pipeline.

  This function assumes the input node id space has been compressed on
  [0, num_nodes-1].

  Args:
    ds: A tf.data.Dataset instance that returns a tuple of positive examples.
    num_nodes: The total number of unique node ids in the dataset.
    num_negs_per_pos: The desired number of random negative examples for each
      positive example.

  Returns:
    A tf.data.Dataset that augments the dataset with uniform random negative
    sampling.
  """
  negative_sampler = RandomUniformNegativeSampler(num_nodes, num_negs_per_pos)
  return ds.map(
      negative_sampler, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE
  )


class ComputeExpectedEdgeScore:
  """Compute the expected edge score (denoted "D" in the model/literature).

  See "Watch Your Step: Learning Node Embeddings via Graph Attention",
    https://arxiv.org/pdf/1710.09599.pdf, Section 3 and 3.1:
    Expectation on the co-occurrence matrix: E[D].

  D is defined as a |V|x|V| matrix containing the co-occurrence observed
  within context window C. That is, D_{v,u} is the number of times nodes v
  and u are co-visited within the context distance c ~ U[1, C] in all
  simulated random walks.
  """

  def __init__(
      self, weights, edge_score_norm
  ):
    """Initialize expected edge score callable.

    Args:
      weights: Optional list of floats to weight each feature by. In practice,
        this is typically a monotonically decreasing sequence that weights the
        importance assigned to the co-occurance value as a function of the walk
        step. The length of this list must be equal to `num_features`.
      edge_score_norm: If this optional float is provided, it will be used to
        scale the positive term of the loss (of the edges) to the magnitude
        specified here. Suggested use: If a graph has many low-weight edges
        (e.g., 0.0001), which does not seem to converge, set this term equal to
        the batch size. This factor has only been shown to help emperically and
        is not theoretically motivated.

    Returns:
      Callable to compute expected edge score.
    """
    self.weights = weights
    self.edge_score_norm = edge_score_norm

  def __call__(
      self, src, dst, feat
  ):
    """Compute the expected edge score (denoted "D" in the model/literature).

    Args:
      src: tf.Tensor will be passed through tf.data.Dataset pipeline.
      dst: tf.Tensor will be passed through the tf.data.Dataset pipeline.
      feat: tf.Tensor of co-occurence features with shape [batch_size,
        num_features]. Used to compute expected edge score.

    Returns:
      A tensor of shape [batch_size]
    """
    if self.weights is not None:
      # Assumes that the feat.dtype is one of the floating point types.
      weights_t = tf.constant(
          self.weights, shape=(len(self.weights), 1), dtype=feat.dtype
      )
      d = tf.matmul(feat, weights_t)
    else:
      d = tf.reduce_sum(feat, axis=1)

    d = tf.squeeze(d)

    if self.edge_score_norm:
      edge_score_norm_t = tf.constant(self.edge_score_norm, dtype=d.dtype)
      d = tf.multiply(edge_score_norm_t, tf.nn.l2_normalize(d, axis=0))

    return src, dst, d


def add_expected_edge_score(
    ds,
    weights = None,
    edge_score_norm = None,
):
  """Compute the expected edge score in a tf.data.Dataset pipeline.

  Args:
    ds: A tf.data.Dataset instance that returns a tuple of positive examples.
    weights: Optional list of floats to weight each feature by.
    edge_score_norm: Optionally scale the norm of the per-batch expected edge
      score by this value. Emperically, setting this value to the batch size
      seems to help Deepwalk convergence if the graph has many low-weight edges
      (e.g., 0.0001).

  Returns:
    A tf.data.Dataset that augments the dataset with expected edge score.
  """
  expected_edge_score_fn = ComputeExpectedEdgeScore(
      weights=weights, edge_score_norm=edge_score_norm
  )
  return ds.map(
      expected_edge_score_fn,
      deterministic=False,
      num_parallel_calls=tf.data.AUTOTUNE,
  )


def load_deepwalk_dataset(
    filenames,
    num_nodes,
    positive_batch_size,
    walk_length,
    num_negs_per_pos = None,
    feature_weights = None,
    edge_score_norm = None,
):
  """Load a tf.data.Dataset that will apply HUGE-TPU transformations.

  Args:
    filenames: List of filenames to load.
    num_nodes: Total number of nodes in the graph.
    positive_batch_size: Desired number of positive examples per batch.
    walk_length: The number of steps taken during each random walk simulation.
    num_negs_per_pos: Desired number of negative examples per positive example.
      The total batch size will be `positive_batch_size * (1 +
      num_negs_per_pos)`.
    feature_weights: Optional list of floats to weights to apply to each feature
      when computing the expected edge score.
    edge_score_norm: Optional per-batch normalization constant. See
      `ComputeExpectedEdgeScore` for more details.

  Returns:
    A tf.data.Dataset object.
  """

  ds = load_positive_dataset(filenames, positive_batch_size, walk_length)
  if num_negs_per_pos is not None:
    ds = add_uniform_random_negatives(ds, num_nodes, num_negs_per_pos)
  return add_expected_edge_score(
      ds, weights=feature_weights, edge_score_norm=edge_score_norm
  )


def deepwalk_input_fn(
    ctx,
    filenames,
    num_nodes,
    positive_batch_size,
    walk_length,
    num_negs_per_pos = None,
    feature_weights = None,
    edge_score_norm = None,
    tf_data_service_address = None,
    tf_data_service_sharding_policy = tf.data.experimental.service.ShardingPolicy.OFF,
):
  """Helper function to satisfy tf.data.experimental.service API."""

  if feature_weights:
    assert len(feature_weights) == walk_length

  if ctx:
    positive_batch_size = ctx.get_per_replica_batch_size(positive_batch_size)

    # Positive weight adjustment is commonly set to the positive batch size,
    # scale by the number of replicas to keep a consistent positive weight
    # adjustment definition.
    if edge_score_norm:
      edge_score_norm /= ctx.num_replicas_in_sync

    logging.info(
        "Context aware num_replicas_in_sync: %d", ctx.num_replicas_in_sync
    )
    logging.info("Context aware positive_batch_size: %d", positive_batch_size)
    if edge_score_norm:
      logging.info("Context aware edge_score_norm: %f", edge_score_norm)
  else:
    logging.info("No input context normalization.")

  ds = load_deepwalk_dataset(
      filenames,
      num_nodes,
      positive_batch_size,
      walk_length,
      num_negs_per_pos,
      feature_weights,
      edge_score_norm,
  ).prefetch(tf.data.AUTOTUNE)

  if tf_data_service_address is not None:
    logging.info(
        "Using tf data service with address: %s", tf_data_service_address
    )
    ds = ds.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf_data_service_sharding_policy,
            service=tf_data_service_address,
        )
    )

  return ds


def create_distributed_dataset_iterator(
    strategy,
    input_fn,
):
  logging.info("Creating distributed dataset")
  ds = strategy.distribute_datasets_from_function(
      input_fn,
      options=tf.distribute.InputOptions(experimental_fetch_to_device=False),
  )
  logging.info("Finished creating distributed dataset.")

  return iter(ds)
