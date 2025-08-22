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

"""Optimization loop for fitting TypeMixtureTopicDistributions to statistics."""

import math
import time
from typing import Callable

from absl import logging
import jax
import jax.numpy as jnp
import jax.profiler
import optax

from topics_api_data_release import pairwise_marginal_queries
from topics_api_data_release import type_mixture_distribution

TypeMixtureTopicDistribution = (
    type_mixture_distribution.TypeMixtureTopicDistribution
)
PairwiseMarginalQueryBatch = (
    pairwise_marginal_queries.PairwiseMarginalQueryBatch
)

# This type represents a function that takes two jax arrays of input statistics
# of shape [batch_size] and returns a jax array of shape [batch_size] containing
# the per-query losses.
LossFn = Callable[[jax.Array, jax.Array], jax.Array]


def fit_distribution(
    rng_key,
    initial_distribution,
    queries,
    targets,
    batch_size,
    num_epochs,
    loss_fn,
    optimizer,
    weights = None,
):
  """Fits a TypeMixtureTopicDistribution to queries and targets.

  In particular, this function uses minibatch gradient descent to search for a
  TypeMixtureTopicDistribution dist such that queries.evaluate(dist) is close to
  targets (according to the provided loss function). Each minibatch consists of
  a query_batch, a target_batch, and a weights_batch which are subsets of the
  complete set of queries, targets, and weights. The update step performed on
  that minibatch adjusts the parameters of the TypeMixtureTopicDistribution to
  minimize

    jnp.mean(loss_fn(query_batch.evaluate(dist), target_batch) * weights_batch).

  We process the queries in epochs. On each epoch, the queries and targets are
  randomly permuted, and we construct consecutive minibatches of size
  batch_size. If the total number of queries is not divisible by batch_size,
  then the final minibatch in each epoch will have size smaller than batch_size.

  The minibatch loss gradients are transformed into parameter updates using the
  provided optax.GradientTransformation. This is also where optimization
  hyperparameters like the learning rate should be specified.

  Args:
    rng_key: A jax random key used to generate shuffle permutations for the
      queries on each epoch of fitting.
    initial_distribution: A TypeMixtureTopicDistribution to start the
      optimization from. This initial point is also how we specify the number of
      types, weeks, slots, and topics.
    queries: A PairwiseMarginalQueryBatch encoding the queries we want to fit.
    targets: For each index i, we want the value of queries.evaluate(dist)[i] to
      be close to targets[i].
    batch_size: The number of queries to process in a single gradient update.
    num_epochs: The number of passes through the collection of queries. In
      particular, each query will be used in num_epochs gradient updates.
    loss_fn: A function that takes as input two jax arrays of shape [batch_size]
      and outputs an array of shape [batch_size] containing per-query losses.
    optimizer: The optimizer to use during fitting. This is also where
      optimization hyperparameters like step size / learning rate are specified.
    weights: An optional jax.Array of size [num_queries] such that weights[i] is
      a weight for query i. The loss associated with query i is scaled by
      weights[i] during optimization.

  Returns:
    The final iterate TypeMixtureTopicDistribution.
  """
  num_queries = queries.num_queries()
  num_batches = math.ceil(num_queries / batch_size)

  if weights is None:
    weights = jnp.ones([num_queries], dtype=jnp.float32)

  @jax.jit
  def evaluate_loss(dist, query_indices):
    query_batch = queries[query_indices]
    target_batch = targets[query_indices]
    weights_batch = weights[query_indices]
    dist_values = query_batch.evaluate(dist)
    return jnp.mean(loss_fn(dist_values, target_batch) * weights_batch)

  @jax.jit
  def step(dist, opt_state, query_indices):
    loss, grads = jax.value_and_grad(evaluate_loss)(dist, query_indices)
    updates, opt_state = optimizer.update(grads, opt_state)
    dist = optax.apply_updates(dist, updates)
    return dist, opt_state, loss

  dist = initial_distribution
  opt_state = optimizer.init(dist)
  global_step = 0

  for epoch_ix in range(num_epochs):
    epoch_start_time = time.time()
    rng_key, shuffle_key = jax.random.split(rng_key)
    query_permutation = jax.random.permutation(shuffle_key, num_queries)
    epoch_total_loss = jnp.array(0.0)

    for batch_ix in range(num_batches):
      global_step += 1
      with jax.profiler.StepTraceAnnotation("batch", step_num=global_step):
        query_indices = query_permutation[
            jnp.arange(
                batch_ix * batch_size,
                jnp.minimum(num_queries, (batch_ix + 1) * batch_size),
            )
        ]
        dist, opt_state, batch_loss = step(dist, opt_state, query_indices)
        epoch_total_loss += batch_loss * len(query_indices)

    epoch_end_time = time.time()
    logging.info(
        "Epoch %d: took %.2f seconds, average loss: %g",
        epoch_ix,
        epoch_end_time - epoch_start_time,
        epoch_total_loss / num_queries,
    )

  return dist
