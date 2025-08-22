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

import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from topics_api_data_release import pairwise_marginal_queries
from topics_api_data_release import type_mixture_distribution

TypeMixtureTopicDistribution = (
    type_mixture_distribution.TypeMixtureTopicDistribution
)
PairwiseMarginalQueryBatch = (
    pairwise_marginal_queries.PairwiseMarginalQueryBatch
)


class PairwiseMarginalQueryBatchBuilder:
  """Helper class for building PairwiseMarginalQueryBatches and targets."""

  weeks: list[tuple[int, int]]
  topics: list[tuple[int, int]]
  targets: list[float]

  def __init__(self):
    self.weeks = []
    self.topics = []
    self.targets = []

  def add_query(
      self,
      w0,
      t0,
      w1,
      t1,
      target,
  ):
    """Add a new query to the batch.

    Calling this method with arguments w0, t0, w1, t1, and expected_rate adds a
    new query to the batch that computes the cooccurrence rate of topic t0 in
    week w0 and topic t1 in week w1. The expected_rate is the value we expect
    when this query is evaluated on the test TypeMixtureTopicDistribution.

    Args:
      w0: The first week.
      t0: The first topic.
      w1: The second week.
      t1: The second topic.
      target: The value we expect for this query.

    Returns:
      Returns self so that add_query calls can be chained.
    """
    self.weeks.append((w0, w1))
    self.topics.append((t0, t1))
    self.targets.append(target)
    return self

  def build(self):
    """Returns the built set of queries as a PairwiseMarginalQueryBatch."""
    weeks_arr = jnp.array(self.weeks, dtype=jnp.int32)
    topics_arr = jnp.array(self.topics, dtype=jnp.int32)
    return PairwiseMarginalQueryBatch(weeks_arr, topics_arr)

  def get_targets(self):
    """Returns the targets for the queries as a jax Array."""
    return jnp.array(self.targets, dtype=jnp.float32)


class PairwiseMarginalQueriesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="batch_size=1", batch_size=1),
      dict(testcase_name="batch_size=10", batch_size=10),
      dict(testcase_name="batch_size=123", batch_size=123),
  )
  def test_output_shape_is_batch_size(self, batch_size):
    query_batch = PairwiseMarginalQueryBatch(
        week_indices=jnp.zeros([batch_size, 2], dtype=jnp.int32),
        topic_indices=jnp.zeros([batch_size, 2], dtype=jnp.int32),
    )

    dist = TypeMixtureTopicDistribution.initialize_randomly(
        rng_key=jax.random.PRNGKey(0),
        num_types=10,
        num_weeks=5,
        num_slots=5,
        num_topics=100,
    )

    chex.assert_shape(query_batch.evaluate(dist), [batch_size])

  def test_get_item_returns_query_subset(self):
    all_queries = (
        PairwiseMarginalQueryBatchBuilder()
        .add_query(w0=0, t0=0, w1=0, t1=0, target=0)
        .add_query(w0=1, t0=1, w1=1, t1=1, target=0)
        .add_query(w0=2, t0=2, w1=2, t1=2, target=0)
        .build()
    )

    first_query = (
        PairwiseMarginalQueryBatchBuilder()
        .add_query(w0=0, t0=0, w1=0, t1=0, target=0)
        .build()
    )
    second_query = (
        PairwiseMarginalQueryBatchBuilder()
        .add_query(w0=1, t0=1, w1=1, t1=1, target=0)
        .build()
    )
    third_query = (
        PairwiseMarginalQueryBatchBuilder()
        .add_query(w0=2, t0=2, w1=2, t1=2, target=0)
        .build()
    )
    first_and_second_query = (
        PairwiseMarginalQueryBatchBuilder()
        .add_query(w0=0, t0=0, w1=0, t1=0, target=0)
        .add_query(w0=1, t0=1, w1=1, t1=1, target=0)
        .build()
    )
    first_and_third_query = (
        PairwiseMarginalQueryBatchBuilder()
        .add_query(w0=0, t0=0, w1=0, t1=0, target=0)
        .add_query(w0=2, t0=2, w1=2, t1=2, target=0)
        .build()
    )

    chex.assert_trees_all_equal(all_queries[0], first_query)
    chex.assert_trees_all_equal(all_queries[1], second_query)
    chex.assert_trees_all_equal(all_queries[2], third_query)
    chex.assert_trees_all_equal(all_queries[0:2], first_and_second_query)
    chex.assert_trees_all_equal(all_queries[[0, 2]], first_and_third_query)
    chex.assert_trees_all_equal(all_queries[[0, 1, 2]], all_queries)
    chex.assert_trees_all_equal(all_queries[:], all_queries)

  @parameterized.named_parameters(
      dict(
          testcase_name="deterministic slots",
          theta_list=[[
              # week 0
              [
                  [100, 0, 0],  # slot 0
                  [0, 100, 0],  # slot 1
                  [100, 0, 0],  # slot 2
              ],
              # week 1
              [
                  [0, 0, 100],  # slot 0
                  [0, 0, 100],  # slot 1
                  [0, 0, 100],  # slot 2
              ],
          ]],
          query_builder=PairwiseMarginalQueryBatchBuilder()
          .add_query(w0=0, t0=0, w1=0, t1=0, target=1.0)
          .add_query(w0=0, t0=1, w1=0, t1=1, target=1.0)
          .add_query(w0=0, t0=2, w1=0, t1=2, target=0.0)
          .add_query(w0=0, t0=0, w1=0, t1=1, target=1.0)
          .add_query(w0=0, t0=0, w1=0, t1=2, target=0.0)
          .add_query(w0=0, t0=1, w1=1, t1=2, target=1.0)
          .add_query(w0=0, t0=2, w1=1, t1=1, target=0.0),
      ),
      dict(
          testcase_name="uniform slots",
          theta_list=[[
              # week 0
              [
                  [0, 0, 0],  # slot 0
                  [0, 0, 0],  # slot 1
                  [0, 0, 0],  # slot 2
              ],
              # week 1
              [
                  [0, 0, 0],  # slot 0
                  [0, 0, 0],  # slot 1
                  [0, 0, 0],  # slot 2
              ],
          ]],
          query_builder=PairwiseMarginalQueryBatchBuilder()
          # Single (week, topic) query:
          .add_query(w0=0, t0=0, w1=0, t1=0, target=1 - (2 / 3) ** 3)
          # Across-weeks query:
          .add_query(
              w0=0, t0=0, w1=1, t1=0, target=(1 - (2 / 3) ** 3) ** 2
          )
          # Within week query:
          .add_query(
              w0=0,
              t0=0,
              w1=0,
              t1=1,
              target=1 - (2 / 3) ** 3 - (2 / 3) ** 3 + (1 / 3) ** 3,
          ),
      ),
      dict(
          testcase_name="single week and slot probabilities",
          theta_list=[[
              # week 0
              [
                  [math.log(0.7), math.log(0.2), math.log(0.1)],  # slot 0
              ],
          ]],
          query_builder=PairwiseMarginalQueryBatchBuilder()
          .add_query(w0=0, t0=0, w1=0, t1=0, target=0.7)
          .add_query(w0=0, t0=1, w1=0, t1=1, target=0.2)
          .add_query(w0=0, t0=2, w1=0, t1=2, target=0.1),
      ),
      dict(
          testcase_name="multiple types are averaged",
          theta_list=[
              [  # Type 0
                  [  # week 0
                      [math.log(0.7), math.log(0.2), math.log(0.1)],  # slot 0
                  ],
              ],
              [  # Type 1
                  [  # week 0
                      [math.log(0.3), math.log(0.5), math.log(0.2)],  # slot 0
                  ],
              ],
          ],
          query_builder=PairwiseMarginalQueryBatchBuilder()
          .add_query(w0=0, t0=0, w1=0, t1=0, target=(0.7 + 0.3)/2)
          .add_query(w0=0, t0=1, w1=0, t1=1, target=(0.2 + 0.5)/2)
          .add_query(w0=0, t0=2, w1=0, t1=2, target=(0.1 + 0.2)/2),
      ),
  )
  def test_output_is_correct(self, theta_list, query_builder):
    theta = jnp.array(theta_list, dtype=jnp.float32)
    dist = TypeMixtureTopicDistribution(theta=theta)
    query_batch = query_builder.build()
    expected_output = query_builder.get_targets()
    chex.assert_trees_all_close(query_batch.evaluate(dist), expected_output)


if __name__ == "__main__":
  absltest.main()
