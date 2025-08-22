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

"""Library for building the synthetic topics data optimization problem."""

import dataclasses

import jax
import jax.numpy as jnp

from topics_api_data_release import pairwise_marginal_queries

PairwiseMarginalQueryBatch = (
    pairwise_marginal_queries.PairwiseMarginalQueryBatch
)


@dataclasses.dataclass(frozen=True)
class OptimizationTarget:
  """Represents a single optimization target.

  An instance of this class represents a single pairwise marginal query together
  with the target value for that query and a weight for this target in the
  optimization objective.

  This class can represent targets for several kinds of queries:

  If week_0 == week_1 and topic_0 == topic_1, then the target corresponds to the
  frequency of observing topic_0 in week_0.

  If week_0 == week_1 and topic_0 != topic_1, then the target corresponds to the
  frequency of observing both topic_0 and topic_1 in week_0.

  Finally, if week_0 != week_1 (and regardless of whether topic_0 == topic_1),
  then the target corresponds to the frequency of observing topic_0 in week_0
  and topic_1 in week_1.

  Attributes:
    week_0: The week index of the first topic.
    topic_0: The topic index of the first topic.
    week_1: The week index of the second topic.
    topic_1: The topic index of the second topic.
    target: The target value for this query.
    weight: The weight of this query in the optimization objective.
  """

  week_0: int
  topic_0: int
  week_1: int
  topic_1: int
  target: float
  weight: float


@dataclasses.dataclass
class TopicsOptimizationProblem:
  """Represents the optimization problem for the topics data release.

  Attributes:
    queries: A collection of marginal queries encoding the measured statistics.
    targets: targets[i] is the target value for queries[i].
    weights: weights[i] is the weight of queries[i].
  """

  queries: PairwiseMarginalQueryBatch
  targets: jax.Array
  weights: jax.Array


class TopicsOptimizationProblemBuilder:
  """A helper class for constructing TopicsOptimizationProblem instances.

  This class represents the optimization problem using python lists so that they
  can be incrementally constructed. The build() method converts the internal
  lists into jax arrays and returns a TopicsOptimizationProblem instance.
  """

  _optimization_targets: list[OptimizationTarget]

  def __init__(self):
    self._optimization_targets = []

  def add_optimization_target(self, opt_target):
    self._optimization_targets.append(opt_target)

  def build(self):
    """Returns the built TopicsOptimizationProblem instance."""
    week_indices_list = []
    topic_indices_list = []
    targets_list = []
    weights_list = []

    for opt_target in self._optimization_targets:
      week_indices_list.append([opt_target.week_0, opt_target.week_1])
      topic_indices_list.append([opt_target.topic_0, opt_target.topic_1])
      targets_list.append(opt_target.target)
      weights_list.append(opt_target.weight)

    queries = PairwiseMarginalQueryBatch(
        week_indices=jnp.array(week_indices_list),
        topic_indices=jnp.array(topic_indices_list),
    )
    targets = jnp.array(targets_list)
    weights = jnp.array(weights_list)

    return TopicsOptimizationProblem(
        queries=queries, targets=targets, weights=weights
    )


def create_topics_optimization_problem(
    num_weeks,
    topics,
    single_topic_stats,
    within_week_stats,
    across_week_stats,
):
  """Creates a TopicsOptimizationProblem to fit num_weeks of data to statistics.

  Args:
    num_weeks: The number of weeks of data to fit.
    topics: The list of unique topic ids. The optimizer works with topic indices
      in the range [0, len(topics)), and this list specifies the mapping from
      topic indices to the canonical topic ids.
    single_topic_stats: A dict such that single_topic_stats[topic] is the rate
      we expect to see topic in each week. Any omitted topics are assumed to
      have a target rate of 0.
    within_week_stats: A dict such that within_week_stats[(topic_0, topic_1)] is
      the rate we expect to see topic_0 and topic_1 in the same week. Any
      omitted pairs are assumed to have a target rate of 0.
    across_week_stats: A dict such that across_week_stats[(topic_0, topic_1)] is
      the rate we expect to see topic_0 in week_0 and topic_1 in week_1. Any
      omitted pairs are assumed to have a target rate of 0.

  Returns:
    A TopicsOptimizationProblem instance for fitting num_weeks of data to the
    provided statistics.
  """
  builder = TopicsOptimizationProblemBuilder()

  # Add all single-topic targets:
  for topic_ix in range(len(topics)):
    for week in range(num_weeks):
      target = single_topic_stats.get(topics[topic_ix], 0.0)
      builder.add_optimization_target(
          OptimizationTarget(
              week_0=week,
              topic_0=topic_ix,
              week_1=week,
              topic_1=topic_ix,
              target=target,
              weight=1.0,
          )
      )

  # Add all within-week targets:
  for topic_ix_0 in range(len(topics)):
    for topic_ix_1 in range(topic_ix_0 + 1, len(topics)):
      for week in range(num_weeks):
        target = within_week_stats.get(
            (topics[topic_ix_0], topics[topic_ix_1]), 0.0
        )
        builder.add_optimization_target(
            OptimizationTarget(
                week_0=week,
                topic_0=topic_ix_0,
                week_1=week,
                topic_1=topic_ix_1,
                target=target,
                weight=1.0,
            )
        )

  # Add all across-week targets:
  for topic_ix_0 in range(len(topics)):
    for topic_ix_1 in range(len(topics)):
      for week_0 in range(num_weeks - 1):
        target = across_week_stats.get(
            (topics[topic_ix_0], topics[topic_ix_1]), 0.0
        )
        builder.add_optimization_target(
            OptimizationTarget(
                week_0=week_0,
                topic_0=topic_ix_0,
                week_1=week_0 + 1,
                topic_1=topic_ix_1,
                target=target,
                weight=1.0,
            )
        )

  return builder.build()
