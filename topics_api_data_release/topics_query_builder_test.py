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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
from topics_api_data_release import topics_query_builder

create_topics_optimization_problem = (
    topics_query_builder.create_topics_optimization_problem
)
OptimizationTarget = topics_query_builder.OptimizationTarget
TopicsOptimizationProblemBuilder = (
    topics_query_builder.TopicsOptimizationProblemBuilder
)


class TopicsQueryBuilderTest(parameterized.TestCase):

  def test_topics_optimization_problem_builder_is_correct(self):
    builder = TopicsOptimizationProblemBuilder()

    # Single topic target:
    builder.add_optimization_target(
        topics_query_builder.OptimizationTarget(
            week_0=0,
            topic_0=0,
            week_1=0,
            topic_1=0,
            target=0.1,
            weight=1.0,
        )
    )

    # Within week target:
    builder.add_optimization_target(
        topics_query_builder.OptimizationTarget(
            week_0=1, topic_0=1, week_1=1, topic_1=2, target=0.2, weight=2.0
        )
    )

    # Across week target:
    builder.add_optimization_target(
        topics_query_builder.OptimizationTarget(
            week_0=2,
            topic_0=3,
            week_1=3,
            topic_1=4,
            target=0.3,
            weight=3.0,
        )
    )

    problem = builder.build()
    chex.assert_trees_all_equal(
        problem.queries.week_indices, jnp.array([[0, 0], [1, 1], [2, 3]])
    )
    chex.assert_trees_all_equal(
        problem.queries.topic_indices, jnp.array([[0, 0], [1, 2], [3, 4]])
    )
    chex.assert_trees_all_equal(problem.targets, jnp.array([0.1, 0.2, 0.3]))
    chex.assert_trees_all_equal(problem.weights, jnp.array([1.0, 2.0, 3.0]))

  @parameterized.named_parameters(
      dict(
          testcase_name="fully specified 2 weeks",
          num_weeks=2,
          topics=[10, 20],
          single_topic_stats={10: 0.1, 20: 0.2},
          within_week_stats={(10, 20): 0.3},
          across_week_stats={
              (10, 10): 0.4,
              (10, 20): 0.5,
              (20, 10): 0.6,
              (20, 20): 0.7,
          },
      ),
      dict(
          testcase_name="fully specified 10 weeks",
          num_weeks=10,
          topics=[10, 20],
          single_topic_stats={10: 0.1, 20: 0.2},
          within_week_stats={(10, 20): 0.3},
          across_week_stats={
              (10, 10): 0.4,
              (10, 20): 0.5,
              (20, 10): 0.6,
              (20, 20): 0.7,
          },
      ),
      dict(
          testcase_name="partially specified 10 weeks",
          num_weeks=10,
          topics=[10, 20, 30],
          single_topic_stats={10: 0.1, 20: 0.2},
          within_week_stats={(10, 20): 0.3},
          across_week_stats={
              (10, 10): 0.4,
              (10, 20): 0.5,
              (20, 20): 0.7,
              (20, 30): 0.8
          },
      ),
  )
  def test_create_topics_optimization_problem_correct(
      self,
      num_weeks,
      topics,
      single_topic_stats,
      within_week_stats,
      across_week_stats,
  ):
    """Tests that the optimization problem correctly encodes the statistics.

    This involves three related checks:
    1. Every row of the optimization problem matches one of the statistics.
    2. The number of rows in the optimization problem is correct.
    3. None of the rows of the optimization problem encode the same query.

    Taken together, these checks ensure that the optimization problem is
    correct.

    Args:
      num_weeks: The number of weeks to fit in the optimization problem.
      topics: The list of topic ids.
      single_topic_stats: The single-topic statistics to fit.
      within_week_stats: The within-week statistics to fit.
      across_week_stats: The across-week statistics to fit.
    """
    problem = create_topics_optimization_problem(
        num_weeks=num_weeks,
        topics=topics,
        single_topic_stats=single_topic_stats,
        within_week_stats=within_week_stats,
        across_week_stats=across_week_stats,
    )

    # Check 1: Every row matches one of the statistics:
    for i in range(problem.queries.num_queries()):
      w0, w1 = problem.queries.week_indices[i, :]
      t0, t1 = problem.queries.topic_indices[i, :]
      target = problem.targets[i]
      weight = problem.weights[i]
      if w0 == w1:
        if t0 == t1:
          self.assertAlmostEqual(
              target, single_topic_stats.get(topics[t0], 0.0)
          )
        else:
          self.assertAlmostEqual(
              target, within_week_stats.get((topics[t0], topics[t1]), 0.0)
          )
      else:
        self.assertAlmostEqual(
            target, across_week_stats.get((topics[t0], topics[t1]), 0.0)
        )
      self.assertAlmostEqual(weight, 1.0)

    # Check 2: The number of rows is correct:
    num_single_week = num_weeks * len(topics)
    num_within_week = num_weeks * len(topics) * (len(topics) - 1) / 2
    num_across_week = (num_weeks - 1) * len(topics) * len(topics)
    total_queries = num_single_week + num_within_week + num_across_week
    self.assertEqual(problem.queries.num_queries(), total_queries)

    # Check 3: The queries encoded by the rows are unique:
    unique_targets = set()
    for i in range(problem.queries.num_queries()):
      w0, w1 = map(int, problem.queries.week_indices[i, :])
      t0, t1 = map(int, problem.queries.topic_indices[i, :])
      unique_targets.add((w0, t0, w1, t1))
    self.assertLen(unique_targets, problem.queries.num_queries())


if __name__ == "__main__":
  absltest.main()
