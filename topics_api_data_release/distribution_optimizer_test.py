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
import optax

from topics_api_data_release import distribution_optimizer
from topics_api_data_release import pairwise_marginal_queries
from topics_api_data_release import type_mixture_distribution

TypeMixtureTopicDistribution = (
    type_mixture_distribution.TypeMixtureTopicDistribution
)
PairwiseMarginalQueryBatch = (
    pairwise_marginal_queries.PairwiseMarginalQueryBatch
)
fit_distribution = distribution_optimizer.fit_distribution


def all_marginals(num_weeks, num_topics):
  """Returns a batch containing all possible marginal queries."""
  weeks = []
  topics = []
  for w0 in range(num_weeks):
    for t0 in range(num_topics):
      for w1 in range(num_weeks):
        for t1 in range(num_topics):
          weeks.append([w0, w1])
          topics.append([t0, t1])
  return PairwiseMarginalQueryBatch(jnp.array(weeks), jnp.array(topics))


class DistributionOptimizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="single type, slot, and week",
          theta_list=[[[[math.log(0.7), math.log(0.2), math.log(0.1)]]]],
      ),
      dict(
          testcase_name="more general case",
          theta_list=[
              [  # Type 0
                  [  # Week 0
                      [1, 2, 3],  # Slot 0
                      [0, 0, 0],  # Slot 1
                      [100, 0, 0],  # Slot 2
                  ],
                  # Week 1
                  [
                      [1, 2, 3],  # Slot 0
                      [1, 4, 8],  # Slot 1
                      [10, 5, -10],  # Slot 2
                  ],
              ],
              [  # Type 1
                  [  # Week 0
                      [3, 1, 2],  # Slot 0
                      [3, 4, 10],  # Slot 1
                      [0, 100, 0],  # Slot 2
                  ],
                  # Week 1
                  [
                      [-1, 1, -1],  # Slot 0
                      [4, 4, 4],  # Slot 1
                      [0, 100, 0],  # Slot 2
                  ],
              ],
          ],
      ),
  )
  def test_finds_consistent_distribution(self, theta_list):
    # This test checks to make sure that, given a collection of (query,target)
    # pairs generated from a TypeMixtureTopicDistribution, the optimizer is able
    # to find a distribution that closely matches the targets.
    theta = jnp.array(theta_list)
    [num_types, num_weeks, num_slots, num_topics] = theta.shape
    consistent_dist = TypeMixtureTopicDistribution(theta=theta)
    queries = all_marginals(num_weeks, num_topics)
    targets = queries.evaluate(consistent_dist)

    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_dist = TypeMixtureTopicDistribution.initialize_randomly(
        init_key, num_types, num_weeks, num_slots, num_topics
    )
    final_dist = fit_distribution(
        rng_key=rng_key,
        initial_distribution=initial_dist,
        queries=queries,
        targets=targets,
        batch_size=queries.num_queries(),
        num_epochs=5000,
        loss_fn=lambda guess, target: (guess - target) ** 2,
        optimizer=optax.adam(1e-2),
    )

    chex.assert_trees_all_close(
        targets, queries.evaluate(final_dist), atol=1e-2
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="All weight on 0.75",
          weights_list=[0.0, 1.0],
          expected_topic_0_rate=0.75,
      ),
      dict(
          testcase_name="All weight on 0.25",
          weights_list=[1.0, 0.0],
          expected_topic_0_rate=0.25,
      ),
      dict(
          testcase_name="Equal weight",
          weights_list=[0.5, 0.5],
          expected_topic_0_rate=0.5,
      ),
  )
  def test_weights_are_used(self, weights_list, expected_topic_0_rate):
    """Checks that query weights are used when fit_distribution is called.

    This test creates two conflicting constraints: one requiring that the rate
    of topic 0 is 0.75, the other requiring the rate of topic 0 is 0.25. Then
    we vary the weights on the two constraints and check that the optimized
    distribution has the expected topic 0 rate.

    Args:
      weights_list: A list of two weights to apply to two constraints.
      expected_topic_0_rate: For the given weights, the expected rate of topic 0
        in the minimizer of the weighted objective.
    """
    queries = PairwiseMarginalQueryBatch(
        week_indices=jnp.array([[0, 0], [0, 0]]),
        topic_indices=jnp.array([[0, 0], [0, 0]]),
    )
    targets = jnp.array([0.25, 0.75])
    weights = jnp.array(weights_list)

    rng_key = jax.random.PRNGKey(0)
    init_key, rng_key = jax.random.split(rng_key)

    initial_dist = TypeMixtureTopicDistribution.initialize_randomly(
        init_key, num_types=1, num_weeks=1, num_slots=1, num_topics=2
    )
    final_dist = fit_distribution(
        rng_key=rng_key,
        initial_distribution=initial_dist,
        queries=queries,
        targets=targets,
        batch_size=2,
        num_epochs=1000,
        loss_fn=lambda guess, target: (guess - target) ** 2,
        optimizer=optax.adam(1e-2),
        weights=weights,
    )

    chex.assert_trees_all_close(
        queries.evaluate(final_dist),
        jnp.array([expected_topic_0_rate, expected_topic_0_rate]),
    )

if __name__ == "__main__":
  absltest.main()
