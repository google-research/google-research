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

"""Implementation of queries measuring pairwise topic coocurrence rates."""

import jax
import jax.numpy as jnp

from topics_api_data_release import type_mixture_distribution

TypeMixtureDistribution = type_mixture_distribution.TypeMixtureTopicDistribution


@jax.tree_util.register_pytree_node_class
class PairwiseMarginalQueryBatch:
  """Represents a batch of queries measuring pairwise topic coocurrence rates.

  Each query is parameterized by two (week, topic) pairs. A query with pairs
  (w0, t0) and (w1, t1) measures the probability of observing topic t0 in week
  w0's topic set and simultaneously observing topic t1 in week w1's topic set.

  Attributes:
    week_indices: A jax array of shape [batch_size, 2] such that the week
      parameters of query i are stored in week_indices[i, :].
    topic_indices: A jax array of shape [batch_size, 2] such that the topic
      parameters of query i are stored in topic_indices[i, :].
  """

  week_indices: jax.Array
  topic_indices: jax.Array

  def __init__(
      self,
      week_indices,
      topic_indices,
  ):
    self.week_indices = week_indices
    self.topic_indices = topic_indices

  def num_queries(self):
    """Returns the number of queries in the batch."""
    return self.week_indices.shape[0]

  def __getitem__(self, ixs):
    """Returns a subset of the query batch."""
    # If the index argument is a single integer, we convert it to a slice so
    # that the result of indexing week_indices and topic_indices is still a rank
    # 2 array.
    if isinstance(ixs, int):
      ixs = slice(ixs, ixs + 1)
    return PairwiseMarginalQueryBatch(
        self.week_indices[ixs, :], self.topic_indices[ixs, :]
    )

  @jax.jit
  def evaluate(self, distribution):
    """Evalates the batch of queries on the given distribution.

    Args:
      distribution: The TypeMixtureDistribution to evaluate the query on.

    Returns:
      A jax array result of shape [batch_size] where result[i] is the value of
      query i on the distribution.
    """
    # Strategy: we implement a function that computes a single query on a single
    # type and then vmap it twice to get one that evaluates all queries on all
    # types. Finally, we average over the types to get the query values on the
    # mixture distribution.

    def single_marginal_single_type(
        weeks, topics, probs
    ):
      """Computes the cooccurrence probability for two topics and one type.

      Let (w0, t0), (w1, t1) be such that w0 and w1 are week indices and t0 and
      t1 are topic indices. This function computes the probability

        Pr( (t0 in week w0) and (t1 in week w1) )

      for a sequence of topic sets sampled from one type.

      Args:
        weeks: A jax array of shape [2] with entries [w0, w1].
        topics: A jax array of shape [2] wioth entries [t0, t1].
        probs: A jax array of shape [num_weeks, num_slots, num_topics] such that
          probs[w, i, t] is the probability of observing topic t in slot i for
          week w.

      Returns:
        The probability of observing topic t0 in week w0 and topic t1 in week
        w1.
      """
      # Strategy: We consider three cases:
      #
      #   Case 1: (w0, t0) == (w1, t1)
      #     In this case, the query is equivalent to the probability of
      #     observing topic t0 in week w0. Using the fact that the slots for
      #     each week are independent, we have
      #
      #         Pr(t0 in w0)
      #       = 1.0 - Pr(t0 not in w0)
      #       = 1.0 - Pr(t0 not in any slot of w0)
      #       = 1.0 - prod(1.0 - probs[w0, :, t0])
      #
      #   Case 2: w0 == w1 but t0 != t1 (the "within week" case):
      #     In this case, the query is equivalent to the probability of
      #     observing both topics t0 and t1 in week w0. We can rewrite the
      #     probability as:
      #
      #        Pr(t0 in w0 and t1 in w0)
      #      = 1.0 - Pr(t0 not in w0 or t1 not in w0)
      #      = 1.0 - Pr(t0 not in w0) - Pr(t1 not in w0)
      #            + Pr(t0 not in w0 and t1 not in w1)
      #
      #     The probabilities Pr(t0 not in w0) and Pr(t1 not in w1) can be
      #     computed as in Case 1. The probability that neither t0 nor t1 are
      #     present in w0 is the probability that none of the slots are equal to
      #     t0 or t1, which similarly to Case 1 can be written as
      #
      #         Pr(t0 not in w0 and t1 not in w0)
      #       = Pr(no slots in week w0 are equal to t0 or t1)
      #       = prod(1.0 - probs[w0, :, t0] - probs[w0, :, t1])
      #
      #  Case 3: w0 != w1 (the "across week" case):
      #    Following almost the identical calculation as in Case 2, we have that
      #
      #        Pr(t0 in w0 and t1 in w1)
      #      = 1.0 - Pr(t0 not in w0) - Pr(t1 not in w1)
      #            + Pr(t0 not in w0 and t1 not in w1).
      #
      #    The key difference is that the events t0 not in w0 and t1 not in w1
      #    are independent in this case, we we have
      #
      #        Pr(t0 not in w0 and t1 not in w1)
      #      = Pr(t0 not in w0) * Pr(t1 not in w1)
      #
      #   and each of the factors can be computed the same way as in case 1.

      pr_t0_not_in_w0 = jnp.prod(1.0 - probs[weeks[0], :, topics[0]])
      pr_t1_not_in_w1 = jnp.prod(1.0 - probs[weeks[1], :, topics[1]])
      pr_neither_w0_neq_w1 = pr_t0_not_in_w0 * pr_t1_not_in_w1
      pr_neither_w0_eq_w1 = jnp.prod(
          1.0 - probs[weeks[0], :, topics[0]] - probs[weeks[0], :, topics[1]]
      )

      case_1_prob = 1.0 - pr_t0_not_in_w0
      case_2_prob = (
          1.0 - pr_t0_not_in_w0 - pr_t1_not_in_w1 + pr_neither_w0_eq_w1
      )
      case_3_prob = (
          1.0 - pr_t0_not_in_w0 - pr_t1_not_in_w1 + pr_neither_w0_neq_w1
      )

      return jax.lax.select(
          weeks[0] == weeks[1],
          on_true=jax.lax.select(
              topics[0] == topics[1], on_true=case_1_prob, on_false=case_2_prob
          ),
          on_false=case_3_prob,
      )

    single_marginal_all_types = jax.vmap(
        single_marginal_single_type, in_axes=[None, None, 0]
    )
    all_marginals_all_types = jax.vmap(
        single_marginal_all_types, in_axes=[0, 0, None]
    )

    per_type_marginals = all_marginals_all_types(
        self.week_indices,
        self.topic_indices,
        distribution.get_slot_prob_array(),
    )

    average_marginals = jnp.mean(per_type_marginals, axis=1)

    return average_marginals

  def tree_flatten(self):
    """Implementation of tree_flatten for register_pytree_node_class."""
    children = [self.week_indices, self.topic_indices]
    aux = None
    return children, aux

  @classmethod
  def tree_unflatten(cls, unused_aux, children):
    """Implementation of tree_unflatten for register_pytree_node_class."""
    [week_indices, topic_indices] = children
    return cls(week_indices, topic_indices)
