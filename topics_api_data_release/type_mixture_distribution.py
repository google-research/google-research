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

"""JAX representation of a mixture of types distribution."""

import json
from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np


@chex.dataclass
class TypeMixtureTopicDistribution:
  """Represents a distribution over sequences of topic sets.

  We fix a number of weeks W and a topic set size k (typically k = 5). This
  class represents a distribution over W-length sequences of topic sets, where
  each set's size is at most k. The distribution is a mixture of types, where
  each type is a simpler distribution over topic set sequences.

  Each type is parameterized by an array of shape [W, k, num_topics]. To sample
  a sequence of topic sets from a type with parameter array theta, we do the
  following: for each week w in 0, ..., W-1, and each slot index i in
  0, ..., k-1, let S[w,k] be a topic id sampled from the distribution given by
  softmax(theta[w, k, :]). Then the topic set for week w is given by the unique
  set of values among S[w,0], S[w,1], ..., S[w,k]. Note that if there are
  repetitions, then the topic set has size smaller than k.

  To sample from the mixture distribution, we pick a type uniformly at random
  and sample from it.

  Attributes:
    theta: A jax array of shape [num_types, W, k, num_topics] where the
      parameters for type t are stored in the slice theta[t, :, :, :].
  """

  theta: jax.Array

  @classmethod
  def initialize_randomly(
      cls,
      rng_key,
      num_types,
      num_weeks,
      num_slots,
      num_topics,
      std_dev = 0.001,
  ):
    """Returns a distribution with parameters drawn iid from a Gaussian.

    Args:
      rng_key: jax random key used to seed randomness.
      num_types: The number of types in the distribution mixture.
      num_weeks: The number of weeks to model with the distribution.
      num_slots: The number of topics per week.
      num_topics: The number of topics in the topic taxonomy.
      std_dev: The standard deviation of the Gaussian distribution used to
        initialize the parameters.
    """
    theta = (
        jax.random.normal(
            rng_key, shape=[num_types, num_weeks, num_slots, num_topics]
        )
        * std_dev
    )
    return cls(theta=theta)

  def get_slot_prob_array(self):
    """Returns an array of the slot topic distributions.

    Returns:
      An array probs of shape [num_types, W, k, num_topics] such for any type
      t, week w, and slot i, the topic in that slot is distributed according
      to the probability vector probs[t, w, i, :].
    """
    return jax.nn.softmax(self.theta, axis=-1)

  def sample_topic_indices(
      self, rng_key, num_samples
  ):
    """Draws num_samples independent samples from the distribution.

    Args:
      rng_key: prng key for sampling.
      num_samples: The number of samples to draw.

    Returns:
     An topic_ixs of shape[num_samples, num_weeks, num_slots] such that the
     entry topic_ixs[i,w,s] is the topic index sampled for slot s of week w
     in the ith sample.
    """

    slot_probs = self.get_slot_prob_array()
    all_cumulative_probs = jnp.cumsum(slot_probs, axis=-1)

    @jax.jit
    def single_sample(rng_key):
      """Draws a single sample from the distribution."""
      num_types, num_weeks, num_slots, num_topics = all_cumulative_probs.shape
      # Pick a type uniformly at random:
      type_key, rng_key = jax.random.split(rng_key)
      type_ix = jax.random.choice(type_key, num_types)
      type_cumulative_probs = all_cumulative_probs[type_ix, :, :, :]

      # Sample a topic for each slot from the type by inverting the CMF via
      # binary search. Note: we vmap twice to map over the weeks and slots.
      def sample_single_topic(
          rng_key, cumulative_probs
      ):
        z = jax.random.uniform(rng_key)
        return jnp.clip(
            jnp.searchsorted(cumulative_probs, z), 0, num_topics - 1
        )

      sample_single_week = jax.vmap(sample_single_topic)
      sample_all_weeks = jax.vmap(sample_single_week)

      slot_keys = jax.random.split(rng_key, (num_weeks, num_slots))
      return sample_all_weeks(slot_keys, type_cumulative_probs)

    sample_keys = jax.random.split(rng_key, num_samples)
    # Note: we use jax.lax.map here instead of vmap to avoid allocating an array
    #  of size [num_samples, num_weeks, num_slots, num_topics] which would
    #  result from instantiating the variable type_cumulative_probs in a vmapped
    #  single_sample function.
    return jax.lax.map(single_sample, sample_keys)

  def format_as_json(self, topic_ids):
    """Returns a JSON representation of the distribution.

    Args:
      topic_ids: A list of topic IDs such that topic_ids[i] is the topic ID of
        the topic with index i.
    """
    dict_repr = {}
    theta = np.array(self.theta)
    num_types, num_weeks, k, num_topics = theta.shape
    for type_ix in range(num_types):
      type_name = f"type {type_ix}"
      dict_repr[type_name] = {}
      for week_ix in range(num_weeks):
        week_name = f"epoch {week_ix}"
        dict_repr[type_name][week_name] = {}
        for slot_ix in range(k):
          slot_name = f"slot {slot_ix}"
          dict_repr[type_name][week_name][slot_name] = {}
          for topic_ix in range(num_topics):
            topic = topic_ids[topic_ix]
            dict_repr[type_name][week_name][slot_name][topic] = float(
                theta[type_ix, week_ix, slot_ix, topic_ix]
            )
    return json.dumps(dict_repr, indent=2)

  @classmethod
  def from_json(
      cls, json_repr, topic_ids
  ):
    """Returns a TypeMixtureTopicDistribution from a JSON representation.

    Args:
      json_repr: A JSON representation of the distribution.
      topic_ids: A list of topic IDs such that topic_ids[i] is the topic ID of
        the topic with index i.
    """
    dict_repr = json.loads(json_repr)

    # Determine the shape of theta.
    num_types = len(dict_repr)
    num_epochs = len(dict_repr["type 0"])
    num_slots = len(dict_repr["type 0"]["epoch 0"])
    num_topics = len(dict_repr["type 0"]["epoch 0"]["slot 0"])

    theta = np.empty(shape=[num_types, num_epochs, num_slots, num_topics])
    for type_ix in range(num_types):
      type_dict = dict_repr[f"type {type_ix}"]
      for epoch_ix in range(num_epochs):
        epoch_dict = type_dict[f"epoch {epoch_ix}"]
        for slot_ix in range(num_slots):
          slot_dict = epoch_dict[f"slot {slot_ix}"]
          for topic_ix, topic_id in enumerate(topic_ids):
            theta[type_ix, epoch_ix, slot_ix, topic_ix] = slot_dict[
                str(topic_id)
            ]

    return cls(theta=jnp.array(theta))
