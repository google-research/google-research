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

"""Library for simulating the Topics API."""

from typing import Dict, List, Iterable

import apache_beam as beam
import numpy as np
import numpy.typing as npt
import tensorflow as tf

TopicProfile = Dict[int, float]


class SimulateTopicsAPIFn(beam.DoFn):
  """Simulates the TopicsAPI.

  This DoFn simulates the Topics API for a collection of users over a sequence
  of epochs. In particular, for each user in the input, we perform the following
  operations:

  1. Extract TopicProfiles from the user's tf.train.Example proto for each of
     the `num_epochs` epochs. A topic profile is a dictionary mapping topic ids
     to weights, where the weight is an approximate measure of user interest.
  2. Compute a set of top_k topics for each epoch. The top_k topics are chosen
     to be the highest weight topics with ties broken randomly. If there are
     fewer than top_k topics in the profile, random topics are drawn from
     all_topics and added to the top topic set until it is of size top_k.
  3. For each epoch, topics are generated for two websites independently. The
     distribution is as follows: with probability prob_random_choice, a topic is
     drawn from all_topics uniformly at random. Otherwise, a topic is chosen
     uniformly at random from the top topic set for that epoch.
  4. Finally, the set of top topics per epoch, and the topics sampled for
     websites one and two are encoded as a tf.train.Example.

  Note: Since the down-stream pipelines are not specific to the Topics API, the
  output format calls each topic a "token", website_1 is "client_1" and
  website_2 is client_2.

  For the expected format of the input tf.train.Examples, see the docstring for
  `extract_topic_profiles`. For the format of output tf.train.Examples, see the
  docstring for `build_topic_simulator_output`.

  Attributes:
    rng: Internal rng.
    num_epochs: The expected number of epochs in the input data.
    top_k: The number of topics to select per week.
  """
  rng: np.random.Generator
  num_epochs: int
  top_k: int
  prob_random_choice: float

  def __init__(self, *, num_epochs, top_k, prob_random_choice):
    super().__init__()
    self.rng = np.random.default_rng()
    self.num_epochs = num_epochs
    self.top_k = top_k
    self.prob_random_choice = prob_random_choice

  def process(
      self, example, all_topics
  ):
    all_topics_array = np.array(all_topics)

    if len(all_topics_array) != len(set(all_topics_array)):
      raise ValueError("The topics in all_topics must be unique.")

    user_id = example.features.feature["user_id"].bytes_list.value[0]
    topic_profiles = extract_topic_profiles(example, self.num_epochs)
    epoch_top_topics = [
        extract_top_k_topics(
            self.rng, topic_profile, self.top_k, all_topics_array
        )
        for topic_profile in topic_profiles
    ]
    website_1_topics = [
        sample_topic(
            self.rng, top_topics, self.prob_random_choice, all_topics_array
        )
        for top_topics in epoch_top_topics
    ]
    website_2_topics = [
        sample_topic(
            self.rng, top_topics, self.prob_random_choice, all_topics_array
        )
        for top_topics in epoch_top_topics
    ]

    return [
        build_topics_simulator_output(user_id, epoch_top_topics,  # pytype: disable=wrong-arg-types  # typed-numpy
                                      website_1_topics, website_2_topics)
    ]


def extract_top_k_topics(rng, topic_profile,
                         top_k,
                         all_topics):
  """Extract the top-k topics from a given topic profile.

  Chooses the top_k highest weight topics from the topic profile, breaking ties
  randomly using the provided rng. If there are fewer than top-k topics in the
  profile, then all topics from the profile are taken and random topics are
  drawn from all_topics until a total of top_k distinct topics is reached.

  Args:
    rng: The random number generator to use for tie breaking and padding.
    topic_profile: The topic profile to select topics from.
    top_k: The number of top topics to select.
    all_topics: A numpy vector containing each topic in the domain / taxonomy.

  Returns:
    A numpy vector containing the top_k topics from the profile.

  Raises:
    ValueError if top_k is larger than the length of all_topics.
  """
  if top_k > len(all_topics):
    raise ValueError(
        "top_k must not be larger than the length of all_topics, but got top_k"
        f" = {top_k} and len(all_topics) = {len(all_topics)}"
    )

  topics = list(topic_profile.keys())
  weights = [topic_profile[id] for id in topics]

  if len(topic_profile) >= top_k:
    # If there are at least top_k topics, choose the highest weighted topics and
    # break ties according to a random permutation sampled from rng.
    tie_break = rng.permutation(len(topics))
    sorted_topics = list(zip(weights, tie_break, topics))
    sorted_topics.sort(reverse=True)
    return np.array([topic_id for (_, _, topic_id) in sorted_topics[:top_k]])

  else:
    # Whenever there are fewer than top_k topics in the profile, we take all of
    # the profile topics and pad with randomly drawn topics from all_topics.
    chosen_topics = set(iter(topic_profile.keys()))
    while len(chosen_topics) < top_k:
      chosen_topics.add(rng.choice(all_topics))
    return np.array(list(chosen_topics))


def sample_topic(rng, user_topics,
                 prob_random_choice,
                 all_topics):
  """Sample a topic for one user on one epoch following the Topics API.

  With probability prob_random_choice, output a uniformly random element of
  all_topics, otherwise output a uniformly random element of user_topics.

  Args:
    rng: The random number generator to use.
    user_topics: The vector of top k topics for the user.
    prob_random_choice: The probability of returning a random topic from the
      entire topic taxonomy.
    all_topics: A vector containing all topics in the taxonomy.

  Returns:
    The integer ID of the sampled topic.
  """
  if rng.random() < prob_random_choice:
    return rng.choice(all_topics)
  else:
    return rng.choice(user_topics)


def extract_topic_profiles(example,
                           num_epochs):
  """Given a tf.train.Example, returns a list of TopicProfiles for each epoch.

  This function assumes that the example encodes the topic profiles for a user
  over `num_epochs` epochs and includes the following features:

  - `user_id`: A bytes feature that contains the user id as its only value. Note
    that user ids must be unique across the dataset.
  - `epoch_{t}_topics` for `t in range(num_epochs)`: An int feature that
    contains the topics in the user's profile for epoch `t`.
  - `epoch_{t}_weights` for `t in range(num_epochs)`: A float feature that
    contains the topic weights for epoch `t` (in the same order as the topics in
    `epoch_{t}_topics`).

  Args:
    example: The tf.train.Example to extract the topic profiles from.
    num_epochs: The number of epochs in the data.

  Returns:
    A list epoch_profiles where epoch_profiles[i] is the topic profile for epoch
    i.

  Raises:
    Value error if the example does not include features called
    "epoch_{t}_topics" and "epoch_{t}_weights" for all t in range(num_epochs).
  """
  epoch_profiles = []
  for t in range(num_epochs):
    ids_fname = f"epoch_{t}_topics"
    weights_fname = f"epoch_{t}_weights"
    if ids_fname not in example.features.feature.keys():
      raise ValueError(f"Example must contain feature {ids_fname}.")
    if weights_fname not in example.features.feature.keys():
      raise ValueError(f"Example must contain feature {weights_fname}.")

    ids = np.array(example.features.feature[ids_fname].int64_list.value)
    weights = np.array(example.features.feature[weights_fname].float_list.value)
    epoch_profiles.append({id: weight for (id, weight) in zip(ids, weights)})
  return epoch_profiles


def build_topics_simulator_output(
    user_id, epoch_top_topics,
    website_1_topics,
    website_2_topics):
  """Encodes a user's top topics over several epochs and sampled topics for websites as a tf.train.Example.

  Let `T` be the number of epochs. This function produces a tf.train.Example
  with the following features:

  - `user_id`: The user's ID.
  - `epoch_{t}_tokens` for `t in range(T)`: An int64 feature that contains the
    top topics for this user during epoch `t`.
  - `client_1_tokens`: An int64 feature with `T` values where the `t`th value
    is the topic sampled for this user on website one in epoch `t`.
  - `client_2_tokens`: An int64 feature with `T` values where the `t`th value
    is the topic sampled for this user on website two in epoch `t`.

  Args:
    user_id: The user's ID.
    epoch_top_topics: A list of `T` numpy vectors, where epoch_top_topics[t] is
      the user's top topics for epoch t.
    website_1_topics: A numpy vector of length T containing the topics sampled
      for this user on website 1.
    website_2_topics: A numpy vector of length T containing the topics sampled
      for this user on website 2.

  Returns:
    A tf.train.Example encoding the epoch_top_topics and website_sampled_topics.

  Raises:
    ValueError if user_id is empty, or epoch_topics, website_1_topics, and
    website_2_topics do not have the same length.
  """
  if not user_id:
    raise ValueError('user_id must not be empty, but got "{user_id}"')
  if len(epoch_top_topics) != len(website_1_topics) or len(
      epoch_top_topics
  ) != len(website_2_topics):
    raise ValueError(
        "epoch_top_topics, website_1_topics, and website_2_topics must have the"
        " same lenghths."
    )

  features = {
      "user_id":
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[user_id])),
      "client_1_tokens":
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=website_1_topics)),
      "client_2_tokens":
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=website_2_topics))
  }

  for (t, top_topics) in enumerate(epoch_top_topics):
    features[f"epoch_{t}_tokens"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=top_topics))

  return tf.train.Example(features=tf.train.Features(feature=features))
