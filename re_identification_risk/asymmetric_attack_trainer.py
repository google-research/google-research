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

"""Library for training the weights of the asymmetric hamming distance reidentificaiton attack."""

import collections
import math
from typing import Dict, List, Tuple, Iterable

import apache_beam as beam
import tensorflow as tf


TrainAsymmetricAttackInput = collections.namedtuple(
    "ComputeAsymmetricWeightsInput", ["simulator_output", "all_topics"]
)

AsymmetricHammingWeights = collections.namedtuple(
    "AsymmetricHammingWeights", ["match", "mismatch"]
)


class TrainAsymmetricAttack(beam.PTransform):
  """Computes the match and mismatch weights for the asymmetric attack.

  Takes as input an instance of TrainAsymmetricAttackInput, which is a pair of
  two PCollections: simulator_output and all_topics. simulator_output should be
  a PCollection of tensorflow examples output by the Topics API simulator, and
  all_topics should be a PCollection of integer topic ids representing the set
  of all topics in the taxonomy. This transform returns a PCollection of
  (topic_id, weights) pairs, where topic_id is an integer and weights is an
  instance of AsymmetricHammingWeights.

  This transform should be constructed with the following parameters of the
  Topics API: num_epochs, top_k, and prob_random_choice. Optionally, we can
  specify an initial_topic_count, in which case the occurrance count for each
  topic starts from that value instead of 0.
  """

  num_epochs: int
  top_k: int
  prob_random_choice: float
  initial_topic_count: int

  def __init__(
      self,
      *args,
      num_epochs,
      top_k,
      prob_random_choice,
      initial_topic_count = 0,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.num_epochs = num_epochs
    self.top_k = top_k
    self.prob_random_choice = prob_random_choice
    self.initial_topic_count = initial_topic_count

  def expand(self, pcols):
    simulator_output = pcols.simulator_output
    all_topics = pcols.all_topics

    num_users = (
        simulator_output | "Count users" >> beam.combiners.Count.Globally()
    )
    num_topics = all_topics | "Count topics" >> beam.combiners.Count.Globally()
    topic_counts_without_init = (
        simulator_output | "Count website one topics" >> CountWebsiteOneTopics()
    )
    topic_counts = AddInitialCountsInput(
        topic_counts=topic_counts_without_init, all_topics=all_topics
    ) | "Add initial counts" >> AddInitialCounts(self.initial_topic_count)
    attack_weights = (
        topic_counts
        | "Compute attack weights"
        >> beam.ParDo(
            AttackWeightsFromTopicCounts(
                num_epochs=self.num_epochs,
                top_k=self.top_k,
                prob_random_choice=self.prob_random_choice,
            ),
            beam.pvalue.AsSingleton(num_users),
            beam.pvalue.AsSingleton(num_topics),
        )
    )
    return attack_weights


TopicObservationProbs = collections.namedtuple(
    "TopicObservationProbs", ["in_top_k", "out_top_k"]
)


def compute_topic_observation_probs(
    top_k, prob_random_choice, num_topics
):
  """Return the probability that the Topics API emits a topic if it is in a user's top_k or not.

  Args:
    top_k: The number of top topics each user has per epoch.
    prob_random_choice: The probability that the Topics API emits a uniformly
      random topic from the set of all topics.
    num_topics: The total number of topics.

  Returns:
    An instance of TopicObservationProbs containing two probabilities: the
    probability that the Topics API emits a topic when it is either in a user's
    top_k topics, or not.
  """
  return TopicObservationProbs(
      in_top_k=(1 - prob_random_choice) / top_k
      + prob_random_choice / num_topics,
      out_top_k=prob_random_choice / num_topics,
  )


def estimate_topic_top_k_prob(
    num_users,
    num_epochs,
    topic_count,
    observation_probs,
):
  """Estimates the probability of observing a topic in a user's top k topic set.

  This function makes the same assumptions as section 8.3.2 of the paper
  "Measuring Re-identificaion Risk".

  Args:
    num_users: The number of users in the input data.
    num_epochs: The number of epochs we observe users for.
    topic_count: The total number of time this topic appeared in the data (i.e.,
      the number of user x epoch pairs for which this topic appeared on website
      one).
    observation_probs: A named tuple describing the probability of observing a
      topic that is in a user's top-k topics or not.

  Returns:
    An estimate of the probability that this topic will appear in a user's top-k
    set.
  """
  q_in = observation_probs.in_top_k
  q_out = observation_probs.out_top_k
  return 1 / (q_in - q_out) * (topic_count / (num_users * num_epochs) - q_out)


def compute_attack_weights(
    top_k,
    topic_top_k_prob,
    observation_probs,
):
  """Computes the agreement and disagreement weights for one topic in the Asymmetric Hamming Distance attack.

  Args:
    top_k: The number of top topics per user per epoch.
    topic_top_k_prob: The probability of observing this topic in a users top-k
      set. Should be the output of `compute_topic_top_k_prob`.
    observation_probs: The probabilities of observing a topic given that it is
      in a user's top-k set or not. Should be the output of
      `compute_topic_observation_probs`.

  Returns:
    An instance of AsymmetricHammingWeights containing the agreement and
    disagreement weights for the topic.
  """
  q_in = observation_probs.in_top_k
  q_out = observation_probs.out_top_k

  match_weight = -math.log(
      q_out
      + (q_in - q_out)
      * q_in
      * topic_top_k_prob
      / (q_out + (q_in - q_out) * topic_top_k_prob)
  )

  mismatch_weight = -math.log(
      q_out
      + (q_in - q_out)
      * (top_k - 1)
      * topic_top_k_prob
      / (top_k - topic_top_k_prob)
  )

  return AsymmetricHammingWeights(match=match_weight, mismatch=mismatch_weight)


def extract_website_one_topics(example):
  """Extracts the website one topics for the user represented by example.

  Args:
    example: An tf.train.Example encoding the output of the Topics API simulator
      for one user.

  Returns:
    A list of topic IDs containing the topics that were observed on website one
    for this user.

  Raises:
    ValueError if the example is missing the feature "website_1_topics".
  """
  if "client_1_tokens" not in example.features.feature.keys():
    raise ValueError("example must contain feature client_1_tokens.")
  return list(example.features.feature["client_1_tokens"].int64_list.value)


class CountWebsiteOneTopics(beam.PTransform):
  """Counts the number of times each topic appeared on website one."""

  def expand(self, simulator_output):
    w1_topics = simulator_output | "Extract website one topics" >> beam.FlatMap(
        extract_website_one_topics
    )
    topic_counts = w1_topics | beam.combiners.Count.PerElement()
    return topic_counts

AddInitialCountsInput = collections.namedtuple(
    "AddInitialCountsInput", ["topic_counts", "all_topics"]
)


def _add_initial_topic_count_and_warn(
    topic_and_counts
):
  """Adds the initial topic count to each topic and warns about 0 counts."""
  topic = topic_and_counts[0]
  counts = topic_and_counts[1]

  data_count = list(counts["data_count"])
  initial_count = list(counts["initial_count"])

  if (not data_count or data_count[0] == 0) and initial_count[0] == 0:
    raise ValueError(
        "When initial_topic_count == 0 are added, every topic must appear at"
        f" least once in some user's website_1_topics, but topic {topic} did"
        " not appear."
    )

  total = initial_count[0]
  if data_count:
    total += data_count[0]

  return (topic, total)


class AddInitialCounts(beam.PTransform):
  """Adds an initial count to each topic's count.

  Raises:
    ValueError if self.num_to_add is 0 and there exists some topic from
    all_topics that either does not appear in topic_counts or has a count of 0.
  """

  num_to_add: int

  def __init__(self, num_to_add, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_to_add = num_to_add

  def expand(self, inputs):
    topic_counts = inputs.topic_counts
    all_topics = inputs.all_topics

    initial_counts = all_topics | beam.Map(
        lambda topic: (topic, self.num_to_add)
    )

    joined_counts = {
        "data_count": topic_counts,
        "initial_count": initial_counts,
    } | "Group data and initial counts" >> beam.CoGroupByKey()

    total_counts = joined_counts | beam.Map(_add_initial_topic_count_and_warn)

    return total_counts


class AttackWeightsFromTopicCounts(beam.DoFn):
  """Computes attack weights for topics given a PCollection of topic counts.

  This DoFn takes as input a PCollection of (topic_id, count) pairs, together
  with two side inputs: singleton PCollections containing the number of users
  and topics, respectively.

  This DoFn should be constructed with the following Topics API parameters:
  num_epochs, top_k, prob_random_choice.

  The resulting PCollection is a collection of (topic_id, weight) pairs, where
  each weight is an instance of AsymmetricHammingWeights.
  """

  num_epochs: int
  top_k: int
  prob_random_choice: float

  def __init__(self, *, num_epochs, top_k, prob_random_choice):
    self.num_epochs = num_epochs
    self.top_k = top_k
    self.prob_random_choice = prob_random_choice

  def process(self, element, num_users, num_topics):
    topic, topic_count = element

    obs_probs = compute_topic_observation_probs(
        self.top_k, self.prob_random_choice, num_topics
    )
    topic_top_k_prob = estimate_topic_top_k_prob(
        num_users, self.num_epochs, topic_count, obs_probs
    )
    attack_weights = compute_attack_weights(
        self.top_k, topic_top_k_prob, obs_probs
    )
    return [(topic, attack_weights)]


def format_weights_as_json_row(
    topic_weights
):
  """Formats the asymmetric hamming weights for a topic as a json file row."""
  topic, weights = topic_weights
  return (
      f'{{"token_id": {topic}, "match_weight": {weights.match},'
      f' "mismatch_weight": {weights.mismatch}}}'
  )
