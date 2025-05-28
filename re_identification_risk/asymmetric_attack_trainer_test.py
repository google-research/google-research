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

import json
import math

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
import apache_beam.testing.util as beam_test_util
from google.protobuf import text_format
import tensorflow as tf

from re_identification_risk.asymmetric_attack_trainer import AddInitialCounts
from re_identification_risk.asymmetric_attack_trainer import AddInitialCountsInput
from re_identification_risk.asymmetric_attack_trainer import AsymmetricHammingWeights
from re_identification_risk.asymmetric_attack_trainer import compute_attack_weights
from re_identification_risk.asymmetric_attack_trainer import compute_topic_observation_probs
from re_identification_risk.asymmetric_attack_trainer import CountWebsiteOneTopics
from re_identification_risk.asymmetric_attack_trainer import estimate_topic_top_k_prob
from re_identification_risk.asymmetric_attack_trainer import extract_website_one_topics
from re_identification_risk.asymmetric_attack_trainer import format_weights_as_json_row
from re_identification_risk.asymmetric_attack_trainer import TopicObservationProbs
from re_identification_risk.asymmetric_attack_trainer import TrainAsymmetricAttack
from re_identification_risk.asymmetric_attack_trainer import TrainAsymmetricAttackInput


class AsymmetricAttackTrainerTest(parameterized.TestCase, absltest.TestCase):

  def test_train_asymmetric_attack_runs(self):
    # Since the TrainAsymmetricAttack PTransform calls other well-tested
    # methods, this test only checks to make sure that the PTransform runs and
    # outputs the expected number of elements.

    simulator_output = [
        text_format.Parse(
            """
        features: {
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [0, 1, 2]}}
            }
        }""",
            tf.train.Example(),
        ),
        text_format.Parse(
            """
        features: {
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [1, 2, 3]}}
            }
        }""",
            tf.train.Example(),
        ),
        text_format.Parse(
            """
        features: {
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [3, 4, 5]}}
            }
        }""",
            tf.train.Example(),
        ),
    ]
    all_topics = [0, 1, 2, 3, 4, 5]
    with TestPipeline() as p:
      simulator_output_pcoll = p | "Create simulator output" >> beam.Create(
          simulator_output
      )
      all_topics_pcoll = p | "Create all topics" >> beam.Create(all_topics)
      inputs = TrainAsymmetricAttackInput(
          simulator_output=simulator_output_pcoll, all_topics=all_topics_pcoll
      )
      trainer = TrainAsymmetricAttack(
          num_epochs=8,
          top_k=5,
          prob_random_choice=0.05,
          initial_topic_count=0,
      )
      output = inputs | trainer
      output_size = output | beam.combiners.Count.Globally()

      beam_test_util.assert_that(output_size, beam_test_util.equal_to([6]))

  def test_compute_topic_observation_probs(self):
    obs_probs_1 = compute_topic_observation_probs(
        top_k=5, prob_random_choice=0, num_topics=300
    )
    self.assertAlmostEqual(1 / 5, obs_probs_1.in_top_k)
    self.assertAlmostEqual(0, obs_probs_1.out_top_k)

    obs_probs_2 = compute_topic_observation_probs(
        top_k=10, prob_random_choice=0, num_topics=300
    )
    self.assertAlmostEqual(1 / 10, obs_probs_2.in_top_k)
    self.assertAlmostEqual(0, obs_probs_2.out_top_k)

    obs_probs_3 = compute_topic_observation_probs(
        top_k=5, prob_random_choice=0.05, num_topics=300
    )
    self.assertAlmostEqual(0.95 / 5 + 0.05 / 300, obs_probs_3.in_top_k)
    self.assertAlmostEqual(0.05 / 300, obs_probs_3.out_top_k)

  @parameterized.named_parameters(
      dict(
          testcase_name="top_k_prob = 0.01, N=any, k = 5, p=0",
          top_k_prob=0.01,
          observation_probs=TopicObservationProbs(1 / 5, 0),
      ),
      dict(
          testcase_name="top_k_prob = 0.1, N=any, k = 5, p=0",
          top_k_prob=0.1,
          observation_probs=TopicObservationProbs(1 / 5, 0),
      ),
      dict(
          testcase_name="top_k_prob = 0.2, N=any, k = 5, p=0",
          top_k_prob=0.2,
          observation_probs=TopicObservationProbs(1 / 5, 0),
      ),
      dict(
          testcase_name="top_k_prob = 0.2, N=100, k = 5, p=0.05",
          top_k_prob=0.2,
          observation_probs=TopicObservationProbs(
              (1 - 0.05) / 5 + 0.05 / 100, 0.05 / 100
          ),
      ),
      dict(
          testcase_name="top_k_prob = 0.2, N=300, k = 5, p=0.05",
          top_k_prob=0.2,
          observation_probs=TopicObservationProbs(
              (1 - 0.05) / 5 + 0.05 / 300, 0.05 / 300
          ),
      ),
  )
  def test_estimate_topic_topic_k_prob_statistical(
      self, top_k_prob, observation_probs
  ):
    """A statistical test for estimate_topic_top_k_prob.

    Given top_k_prob, the probability that a given topic appears in a user's
    top-k topic set, the probability that any user reveals that topic to a
    website on any epoch is given by:

    Pr(topic is output) =
      Pr(topic is output | topic in top k set) * Pr(topic in top k set) +
      Pr(topic is output | topic not in top k set) * Pr(topic not in top k set)
    = q_in * top_k_prob + q_out * (1 - top_k_prob).

    Therefore, given top_k_prob, q_in, and q_out, we can sample the number of
    times a given topic is observed across num_users users and num_epcohs epochs
    by sampling a Binomial distribution with num_users*num_epochs trials and
    success rate equal to q_in * top_k_prob + q_out * (1 - top_k_prob).

    This statistical test ensures that when run with a sample from the above
    binomial distribution, the output of estimate_topic_top_k_prob is close
    to top_k_prob.

    Args:
      top_k_prob: The probability that the topic appears in a user's top-k set
        for one epoch.
      observation_probs: The values of q_in and q_out to use (note that these
        are derived parameters of the Topics API that depend on the number of
        topics and the random topic probability).
    """
    num_users = 100000
    num_epochs = 10
    success_rate = (
        observation_probs.in_top_k * top_k_prob
        + observation_probs.out_top_k * (1 - top_k_prob)
    )

    rng = tf.random.Generator.from_seed(0)
    topic_count_sample = rng.binomial(
        [], counts=float(num_users * num_epochs), probs=success_rate
    ).numpy()

    # This is the high probability bound for a single topic given by Lemma 10
    # from the paper. It holds with probability 99.9%. The number of topics does
    # not appear because we are not applying the union bound over all topics.
    error_bound = (
        1.0
        / (observation_probs.in_top_k - observation_probs.out_top_k)
        * math.sqrt(math.log(2/0.001) / 2 / num_users / num_epochs)
    )

    self.assertAlmostEqual(
        estimate_topic_top_k_prob(
            num_users=num_users,
            num_epochs=num_epochs,
            topic_count=topic_count_sample,
            observation_probs=observation_probs,
        ),
        top_k_prob,
        delta=error_bound,
    )

  def test_compute_attack_weights(self):
    weights_1 = compute_attack_weights(
        top_k=5,
        topic_top_k_prob=0.1,
        observation_probs=TopicObservationProbs(1 / 5, 0),
    )
    self.assertAlmostEqual(-math.log(1 / 5), weights_1.match)
    self.assertAlmostEqual(
        -math.log(1 / 5 * 4 * 0.1 / (5 - 0.1)), weights_1.mismatch
    )

    weights_2 = compute_attack_weights(
        top_k=10,
        topic_top_k_prob=0.1,
        observation_probs=TopicObservationProbs(1 / 10, 0),
    )
    self.assertAlmostEqual(-math.log(1 / 10), weights_2.match)
    self.assertAlmostEqual(
        -math.log(1 / 10 * 9 * 0.1 / (10 - 0.1)), weights_2.mismatch
    )

    weights_3 = compute_attack_weights(
        top_k=5,
        topic_top_k_prob=0.1,
        observation_probs=TopicObservationProbs(
            0.95 / 5 + 0.05 / 300, 0.05 / 300
        ),
    )
    self.assertAlmostEqual(
        -math.log(
            0.05 / 300
            + 0.95
            / 5
            * (0.95 / 5 + 0.05 / 300)
            * 0.1
            / (0.05 / 300 + 0.95 / 5 * 0.1)
        ),
        weights_3.match,
    )
    self.assertAlmostEqual(
        -math.log(0.05 / 300 + 0.95 / 5 * 4 * 0.1 / (5 - 0.1)),
        weights_3.mismatch,
    )

  def test_extract_website_one_topics(self):
    example = text_format.Parse(
        """
        features: {
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [0, 8]}}
            }
        }""",
        tf.train.Example(),
    )
    website_one_topics = extract_website_one_topics(example)
    self.assertEqual([0, 8], website_one_topics)

  def test_count_website_one_topics(self):
    examples = [
        text_format.Parse(
            """
        features: {
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [0, 8]}}
            }
        }""",
            tf.train.Example(),
        ),
        text_format.Parse(
            """
        features: {
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [8, 2]}}
            }
        }""",
            tf.train.Example(),
        ),
        text_format.Parse(
            """
        features: {
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [1, 2]}}
            }
        }""",
            tf.train.Example(),
        ),
    ]
    with TestPipeline() as p:
      examples_pcoll = p | beam.Create(examples)
      counts = examples_pcoll | CountWebsiteOneTopics()

      beam_test_util.assert_that(
          counts, beam_test_util.equal_to([(0, 1), (1, 1), (2, 2), (8, 2)])
      )

  def test_add_initial_counts(self):
    topic_counts = [(0, 1), (2, 5)]
    all_topics = [0, 1, 2]

    with TestPipeline() as p:
      topic_counts_pcoll = p | "Create topic_counts" >> beam.Create(
          topic_counts
      )
      all_topics_pcoll = p | "Create all_topics" >> beam.Create(all_topics)

      total_counts = AddInitialCountsInput(
          topic_counts=topic_counts_pcoll, all_topics=all_topics_pcoll
      ) | "Add initial counts" >> AddInitialCounts(5)

      beam_test_util.assert_that(
          total_counts, beam_test_util.equal_to([(0, 6), (1, 5), (2, 10)])
      )

  def test_add_initial_counts_crashes(self):
    topic_counts = [(0, 1)]
    all_topics = [0, 1]

    with self.assertRaises(ValueError):
      with TestPipeline() as p:
        topic_counts_pcoll = p | "Create topic_counts" >> beam.Create(
            topic_counts
        )
        all_topics_pcoll = p | "Create all_topics" >> beam.Create(all_topics)

        _ = AddInitialCountsInput(
            topic_counts=topic_counts_pcoll, all_topics=all_topics_pcoll
        ) | "Add initial counts" >> AddInitialCounts(0)

  def test_format_weights_as_json(self):
    topic = 12
    weights = AsymmetricHammingWeights(match=2.5, mismatch=0.5)
    json_str = format_weights_as_json_row((topic, weights))
    self.assertEqual(
        json.loads(json_str),
        {"token_id": 12, "match_weight": 2.5, "mismatch_weight": 0.5},
    )


if __name__ == "__main__":
  absltest.main()
