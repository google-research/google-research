# coding=utf-8
# Copyright 2023 The Google Research Authors.
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


class AsymmetricAttackTrainerTest(absltest.TestCase):

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

  def test_estimate_topic_top_k_prob(self):
    # When we observe a topic in 10% of all (user, epoch) pairs and topics are
    # prob_random_choice = 0, then the estimated top_k probability is 0.1.
    self.assertAlmostEqual(
        0.1,
        estimate_topic_top_k_prob(
            num_users=100,
            num_epochs=1,
            topic_count=10,
            observation_probs=TopicObservationProbs(1 / 5, 0),
        ),
    )

    # When we observe a topic in 50% of all (user, epoch) pairs and topics are
    # prob_random_choice = 0, then the estimated top_k probability is 0.5.
    self.assertAlmostEqual(
        0.5,
        estimate_topic_top_k_prob(
            num_users=100,
            num_epochs=1,
            topic_count=50,
            observation_probs=TopicObservationProbs(1 / 5, 0),
        ),
    )

    self.assertAlmostEqual(
        0.1 * (0.95 / 5 + 0.05 / 300) / (0.95 / 5),
        estimate_topic_top_k_prob(
            num_users=100,
            num_epochs=1,
            topic_count=10,
            observation_probs=TopicObservationProbs(
                0.95 / 5 + 0.05 / 300, 0.05 / 300
            ),
        ),
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
