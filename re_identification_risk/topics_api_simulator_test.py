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
from google.protobuf import text_format
import numpy as np
import tensorflow as tf

from re_identification_risk.topics_api_simulator import build_topics_simulator_output
from re_identification_risk.topics_api_simulator import extract_top_k_topics
from re_identification_risk.topics_api_simulator import extract_topic_profiles


class TopicsApiSimulatorTest(absltest.TestCase):

  def test_extract_top_k_topics_no_tie_breaking(self):
    rng = np.random.default_rng(0)
    profile = {
        1: 0.2,
        2: 0.8,
        3: 0.1,
        4: 0.9
    }
    all_topics = np.arange(10)

    top_1 = extract_top_k_topics(rng, profile, 1, all_topics)
    top_2 = extract_top_k_topics(rng, profile, 2, all_topics)
    top_3 = extract_top_k_topics(rng, profile, 3, all_topics)
    top_4 = extract_top_k_topics(rng, profile, 4, all_topics)

    self.assertSameElements([4], top_1)
    self.assertSameElements([4, 2], top_2)
    self.assertSameElements([4, 2, 1], top_3)
    self.assertSameElements([4, 2, 1, 3], top_4)

  def test_extract_top_k_padding(self):
    rng = np.random.default_rng(0)
    profile = {1: 1.0, 2: 1.0}
    all_topics = [1, 2, 20, 30, 40, 50]

    top_5 = extract_top_k_topics(rng, profile, 5, all_topics)
    top_6 = extract_top_k_topics(rng, profile, 6, all_topics)

    self.assertContainsSubset(top_5, all_topics)
    self.assertLen(top_5, 5)

    self.assertContainsSubset(top_6, all_topics)
    self.assertLen(top_6, 6)

  def test_extract_topic_profiles(self):
    example = text_format.Parse(
        """
      features {
        feature {
          key: "user_id",
          value: {bytes_list: {value: ["user_1"]}}
        }
        feature {
          key: "epoch_0_topics",
          value: {int64_list: {value: [0, 1, 5]}}
        }
        feature {
          key: "epoch_0_weights",
          value: {float_list: {value: [1, 2, 10]}}
        }
        feature {
          key: "epoch_1_topics",
          value: {int64_list: {value: [0, 2]}}
        }
        feature {
          key: "epoch_1_weights",
          value: {float_list: {value: [1, 2]}}
        }
      }""", tf.train.Example())
    topic_profiles = extract_topic_profiles(example, num_epochs=2)

    self.assertDictEqual(topic_profiles[0], {0: 1, 1: 2, 5: 10})
    self.assertDictEqual(topic_profiles[1], {0: 1, 2: 2})

  def test_build_topics_simulator_output(self):
    user_id = b"user_1"
    epoch_top_topics = [np.array([0, 1, 2, 3, 4]), np.array([2, 1, 3, 8, 100])]
    website_1_topics = np.array([0, 8])
    website_2_topics = np.array([2, 100])

    example = build_topics_simulator_output(user_id, epoch_top_topics,
                                            website_1_topics, website_2_topics)

    expected_example = text_format.Parse(
        """
        features: {
            feature: {
                key: "user_id",
                value: {bytes_list: {value: ["user_1"]}}
            }
            feature: {
                key: "epoch_0_tokens"
                value: {int64_list: {value: [0, 1, 2, 3, 4]}}
            }
            feature: {
                key: "epoch_1_tokens"
                value: {int64_list: {value: [2, 1, 3, 8, 100]}}
            }
            feature: {
                key: "client_1_tokens"
                value: {int64_list: {value: [0, 8]}}
            }
            feature: {
                key: "client_2_tokens"
                value: {int64_list: {value: [2, 100]}}
            }
        }""", tf.train.Example())

    self.assertEqual(example, expected_example)

if __name__ == "__main__":
  absltest.main()
