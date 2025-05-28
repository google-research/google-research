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

"""Tests for google3.third_party.google_research.google_research.re_identification_risk.reidentification."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
from apache_beam.testing.test_pipeline import TestPipeline
from google.protobuf import text_format
import tensorflow as tf

from re_identification_risk import reidentification

Example = tf.train.Example


def _get_list_of_test_examples():
  """Return a list of test examples."""
  return [
      text_format.Parse(
          """
      features {
        feature {
          key: "user_id",
          value: {bytes_list: {value: ["user_0"]}}
        }
        feature {
          key: "client_1_tokens",
          value: {int64_list: {value: [1, 2, 3]}}
        }
        feature {
          key: "client_2_tokens",
          value: {int64_list: {value: [1, 4, 3]}}
        }
      }""",
          Example(),
      ),
      text_format.Parse(
          """
      features {
        feature {
          key: "user_id",
          value: {bytes_list: {value: ["user_1"]}}
        }
        feature {
          key: "client_1_tokens",
          value: {int64_list: {value: [4, 3, 1]}}
        }
        feature {
          key: "client_2_tokens",
          value: {int64_list: {value: [1, 2, 3]}}
        }
      }""",
          Example(),
      ),
      text_format.Parse(
          """
      features {
        feature {
          key: "user_id",
          value: {bytes_list: {value: ["user_2"]}}
        }
        feature {
          key: "client_1_tokens",
          value: {int64_list: {value: [4, 5, 6]}}
        }
        feature {
          key: "client_2_tokens",
          value: {int64_list: {value: [4, 5, 3]}}
        }
      }""",
          Example(),
      ),
      text_format.Parse(
          """
      features {
        feature {
          key: "user_id",
          value: {bytes_list: {value: ["user_3"]}}
        }
        feature {
          key: "client_1_tokens",
          value: {int64_list: {value: [4, 6, 6]}}
        }
        feature {
          key: "client_2_tokens",
          value: {int64_list: {value: [1, 6, 6]}}
        }
      }""",
          Example(),
      ),
  ]


class TokensReidentificationTest(absltest.TestCase):

  def test_nearest_neighbor_search_without_weights(self):
    with TestPipeline() as pipeline:
      test_data = pipeline | 'CreateTestData' >> beam.Create(
          _get_list_of_test_examples()
      )
      empty_weights = pipeline | 'CreateEmptyWeights' >> beam.Create([])

      token_matches = reidentification.user_tokens_nearest_neighbor_search(
          test_data, test_data, empty_weights
      )

      expected = [
          (b'user_0', (2, b'user_0')),
          (b'user_1', (3, b'user_0')),
          (b'user_2', (2, b'user_2')),
          (b'user_3', (2, b'user_3')),
      ]

      util.assert_that(token_matches, util.equal_to(expected))

  def test_nearest_neighbor_search_with_weights(self):
    with TestPipeline() as pipeline:
      test_data = pipeline | 'CreateTestData' >> beam.Create(
          _get_list_of_test_examples()
      )
      token_weights = pipeline | 'CreateTokenWeights' >> beam.Create([
          (1, (-100.0, -1.0)),
          (2, (0.0, 1.0)),
          (3, (0.0, 1.0)),
          (4, (0.0, 1.0)),
          (5, (0.0, 1.0)),
          (6, (0.0, 1.0)),
      ])

      token_matches = reidentification.user_tokens_nearest_neighbor_search(
          test_data, test_data, token_weights
      )

      expected = [
          (b'user_0', (0, b'user_0')),
          (b'user_1', (1, b'user_0')),
          (b'user_2', (2, b'user_2')),
          (b'user_3', (-1, b'user_0')),
      ]

      util.assert_that(token_matches, util.equal_to(expected))


if __name__ == '__main__':
  absltest.main()
