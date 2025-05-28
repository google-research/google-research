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

"""Tests for models."""

from absl.testing import absltest
import immutabledict
# import lightgbm as lgb
import numpy as np
# import pandas as pd
from sklearn import ensemble

from covid_vhh_design import covid
from covid_vhh_design import models


PARENT_SEQ = covid.PARENT_SEQ
NUM_POS = len(PARENT_SEQ)
NUM_ALLOWED_POS = len(covid.ALLOWED_POS)

PARENT_SCORES = immutabledict.immutabledict({
    'SARS-CoV1_RBD': 5.217192470821265,
    'SARS-CoV2_RBD': 5.04978131798233,
    'SARS-CoV2_RBD_G502D': 4.778652645012139,
    'SARS-CoV2_RBD_N439K': 4.994606096279198,
    'SARS-CoV2_RBD_N501D': 4.724652525852702,
    'SARS-CoV2_RBD_N501F': 4.778145026905266,
    'SARS-CoV2_RBD_N501Y': 5.062748653051016,
    'SARS-CoV2_RBD_N501Y+K417N+E484K': 4.945958219881223,
    'SARS-CoV2_RBD_N501Y+K417T+E484K': 5.172440590969989,
    'SARS-CoV2_RBD_R408I': 4.5446856981352335,
    'SARS-CoV2_RBD_S477N': 4.935584968209451,
    'SARS-CoV2_RBD_V367F': 5.0165136977518605,
})


class MultiOutputClassifierTest(absltest.TestCase):

  def test_predict_proba(self):
    num_samples = 10
    num_outputs = 8
    x = np.random.rand(num_samples, 3)
    y = np.random.binomial(1, 0.5, (num_samples, num_outputs))
    model = models.MultiOutputClassifier(ensemble.RandomForestClassifier())
    model.fit(x, y)
    y_pred = model.predict_proba(x)
    self.assertEqual(y_pred.shape, (num_samples, num_outputs))
    self.assertGreaterEqual(y_pred.min(), 0)
    self.assertLessEqual(y_pred.max(), 1)


# class LGBMBoosterClassifierTest(absltest.TestCase):

#   def test_predict_proba(self):
#     num_samples = 10
#     x = np.random.rand(num_samples, 3)
#     y = np.random.binomial(1, 0.5, num_samples)
#     booster = lgb.LGBMClassifier()
#     booster.fit(x, y)
#     wrapped_booster = models.LGBMBoosterClassifier(booster)
#     y_pred = wrapped_booster.predict_proba(x)
#     self.assertEqual(y_pred.shape, (num_samples, 2))
#     self.assertGreaterEqual(y_pred.min(), 0)
#     self.assertLessEqual(y_pred.max(), 1)


# class SequenceEncoderTest(absltest.TestCase):

#   def setUp(self):
#     super().setUp()
#     self._encoder = models.SequenceEncoder()

#   def test_encode_token(self):
#     expected_onehot_features = np.zeros(20)
#     expected_onehot_features[1] = 1
#     expected_aaindex_features = [
#         0.576,
#         0.8463,
#         -0.7165,
#         -2.1895,
#         2.1857,
#         -1.3886,
#         -0.7552,
#         -1.339,
#         -0.5288,
#         1.1555,
#     ]

#     actual_features = self._encoder.encode_token('C')
#     self.assertLen(actual_features, 30)
#     np.testing.assert_equal(actual_features[:20], expected_onehot_features)
#     np.testing.assert_almost_equal(
#         actual_features[20:], expected_aaindex_features, decimal=2
#     )

#   def test_encode_sequence(self):
#     features = self._encoder.encode_sequence(PARENT_SEQ)
#     self.assertSequenceEqual(features.shape, (NUM_ALLOWED_POS, 30))
#     self.assertTrue(np.all(features[:, :20].sum(axis=1) == 1))

#   def test_encode_sequences(self):
#     features = self._encoder.encode_sequences(['A' * NUM_POS, 'C' * NUM_POS])
#     self.assertSequenceEqual(features.shape, (2, NUM_ALLOWED_POS * 30))
#     self.assertTrue(
#         np.all(
#             features[:, : NUM_ALLOWED_POS * 20].sum(axis=1) == NUM_ALLOWED_POS
#         )
#     )
#     self.assertTrue(
#         np.all(features[0, np.arange(0, NUM_ALLOWED_POS * 20, 20)] == 1)
#     )
#     self.assertTrue(
#         np.all(features[1, np.arange(1, NUM_ALLOWED_POS * 20, 20)] == 1)
#     )


# class ModelsTest(absltest.TestCase):

#   def test_end_to_end(self):
#     expected_scores = pd.DataFrame(
#         [PARENT_SCORES], index=pd.Index(['parent'], name='label')
#     )
#     model = models.CombinedModel.load()
#     encoder = models.SequenceEncoder()
#     actual_scores = models.score_labeled_sequences(
#         model, encoder, dict(parent=PARENT_SEQ)
#     )
#     pd.testing.assert_frame_equal(actual_scores.sort_index(1),expected_scores)


if __name__ == '__main__':
  absltest.main()
