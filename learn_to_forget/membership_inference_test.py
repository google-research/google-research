# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for membership_inference module.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection

from learn_to_forget import membership_inference


class MembershipInferenceTest(parameterized.TestCase):

  n_samples = 500
  n_features = 500

  def test_evaluate_attack_model(self):
    data, target = datasets.make_classification(
        n_samples=self.n_samples,
        n_features=self.n_features,
        n_informative=self.n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0,
        random_state=0)
    cv = model_selection.StratifiedShuffleSplit(test_size=0.5)
    idx_train, _ = next(cv.split(data, target))
    clf = linear_model.LogisticRegression(
        C=1e3, max_iter=2000).fit(data[idx_train], target[idx_train])
    # check that the classifier has a high train set accuracy
    self.assertGreater(clf.score(data[idx_train], target[idx_train]), 0.99)

    members = np.zeros_like(target)
    members[idx_train] = 1

    # the logistic loss on the single data iterms, minus regularization
    loss_data = clf.predict_log_proba(data)[:, target]
    # it can be - inf, so we clip the data
    loss_data = np.clip(loss_data, -100, 100)
    cv_scores = membership_inference.evaluate_attack_model(loss_data, members)
    self.assertGreater(np.mean(cv_scores), 0.8)


if __name__ == '__main__':
  absltest.main()
