# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Small models to be finetuned on embeddings."""

from sklearn import discriminant_analysis
from sklearn import ensemble
from sklearn import linear_model

LogisticRegression = linear_model.LogisticRegression
LinearDiscriminantAnalysis = discriminant_analysis.LinearDiscriminantAnalysis
RandomForestClassifier = ensemble.RandomForestClassifier


# pylint:disable=line-too-long,unused-variable, g-no-space-after-comment, g-long-lambda
def get_sklearn_models():
  return {
      'LogisticRegression':
          lambda: LogisticRegression(
              C=1e5, solver='lbfgs', multi_class='multinomial'),
      'LogisticRegression_balanced':
          lambda: LogisticRegression(
              C=1e5,
              solver='lbfgs',
              multi_class='multinomial',
              class_weight='balanced'),
      'LDA_LSQR_AUTO':
          lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
  }
# pylint:enable=line-too-long,unused-variable, g-no-space-after-comment, g-long-lambda
