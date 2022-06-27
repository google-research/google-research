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

"""Near-Optimal Debiasing Algorithms for ML."""

from absl import app
import numpy as np
import sklearn.ensemble

from ml_debiaser import randomized_threshold
from ml_debiaser import reduce_to_binary


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # We begin with the post-processing rule for binary classifiers.
  num_examples_train = 10000
  num_examples_val = 1000

  num_features = 100
  num_classes = 2
  num_groups = 10

  # Generate synthetic data.
  x_train = np.random.randn(num_examples_train, num_features)
  y_train = np.random.randint(0, num_classes, size=num_examples_train)
  s_train = np.random.randint(0, num_groups, size=num_examples_train)

  x_val = np.random.randn(num_examples_val, num_features)
  s_val = np.random.randint(0, num_groups, size=num_examples_val)

  # Train a classifier.
  clf = sklearn.ensemble.RandomForestClassifier(max_depth=10)
  clf.fit(x_train, y_train)

  # Debias the classifier.
  # Note that the debiaser should be trained on a fresh sample.
  y_pred = clf.predict(x_val)
  y_pred = 2 * y_pred - 1  # rescaling to [-1, +1]
  rto = randomized_threshold.RandomizedThreshold(gamma=0.05, eps=0)
  rto.fit(y_pred, s_val, sgd_steps=10_000, full_gradient_epochs=100)

  # Test if debiasing is successful.
  ydeb_val = rto.predict(y_pred, s_val)

  mean_scores_before = [np.mean(y_pred[s_val == k]) for k in range(num_groups)]
  mean_scores_after = [np.mean(ydeb_val[s_val == k]) for k in range(num_groups)]

  print('DP before: ', max(mean_scores_before) - min(mean_scores_before))
  print('DP after: ', max(mean_scores_after) - min(mean_scores_after))

  # Next, we debias multiclass datasets.
  num_examples_train = 1000
  num_features = 100
  num_classes = 5
  num_groups = 10

  x_train = np.random.randn(num_examples_train, num_features)
  s_train = np.random.randint(0, num_groups, size=num_examples_train)

  y_train_1d = np.random.randint(0, num_classes, size=num_examples_train)
  # Convert to one_hot encoding.
  y_train = np.zeros((num_examples_train, num_classes))
  y_train[np.arange(num_examples_train), y_train_1d] = 1.0

  # Apply the debiaser.
  r2b = reduce_to_binary.Reduce2Binary(num_classes=num_classes)
  ydeb_train = r2b.fit(y_train, s_train, sgd_steps=10_000,
                       full_gradient_epochs=500, max_admm_iter=5)
  print(np.mean(ydeb_train[s_train == 0], axis=0) -
        np.mean(ydeb_train[s_train == 1], axis=0))


if __name__ == '__main__':
  app.run(main)
