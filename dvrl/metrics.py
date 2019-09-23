# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Performance metrics.

1. Noise label discovery rate (from high/low to low/high data values)
2. Prediction performance after removing high/low value samples
"""

# Necessary packages and function call
import lightgbm
import numpy as np

# Sklearn packages
from sklearn.metrics import accuracy_score
from tqdm import tqdm


#%% Noise label discovery
def noise_label_discovery(dve_out, noise_idx):
  """Noise label discovery rate metrics.

  Args:
    dve_out: data values
    noise_idx: noise index

  Returns:
    output_perf: noise label discovery rate (per 5 percentiles)
  """
  # Sorting
  divide = 20  # Per 100/20 percentile
  sort_idx = np.argsort(dve_out)
  neg_sort_idx = np.argsort(-dve_out)

  # Output initialization
  output_perf = np.zeros([divide, 2])

  # For each percentile
  for itt in range(divide):
    # from low to high data values
    output_perf[itt, 0] = len(np.intersect1d(sort_idx[:int((itt+1)* \
                              len(dve_out)/divide)], noise_idx)) \
                              / len(noise_idx)
    # from high to low data values
    output_perf[itt, 1] = len(np.intersect1d(neg_sort_idx[:int((itt+1)* \
                              len(dve_out)/divide)], noise_idx)) \
                              / len(noise_idx)
  # Returns TPR of discovered noisy samples
  return output_perf


#%% Remove most/least valuable samples
def remove_high_low(dve_out, x_train, y_train, x_test, y_test):
  """Performances after removing most / least valuable samples.

  Args:
    dve_out: data values
    x_train: training features
    y_train: training labels
    x_test: testing features
    y_test: testing labels

  Returns:
    output_perf: performances
  """

  # Sorting
  divide = 20
  sort_idx = np.argsort(dve_out)
  neg_sort_idx = np.argsort(-dve_out)

  temp_output = np.zeros([divide, 2])

  # For each division
  for itt in tqdm(range(divide)):

    # 1. Least
    new_train_x = x_train[sort_idx[int(itt*len(x_train[:, 0])/divide):], :]
    new_train_y = y_train[sort_idx[int(itt*len(x_train[:, 0])/divide):]]

    model = lightgbm.LGBMClassifier()

    if len(np.unique(new_train_y)) > 1:
      model.fit(new_train_x, new_train_y)
      test_y_hat = model.predict_proba(x_test)

      temp_output[itt, 0] = accuracy_score(y_test, np.argmax(test_y_hat,
                                                             axis=1))

    # 2. Most
    new_train_x = x_train[neg_sort_idx[int(itt*len(x_train[:, 0])/divide):], :]
    new_train_y = y_train[neg_sort_idx[int(itt*len(x_train[:, 0])/divide):]]

    model = lightgbm.LGBMClassifier()

    if len(np.unique(new_train_y)) > 1:
      model.fit(new_train_x, new_train_y)
      test_y_hat = model.predict_proba(x_test)

      temp_output[itt, 1] = accuracy_score(y_test, np.argmax(test_y_hat,
                                                             axis=1))

  return temp_output
