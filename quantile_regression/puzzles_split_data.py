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

# pylint: skip-file
"""Splits out a test set containing the 876 most recent examples from the full

Hoefnagel Hold Times puzzles dataset. Takes as input the full HoefnagelHoldTimes
dataset, and outputs a combined train/validation set and a test set. The
combined train/validation set can separately be further split IID.
"""

from absl import app
from absl import flags
import numpy as np
import pandas as pd

FLAGS = flags.FLAGS

# Dataset flags
flags.DEFINE_string(
    'full_puzzles_data_file',
    'HoefnagelHoldTimes_Jan29_2021.csv',
    'Filename of full Hoefnagel Hold Times dataset.',
)
flags.DEFINE_string(
    'output_trainval_filename',
    '',
    'Filename of the output combined train/validation set.',
)
flags.DEFINE_string(
    'output_test_filename', '', 'Filename of the output test dataset.'
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with open(FLAGS.full_puzzles_data_file, 'r') as f:
    rows = f.readlines()
  max_columns = max(len(row.split(',')) for row in rows)

  holdTimesDF = pd.read_csv(
      FLAGS.full_puzzles_data_file, header=None, names=list(range(max_columns))
  )

  holdTimesDF = holdTimesDF.replace(to_replace=np.nan, value=-1)

  holdTimes = holdTimesDF.to_numpy()

  testSet_1Features = []
  trainSet_1Features = []

  # Iterate through each row
  for row in holdTimes:
    for j in range(row.shape[0]):
      if row[j] == -1:
        labelIndex = j - 1
        break
    else:
      labelIndex = row.shape[0] - 1
    label = row[labelIndex]
    feature1 = -1
    if labelIndex > 0:
      feature1 = row[labelIndex - 1]

    # This gives us the most recent feature and label
    # store as a test examples

    # Let's get a group label
    group = -1
    past = row[0 : (labelIndex - 1)]
    if len(past) > 1:
      if past.mean() <= 4:  # Active user
        group = 1
      if past.max() >= 40:  # High-risk user, supercedes
        group = 2
      if labelIndex == 0:  # New user, supercedes
        group = 3

    testSet_1Features.append(np.array([feature1, label, group]))

    # Next slide time window back making train samples
    for r in range(labelIndex):
      label = row[r]
      feature1 = -1
      if r > 0:
        feature1 = row[r - 1]

      group = -1
      past = row[0 : (r - 1)]
      if len(past) > 1:
        if past.mean() <= 4:  # Active user
          group = 1
        if past.max() >= 40:  # High-risk user, supercedes
          group = 2
        if labelIndex == 0:  # New user, supercedes
          group = 3

      trainSet_1Features.append(np.array([feature1, label, group]))

  # Make training data where there are 5 features from 5 past events
  testSet_5Features = []
  trainSet_5Features = []

  # Iterate through each row
  for row in holdTimes:
    for j in range(row.shape[0]):
      if row[j] == -1:
        labelIndex = j - 1
        break
    else:
      labelIndex = row.shape[0] - 1

    # At this point, labelIndex is the index of the most recent holdTime for
    # this row
    label = row[labelIndex]
    feature1 = -1
    feature2 = -1
    feature3 = -1
    feature4 = -1
    feature5 = -1

    if labelIndex > 4:
      feature5 = row[labelIndex - 1]
      feature4 = row[labelIndex - 2]
      feature3 = row[labelIndex - 3]
      feature2 = row[labelIndex - 4]
      feature1 = row[labelIndex - 5]

    if labelIndex == 4:
      feature5 = row[labelIndex - 1]
      feature4 = row[labelIndex - 2]
      feature3 = row[labelIndex - 3]
      feature2 = row[labelIndex - 4]

    if labelIndex == 3:
      feature5 = row[labelIndex - 1]
      feature4 = row[labelIndex - 2]
      feature3 = row[labelIndex - 3]

    if labelIndex == 2:
      feature5 = row[labelIndex - 1]
      feature4 = row[labelIndex - 2]

    if labelIndex == 1:
      feature5 = row[labelIndex - 1]

    group = -1
    past = row[0 : (labelIndex - 1)]
    if len(past) > 1:
      if past.mean() <= 4:  # Active user
        group = 1
      if past.max() >= 40:  # High-risk user, supercedes
        group = 2
      if labelIndex == 0:  # New user, supercedes
        group = 3

    testSet_5Features.append(
        np.array(
            [feature1, feature2, feature3, feature4, feature5, label, group]
        )
    )

    # Next slide time window back making train samples
    for r in range(labelIndex):
      label = row[r]
      feature5 = -1
      feature4 = -1
      feature3 = -1
      feature2 = -1
      feature1 = -1

      if r > 4:
        feature5 = row[r - 1]
        feature4 = row[r - 2]
        feature3 = row[r - 3]
        feature2 = row[r - 4]
        feature1 = row[r - 5]

      if r == 4:
        feature5 = row[r - 1]
        feature4 = row[r - 2]
        feature3 = row[r - 3]
        feature2 = row[r - 4]

      if r == 3:
        feature5 = row[r - 1]
        feature4 = row[r - 2]
        feature3 = row[r - 3]

      if r == 2:
        feature5 = row[r - 1]
        feature4 = row[r - 2]

      if r == 1:
        feature5 = row[r - 1]

      group = -1
      past = row[0 : (r - 1)]
      if len(past) > 1:
        if past.mean() <= 4:  # Active user
          group = 1
        if past.max() >= 40:  # High-risk user, supercedes
          group = 2
        if labelIndex == 0:  # New user, supercedes
          group = 3

      trainSet_5Features.append(
          np.array(
              [feature1, feature2, feature3, feature4, feature5, label, group]
          )
      )

  colnames = ['f1', 'f2', 'f3', 'f4', 'f5', 'label', 'group']

  testSet_5Features_df = pd.DataFrame(testSet_5Features, columns=colnames)
  testSet_5Features_df.to_csv(FLAGS.output_test_filename)

  trainvalSet_5Features_df = pd.DataFrame(trainSet_5Features, columns=colnames)
  trainvalSet_5Features_df.to_csv(FLAGS.output_trainval_filename)

  return 0


if __name__ == '__main__':
  app.run(main)
