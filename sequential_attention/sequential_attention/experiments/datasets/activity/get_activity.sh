#!/bin/bash
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


DATA_DIR='.'

# https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
ACTIVITY_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
ACTIVITY_FILE=activity_dataset
curl "${ACTIVITY_URL}" --output "${ACTIVITY_FILE}.zip"
unzip "${ACTIVITY_FILE}.zip"

cp "UCI HAR Dataset/train/X_train.txt" "${DATA_DIR}"
cp "UCI HAR Dataset/train/y_train.txt" "${DATA_DIR}"
cp "UCI HAR Dataset/test/X_test.txt" "${DATA_DIR}"
cp "UCI HAR Dataset/test/y_test.txt" "${DATA_DIR}"