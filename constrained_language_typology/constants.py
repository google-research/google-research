# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Common constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Use this encoding for all the I/O.
ENCODING = "utf-8"

# Training data.
TRAIN_FILENAME = "train"

# Development data.
DEV_FILENAME = "dev"

# Test data.
TEST_BLIND_FILENAME = "test_blinded"

# Key for the combined train/dev data.
TRAIN_DEV_FILENAME = "train_dev"

# Key for the combined train/dev/test data.
TRAIN_DEV_TEST_FILENAME = "train_dev_test"

# Data info section key for languages.
DATA_KEY_LANGUAGES = "languages"

# Data info section key for features.
DATA_KEY_FEATURES = "features"

# Data info section key for values.
DATA_KEY_VALUES = "values"

# Data info section key for genus.
DATA_KEY_GENERA = "genera"

# Data info section key for family.
DATA_KEY_FAMILIES = "families"

# Key for the features for prediction per language.
DATA_KEY_FEATURES_TO_PREDICT = "features_to_predict"

# Data info name for the JSON file.
DATA_INFO_FILENAME = "data_info"

# Filename for the test features to predict.
FEATURES_TO_PREDICT_FILENAME = "test_features_to_predict"

# Unknown feature value in the test data.
UNKNOWN_FEATURE_VALUE = "?"
