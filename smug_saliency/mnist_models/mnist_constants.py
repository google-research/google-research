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

"""MNIST Constants.

These constants, specific to the MNIST dataset, are used across multiple places
in this project.
"""


NUM_OUTPUTS = 10
# Using 80% of the train data for training and 20% for validation
TRAIN_DATA_PERCENT = 80
TRAIN_VAL_SPLIT = (4, 1)
NUM_TRAIN_EXAMPLES = 48000
IMAGE_EDGE_LENGTH = 28
NUM_FLATTEN_FEATURES = IMAGE_EDGE_LENGTH * IMAGE_EDGE_LENGTH
