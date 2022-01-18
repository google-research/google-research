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

"""Configurations for all word2act data generation global configs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Android Emulator Config
ANDROID_LOG_MAX_ABS_X = 32676
ANDROID_LOG_MAX_ABS_Y = 32676
SCREEN_WIDTH = 540
SCREEN_HEIGHT = 960
SCREEN_CHANNEL = 4
# Rico dataset screen config
RICO_SCREEN_WIDTH = 1440
RICO_SCREEN_HEIGHT = 2560

# Data Generation Config
LABEL_DEFAULT_VALUE_INT = 0
LABEL_DEFAULT_VALUE_STRING = ''
LABEL_DEFAULT_INVALID_INT = -1
LABEL_DEFAULT_INVALID_STRING = ''

FEATURE_ANCHOR_PADDING_INT = -1
FEATURE_DEFAULT_PADDING_INT = 0
FEATURE_DEFAULT_PADDING_FLOAT = -0.0
FEATURE_DEFAULT_PADDING_STR = ''
TOKEN_DEFAULT_PADDING_INT = 0

MAX_WORD_NUM_UPPER_BOUND = 30
MAX_WORD_LENGTH_UPPER_BOUND = 50
SHARD_NUM = 10
MAX_INPUT_WORD_NUMBER = 5

# synthetic action config
MAX_OBJ_NAME_WORD_NUM = 3
MAX_WIN_OBJ_NAME_WORD_NUM = 10
MAX_INPUT_STR_LENGTH = 10
NORM_VERTICAL_NEIGHBOR_MARGIN = 0.01
NORM_HORIZONTAL_NEIGHBOR_MARGIN = 0.01
INPUT_ACTION_UPSAMPLE_RATIO = 1

# Windows data dimension Config. The numbers are set based on real data
# dimension distribution.
MAX_UI_OBJ_WORD_NUM_UPPER_BOUND = 20
MAX_UI_OBJ_WORD_LENGTH_UPPER_BOUND = 21

# view hierarchy config
MAX_PER_OBJECT_WORD_NUM = 10
MAX_WORD_LENGTH = 100
TRAINING_BATCH_SIZE = 2
UI_OBJECT_TYPE_NUM = 15
ADJACENT_BOUNDING_BOX_THRESHOLD = 3
