# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Test utils for ROUGE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

_TESTDATA_PREFIX = os.path.join(os.path.dirname(__file__), "testdata")

TARGETS_FILE = os.path.join(_TESTDATA_PREFIX, "target.txt")

PREDICTIONS_FILE = os.path.join(_TESTDATA_PREFIX, "prediction.txt")

LARGE_TARGETS_FILE = os.path.join(_TESTDATA_PREFIX, "target_large.txt")

LARGE_PREDICTIONS_FILE = os.path.join(_TESTDATA_PREFIX, "prediction_large.txt")

DELIMITED_FILE = os.path.join(_TESTDATA_PREFIX, "delimited.txt")
