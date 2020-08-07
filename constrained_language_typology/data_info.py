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

"""Utility functions for working with various data mapping dictionaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip
import json
import os

from absl import logging

import constants as const

FILE_EXTENSION = ".json.gz"


def write_data_info(filename, data):
  """Saves data info mappings to compressed JSON file."""
  logging.info("Writing data info to \"%s\" ...", filename)
  with gzip.open(filename, "wb") as f:
    f.write(json.dumps(data, ensure_ascii=False).encode(const.ENCODING))


def load_data_info(filename):
  """Loads data info mappings from a compressed JSON file."""
  logging.info("Loading data info from \"%s\" ...", filename)
  with gzip.open(filename, "rb") as f:
    contents = json.loads(f.read().decode(const.ENCODING))
  return contents


def data_info_path_for_testing(data_dir):
  """Returns data info path we use at test time."""
  return os.path.join(data_dir,
                      (const.DATA_INFO_FILENAME + "_" +
                       const.TRAIN_DEV_TEST_FILENAME +
                       FILE_EXTENSION))
