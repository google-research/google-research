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

# Lint as: python3
"""Utilities for interacting with filesystem."""

import glob
import os
from absl import logging


def Glob(glob_pattern):
  return glob.glob(glob_pattern)  # pylint: disable=unreachable


def MaybeMakeDirs(output_dir):
  if not os.path.exists(output_dir):  # pylint: disable=unreachable
    logging.info('Creating output directory: %s', output_dir)
    os.makedirs(output_dir)


def Open(filename, mode):
  return open(filename, mode)  # pylint: disable=unreachable
