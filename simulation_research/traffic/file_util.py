# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utility functions for the traffic simulation project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import logging
from six.moves import cPickle

f_open = open
f_exists = os.path.exists
f_mkdir = os.mkdir
f_makedirs = os.makedirs
f_abspath = os.path.abspath
FileIOError = IOError



def append_line_to_file(file_path, line):
  """Append the line to the end of the file."""
  if not f_exists(file_path):
    logging.info('The file does not exist, writing to %s', file_path)
  with f_open(file_path, 'a') as f:
    f.write(line + '\n')


def save_variable(file_path, variable):
  """Save variables to file."""
  # defaultdict can not be saved using cPickle, but dict can.
  if isinstance(variable, collections.defaultdict):
    variable = dict(variable)
  with f_open(file_path, 'wb') as f:
    # Automatically gives you the highest protocol for your Python version.
    cPickle.dump(variable, f, cPickle.HIGHEST_PROTOCOL)
    logging.info('Save file to %s', file_path)


def load_variable(file_path):
  """Load variable from file."""
  with f_open(file_path, 'rb') as f:
    logging.info('Load file from %s', file_path)
    return cPickle.load(f)
