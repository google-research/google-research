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

"""Generic Utilities for Transfer Experiments."""

import os
from typing import Any, Dict, Text

import ml_collections
from six.moves import cPickle as pickle  # for performance
import tensorflow as tf


def save_dict(d, filename):
  """Save dict to pickle file."""
  filepath = os.path.dirname(filename)
  if not tf.io.gfile.isdir(filepath):
    tf.io.gfile.makedirs(filepath)

  with tf.io.gfile.GFile(filename, 'wb') as f:
    pickle.dump(d, f)


def load_dict(filename):
  """Load dict to return dictionary."""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    return pickle.load(f)


def config_dict_flatten(config_dict,
                        prefix = ''):
  """Returns a flattened config dictionary as dict."""
  flattened_items = []

  if prefix:
    prefix += '.'

  for k, v in config_dict.items():
    k_flat = f'{prefix}{k}'

    if isinstance(v, dict) or isinstance(v, ml_collections.ConfigDict):
      flattened_items.extend(config_dict_flatten(v, prefix=k_flat).items())
    else:
      flattened_items.append((k_flat, v))

  return dict(flattened_items)

