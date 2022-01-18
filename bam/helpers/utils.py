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

"""A collection of general utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle
import sys

import tensorflow.compat.v1 as tf


def load_json(path):
  with tf.gfile.GFile(path, 'r') as f:
    return json.load(f)


def write_json(o, path):
  tf.gfile.MakeDirs(path.rsplit('/', 1)[0])
  with tf.gfile.GFile(path, 'w') as f:
    json.dump(o, f)


def load_pickle(path):
  with tf.gfile.GFile(path, 'rb') as f:
    return pickle.load(f)


def write_pickle(o, path):
  tf.gfile.MakeDirs(path.rsplit('/', 1)[0])
  with tf.gfile.GFile(path, 'wb') as f:
    pickle.dump(o, f, -1)


def mkdir(path):
  if not tf.gfile.Exists(path):
    tf.gfile.MakeDirs(path)


def rmrf(path):
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)


def rmkdir(path):
  rmrf(path)
  mkdir(path)


def log(*args):
  msg = ' '.join(map(str, args))
  sys.stdout.write(msg + '\n')
  sys.stdout.flush()


def heading(*args):
  log(80 * '=')
  log(*args)
  log(80 * '=')


def nest_dict(d, prefixes, delim='_'):
  """Go from {prefix_key: value} to {prefix: {key: value}}."""
  nested = {}
  for k, v in d.items():
    for prefix in prefixes:
      if k.startswith(prefix + delim):
        if prefix not in nested:
          nested[prefix] = {}
        nested[prefix][k.split(delim, 1)[1]] = v
      else:
        nested[k] = v
  return nested


def flatten_dict(d, delim='_'):
  """Go from {prefix: {key: value}} to {prefix_key: value}."""
  flattened = {}
  for k, v in d.items():
    if isinstance(v, dict):
      for k2, v2 in v.items():
        flattened[k + delim + k2] = v2
    else:
      flattened[k] = v
  return flattened
