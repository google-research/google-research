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

"""Utility functions for logging during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import pickle
import numpy as np
import tensorflow.compat.v1 as tf


def custom_clip(vec, low, high):
  new_vec = []
  for i in range(len(vec)):
    new_val = min(vec[i], high)
    new_val = max(new_val, low)
    new_vec.append(new_val)
  return np.array(new_vec)


def log_row(csv_file, row):
  with tf.gfile.Open(csv_file, 'ab') as csvfile:
    cw = csv.writer(
        csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    cw.writerow(row)


class NumpyEncoder(json.JSONEncoder):
  """Special json encoder for numpy types."""

  def default(self, obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
      return float(obj)
    elif isinstance(obj, (np.ndarray,)):  #### This is the fix
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


class AlgorithmState(object):
  """Saves the algorithm state into a pickle.

  Particularly useful for resuming real robot experiments.
  """

  def __init__(self):
    self.fresh = True
    self.meta_eval_passed = False
    self.single_values = []
    self.query_index = 0
    self.temp_perturbations = []

  def load(self, load_dir):
    attr_dict = pickle.load(tf.gfile.Open(load_dir, 'r'))
    for k, v in attr_dict.items():
      setattr(self, k, v)
    self.fresh = False

  def save(self, save_dir):
    pickle.dump(self.__dict__, tf.gfile.GFile(save_dir, 'w'))
