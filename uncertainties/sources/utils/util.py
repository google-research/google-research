# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os

from absl import flags
import numpy as np
import tensorflow as tf

import gin.tf

FLAGS = flags.FLAGS


def clear_folder(full_path):
  """Clear the folder."""
  if tf.gfile.Exists(full_path):
    tf.gfile.DeleteRecursively(full_path)
  tf.gfile.MakeDirs(full_path)


def cummean(arr, axis):
  """Returns the cumulative mean of array along the axis.

  Args:
    arr: numpy array
    axis: axis over which to compute the cumulative mean.
  """
  n = arr.shape[axis]
  res = np.cumsum(arr, axis=axis)
  res = np.apply_along_axis(lambda x: np.divide(x, np.arange(1, n+1)),
                            axis=axis, arr=res)
  return res


def truncate_data(perc, dataset):
  """Truncate the training dataset.

  Args:
    perc: float between 0 and 1, percentage of data kept.
    dataset: data, under the form (x_train, y_train), (x_test, y_test)
  Returns:
    dataset: truncated training dataset, full test dataset
  """
  (x_train, y_train), (x_test, y_test) = dataset
  n = x_train.shape[0]
  n_trunc = int(perc * n)
  return (x_train[:n_trunc, :], y_train[:n_trunc, :]), (x_test, y_test)


def save_gin(config_path):
  """Safely saves a gin config to a file.

  Args:
    config_path: String with path where to save the gin config.
  """
  # Ensure that the folder exists.
  directory = os.path.dirname(config_path)
  if not tf.gfile.IsDirectory(directory):
    tf.gfile.MakeDirs(directory)
  # Save the actual config.
  with tf.gfile.GFile(config_path, "w") as f:
    f.write(gin.operative_config_str())


def write_gin(output_dir):
  """Write result dictionary, gin config and other kw arguments to CSV file."""
  # Obtain the ordered gin dictionary.
  output_dict = get_gin_dict()
  # Write result to disk.
  result_path = os.path.join(output_dir, "gin_config.csv")
  with tf.gfile.GFile(result_path, "w") as f:
    writer = csv.DictWriter(f, fieldnames=output_dict.keys())
    writer.writeheader()
    writer.writerow(output_dict)


def get_gin_dict():
  """Returns ordered dictionary with specific gin configs."""
  result = collections.OrderedDict()
  # Gin does not allow to retrieve such a dictionary but it allows to obtain a
  # string with all active configs in human readable format.
  for line in gin.operative_config_str().split("\n"):
    # We need to filter out the auto-generated comments and make sure the line
    # contains a valid assignment.
    if not line.startswith("#") and not line.endswith("\\") and " = " in line:
      # We are content with the string representations (we only want to use it
      # for analysis in colab after all).
      key, value = line.split(" = ", 2)
      _, key_2 = key.split(".", 2)
      result["%s" % key_2] = value
  return result
