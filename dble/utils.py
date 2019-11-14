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

"""Contains util functions for the training of DBLE.
"""
import json
import logging
import os
import numpy as np
import tensorflow as tf


def find_variables(param_dict):
  """Finds items in dictionary that are lists and treat them as variables."""
  variables = []
  for key, val in param_dict.items():
    if isinstance(val, list):
      variables.append(key)
  return variables


def variable_report(report_non_trainable=True):
  """Prints the shapes of all trainable variables."""
  total_params = 0
  lines = ['Trainable Variables Report', '--------------------------']

  trainable_variables = tf.trainable_variables()

  for var in trainable_variables:
    shape = var.get_shape().as_list()
    num_params = np.prod(shape)
    total_params += num_params
    lines.append('shape: %15s, %5d, %s, %s' %
                 (shape, num_params, var.name, var.dtype))
  lines.append('Total number of trainable parameters: %d' % total_params)

  if report_non_trainable:
    lines.extend(['', 'Non-Trainable Variables', '---------------------'])
    for var in tf.global_variables():
      if var in trainable_variables:
        continue
      shape = var.get_shape().as_list()
      num_params = np.prod(shape)
      lines.append('shape: %15s, %5d, %s, %s' %
                   (shape, num_params, var.name, var.dtype))

  return '\n'.join(lines)


def summary_writer(log_dir):
  """Convenient wrapper for writing summaries."""
  writer = tf.summary.FileWriter(log_dir)

  def call(step, **value_dict):
    """Adds values into the summary."""
    summary = tf.Summary()
    for tag, value in value_dict.items():
      if isinstance(value, (float, np.float32)):
        summary.value.add(tag=tag, simple_value=value)

      else:
        l = len(value)
        for i in range(l):
          summary.value.add(tag=tag + '_' + str(i), simple_value=value[i])
    writer.add_summary(summary, step)
    writer.flush()

  return call


def load_and_save_params(default_params, exp_dir, ignore_existing=False):
  """Loads and saves parameters."""
  default_params = json.loads(json.dumps(default_params))
  param_path = os.path.join(exp_dir, 'params.json')
  logging.info("Searching for '%s'", param_path)
  if os.path.exists(param_path) and not ignore_existing:
    logging.info('Loading existing params.')
    with open(param_path, 'r') as fd:
      params = json.load(fd)
    default_params.update(params)

  with open(param_path, 'w') as fd:
    json.dump(default_params, fd, indent=4)

  return default_params
