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

"""Utility for saving/loading training configs."""
import os
import tensorflow.compat.v1 as tf
import yaml


def save_config(config, logdir=None):
  """Save a new configuration by name.

  If a logging directory is specified, is will be created and the configuration
  will be stored there. Otherwise, a log message will be printed.

  Args:
    config: Configuration object.
    logdir: Location for writing summaries and checkpoints if specified.

  Returns:
    Configuration object.
  """
  if logdir:
    # with config.unlocked:
    config.logdir = logdir
    message = 'Start a new run and write summaries and checkpoints to {}.'
    tf.logging.info(message.format(config.logdir))
    tf.gfile.MakeDirs(config.logdir)
    config_path = os.path.join(config.logdir, 'config.yaml')
    with tf.gfile.GFile(config_path, 'w') as file_:
      yaml.dump(config, file_, default_flow_style=False)
  else:
    message = (
        'Start a new run without storing summaries and checkpoints since no '
        'logging directory was specified.')
    tf.logging.info(message)
  return config


def load_config(logdir):
  """Load a configuration from the log directory.

  Args:
    logdir: The logging directory containing the configuration file.

  Raises:
    IOError: The logging directory does not contain a configuration file.

  Returns:
    Configuration object.
  """
  config_path = logdir and os.path.join(logdir, 'config.yaml')
  if not config_path or not tf.gfile.Exists(config_path):
    message = (
        'Cannot resume an existing run since the logging directory does not '
        'contain a configuration file.')
    raise IOError(message)
  with tf.gfile.GFile(config_path, 'r') as file_:
    config = yaml.load(file_)
  print('Resume run and write summaries and checkpoints to {}.'.format(
      config.logdir))
  return config
