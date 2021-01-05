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

"""Utility code for the default platform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile

from cnn_quantization.tf_cnn_benchmarks import cnn_util


_ROOT_PROJECT_DIR = os.path.dirname(cnn_util.__file__)


def define_platform_params():
  """Defines platform-specific parameters.

  Currently there are no platform-specific parameters to be defined.
  """
  pass


def get_cluster_manager(params, config_proto):
  """Returns the cluster manager to be used."""
  return cnn_util.GrpcClusterManager(params, config_proto)


def get_command_to_run_python_module(module):
  """Returns a command to run a Python module."""
  python_interpretter = sys.executable
  if not python_interpretter:
    raise ValueError('Could not find Python interpreter')
  return [python_interpretter,
          os.path.join(_ROOT_PROJECT_DIR, module + '.py')]


def get_test_output_dir():
  """Returns a directory where test outputs should be placed."""
  base_dir = os.environ.get('TEST_OUTPUTS_DIR',
                            '/tmp/tf_cnn_benchmarks_test_outputs')
  if not os.path.exists(base_dir):
    os.mkdir(base_dir)
  return tempfile.mkdtemp(dir=base_dir)


def get_test_data_dir():
  """Returns the path to the test_data directory."""
  return os.path.join(_ROOT_PROJECT_DIR, 'test_data')


def _initialize(params, config_proto):
  # Currently, no platform initialization needs to be done.
  del params, config_proto


_is_initalized = False


def initialize(params, config_proto):
  global _is_initalized
  if _is_initalized:
    return
  _is_initalized = True
  _initialize(params, config_proto)
