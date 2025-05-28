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

"""Parse commandline args."""
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=dangerous-default-value
# pylint: disable=unused-import
# pylint: disable=g-multiple-import
import argparse
import pdb
import sys


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description="Train a OCI Model")
  parser.add_argument(
      "--gpu",
      dest="gpu_id",
      help="GPU device id to use [0]",
      default=0,
      type=int)
  parser.add_argument(
      "--cfg",
      dest="cfg_file",
      help="optional config file",
      default=None,
      type=str)
  parser.add_argument(
      "--set",
      dest="set_cfgs",
      help="set config keys",
      default=None,
      nargs=argparse.REMAINDER,
  )
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  results = parser.parse_args()
  return results
