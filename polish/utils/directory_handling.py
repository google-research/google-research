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

"""Helper methods to perform directory handling (e.g. making directories)."""
import os
from absl import logging


def ensure_dir_exists(directory):
  """Creates local directories if they don't exist."""
  if not os.path.isdir(directory):
    logging.info("Making dir %s", directory)
    os.makedirs(directory)
