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

"""Utility functions."""

import random
import numpy as np
import torch


# def get_device() -> torch.device:
#   """Returns the GPU device if available else CPU."""
#   return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path():
  """Returns the base bath where to log accuracies and tensorboard data."""
  return "/tmp/data/"


def set_random_seed(seed):
  """Sets the seeds at a certain value.

  Args:
    seed: the value to be set
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
