# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A running average loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
FLAGS = flags.FLAGS


class RunningAverageLoss(object):
  """A loss which tracks a running average."""

  def __init__(self, name, run_avg_len):
    self.name = name
    self.run_avg_len = run_avg_len
    self.history = []

  def get_value(self, i=None):
    """Returns the average of this loss over a range of epochs.

    Args:
      i (int): the epoch number to take the average at. If i is None, we take
        the run_avg_len most recent epochs. Otherwise, we take the run_avg_len
        epochs leading up to i.
    Returns:
      avg (float): the average loss over the specified epoch range.
    """
    if i is None:
      return np.mean(self.history[-self.run_avg_len:])
    else:
      return np.mean(self.history[i - self.run_avg_len + 1: i + 1])

  def get_history(self):
    return self.history

  def update(self, x):
    self.history.append(x)
