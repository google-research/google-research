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

import collections
from absl import flags
FLAGS = flags.FLAGS


class RunningAverageLoss(object):
  """A loss which tracks a running average."""

  def __init__(self, name, run_avg_len):
    self.name = name
    self.run_avg_len = run_avg_len
    self.history = []
    self.curr_state = collections.deque()
    self.curr_value = 0.
    self.all_values = {}

  def get_value(self, i=None):
    """Returns the average of this loss over a range of epochs.

    Args:
      i (int): the epoch number to take the average at. If i is None, we take
        the run_avg_len most recent epochs. Otherwise, we take the run_avg_len
        epochs leading up to i.
    Returns:
      avg (float): the average loss over the specified epoch range.
    """
    return self.all_values[i if i is not None else len(self.history) - 1]

  def get_history(self):
    return self.history

  def update(self, x):
    """Update the loss tracker with a new point x.

    Args:
      x (number): new loss point to add to the object.
    """

    # Add the new data point to our loss history.
    self.history.append(x)

    # Add the point to our current window
    self.curr_state.append(x)

    # If our history length equals our window length, we can start
    # caching the running average calculation
    if len(self.history) == self.run_avg_len:
      self.curr_value = self.curr_value / self.run_avg_len

    # When we add a point to the end of the window, we pop one off the front
    if len(self.curr_state) >= self.run_avg_len:
      # Update our running average current value by adding the new point ...
      self.curr_value += float(x) / self.run_avg_len
      # ... and removing the oldest one
      first_val = self.curr_state.popleft()
      # Store our current running average value
      self.all_values[len(self.history) - 1] = self.curr_value
      # Remove the oldest value from our running average
      self.curr_value -= float(first_val) / self.run_avg_len
    else:
      # If our window is longer than our history, we need to do the averaging
      # manually, since the denominator is a different value each time
      self.curr_value += float(x)
      self.all_values[len(self.history) - 1] = (self.curr_value /
                                                len(self.history))
