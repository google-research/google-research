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

# Lint as: python3
"""Replay buffers."""

from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np


class ReplayBuffer():
  """Uniform replay buffer.

  Attributes:
    buffer: buffer where experience is stored
    buffer_size: maximum replay buffer size
  """

  def __init__(self, buffer_size=50000):
    self.buffer = []
    self.buffer_size = buffer_size
    self._delete_frac = 1./buffer_size if buffer_size < 10000 else 0.0001

  def add(self, experience):
    """Store experience in the buffer."""
    self.buffer.append(experience)
    if len(self.buffer) > self.buffer_size:
      self.buffer = self.buffer[int(0.0001 * self.buffer_size):]

  def sample(self, size):
    """Sample a batch of experience with specified batch size."""
    if len(self.buffer) >= size:
      experience_buffer = self.buffer
    else:
      experience_buffer = self.buffer * size
    sample = random.sample(experience_buffer, size)
    sample = np.array(sample)
    return np.copy(sample)
