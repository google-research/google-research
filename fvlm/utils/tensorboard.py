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

"""Custom tensorboard writer that also allows saving as a file."""
import os
from flax.metrics import tensorboard
import tensorflow as tf


class SummaryWriter(tensorboard.SummaryWriter):
  """Saves data in event and summary protos for tensorboard.

  Also allows writing directly to a file, useful for datasets such as NoCaps
  where the json file must be saved to be submitted to the eval server.
  """

  def __init__(self, log_dir):
    super().__init__(log_dir)
    self.log_dir = log_dir

  def write_file(self, filename, data):
    with tf.io.gfile.GFile(os.path.join(self.log_dir, filename), 'w') as out:
      out.write(data)
