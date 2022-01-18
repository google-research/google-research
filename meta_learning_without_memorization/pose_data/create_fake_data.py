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

# Lint as: python3
"""Create fake train_data.pkl and val_data.pkl for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS


def create_example():
  x = []
  y = []
  for _ in range(100):
    x.append(np.zeros((128, 128), dtype=np.uint8))
    y.append(np.random.rand(3).tolist())
  return x, y


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train = zip(*[create_example() for _ in range(10)])
  valid = zip(*[create_example() for _ in range(10)])

  with open('train_data.pkl', 'wb') as out:
    pickle.dump(train, out)
  with open('val_data.pkl', 'wb') as out:
    pickle.dump(valid, out)


if __name__ == '__main__':
  app.run(main)
