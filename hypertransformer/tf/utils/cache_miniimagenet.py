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

"""Converts miniimagenet dataset from pickled files to NumPy."""

import dataclasses
import os
import pickle

from typing import Any, Dict, Sequence

from absl import app
from absl import flags

import numpy as np

INPUT_PATH = flags.DEFINE_string(
    'input_path', '', 'Path with miniImageNet pickle files.')
OUTPUT_PATH = flags.DEFINE_string(
    'output_path', '', 'Path with miniImageNet pickle files.')


@dataclasses.dataclass
class Sources:
  data: Dict[Any, Any] = dataclasses.field(default_factory=dict)


def pickle_path(root, split):
  path = os.path.join(root, f'mini-imagenet-cache-{split}.pkl')
  if not os.path.exists(path):
    raise RuntimeError(f'Pickle file {path} is not found!')
  return path


def get_data(root):
  data = {split: pickle.loads(open(pickle_path(root, split), 'rb').read())
          for split in ['train', 'test', 'val']}
  return Sources(data=data)


def get_combined(data):
  outputs = []
  for split in ['train', 'val', 'test']:
    classes = data.data[split]['class_dict']
    images = data.data[split]['image_data']
    for values in classes.values():
      from_class = np.min(values)
      to_class = np.max(values) + 1
      outputs.append(images[from_class:to_class])
  return np.stack(outputs, axis=0)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  data = get_data(INPUT_PATH.value)
  combined = get_combined(data)
  assert combined.shape == (100, 600, 84, 84, 3)
  try:
    os.makedirs(OUTPUT_PATH.value)
  finally:
    np.save(os.path.join(OUTPUT_PATH.value, 'miniimagenet'), combined)

if __name__ == '__main__':
  app.run(main)
