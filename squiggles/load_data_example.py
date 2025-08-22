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

"""Demonstrates iterating over a saved dataset."""

from typing import Sequence

from absl import app
from absl import flags

from squiggles import generate_data

_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    None,
    'Directory containing all splits of the dataset',
    required=True,
)
_SPLIT = flags.DEFINE_string(
    'split', 'train',
    'Which dataset split to use. Usually either "test" or "train".')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  ds = generate_data.read_dataset(base_dir=_DATA_DIR.value, split=_SPLIT.value)
  i = -1
  for i, example in enumerate(ds):
    if i < 2:
      print(example)
  print(f'num examples overall: {i + 1}')


if __name__ == '__main__':
  app.run(main)
