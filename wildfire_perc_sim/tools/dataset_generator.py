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

"""Dataset generator example."""
from typing import Sequence

from absl import app
from absl import flags

import tensorflow_datasets as tfds

from wildfire_perc_sim import datasets  # pylint:disable=unused-import

FLAGS = flags.FLAGS

DATASET = flags.DEFINE_string(
    'dataset', 'wildfire_dataset/wind_fixed_ts', 'Dataset to generate.')
DATA_PATH = flags.DEFINE_string(
    'data_path', '', 'Path for saving the dataset.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  ds_builder = tfds.builder(DATASET.value, data_dir=DATA_PATH.value)
  ds_builder.download_and_prepare()
  ds = ds_builder.as_dataset()
  for data in tfds.as_numpy(ds['train'].batch(1)):
    print('Example: single-batch mean burn duration:',
          data['burn_duration'].mean())
    break


if __name__ == '__main__':
  app.run(main)
