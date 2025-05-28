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

"""Write metadata files to make dataset easy to load."""

import pathlib
from typing import Sequence

from absl import app
from absl import flags

from squiggles import generate_data

_BASE_PATH = flags.DEFINE_string(
    'base_path',
    None,
    'Output filename base (including directory path).',
    required=True,
)
_SQUIGGLE_ALGORITHM = flags.DEFINE_enum_class(
    'squiggle_algorithm',
    generate_data.LatentSpace.UNDEFINED,
    generate_data.LatentSpace,
    'Which latent space to use to generate squiggles: "sine_net" or "taylor"',
)
_SAMPLES_PER_SHARD = flags.DEFINE_integer(
    'samples_per_shard',
    None,
    'The number of squiggle examples to include in each shard.',
    required=True,
)
_NUM_TRAIN_SHARDS = flags.DEFINE_integer(
    'num_train_shards',
    1,
    'how many shards in the "train" split',
)
_NUM_TEST_SHARDS = flags.DEFINE_integer(
    'num_test_shards',
    1,
    'how many shards in the "test" split',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError(
        f'Too many command-line arguments. Here\'s the excess: {argv[0:]}')
  base_dir = pathlib.PurePath(_BASE_PATH.value).parent
  latent_space = _SQUIGGLE_ALGORITHM.value
  taylor_hidden_size = (
      generate_data.LATENT_SPACE_TO_DEFAULT_HIDDEN_SIZE[latent_space]
      if latent_space == generate_data.LatentSpace.TAYLOR else None)
  sine_net_hidden_size = (
      generate_data.LATENT_SPACE_TO_DEFAULT_HIDDEN_SIZE[latent_space]
      if latent_space == generate_data.LatentSpace.SINE_NET else None)
  generate_data.write_metadata(
      str(base_dir),
      _SAMPLES_PER_SHARD.value,
      {
          'train': _NUM_TRAIN_SHARDS.value,
          'test': _NUM_TEST_SHARDS.value
      },
      taylor_hidden_size=taylor_hidden_size,
      sine_net_hidden_size=sine_net_hidden_size,
  )


if __name__ == '__main__':
  app.run(main)
