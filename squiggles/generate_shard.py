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

"""Generate a single shard of Squiggles data."""

import os
import textwrap
from typing import Dict, Sequence

from absl import app
from absl import flags

from squiggles import generate_data

_BASE_PATH = flags.DEFINE_string(
    'base_path',
    None,
    'Output filename base (including directory path).',
    required=True,
)
_SPLIT_NAME = flags.DEFINE_string(
    'split_name',
    'train',
    'Typically "train" or "test". Incorporated into the file name but also affects random seed.',
)
_SHARD_NUM = flags.DEFINE_integer(
    'shard_num',
    0,
    'The shard number. Incorporated into filename but also affects random seed.',
)
_NUM_SHARDS = flags.DEFINE_integer(
    'num_shards',
    1,
    'The total number of shards. Incorporated into filename.',
)
_SAMPLES_PER_SHARD = flags.DEFINE_integer(
    'samples_per_shard',
    None,
    'The number of squiggle examples to include in each shard.',
    required=True,
)
_SQUIGGLE_ALGORITHM = flags.DEFINE_enum_class(
    'squiggle_algorithm',
    generate_data.LatentSpace.SINE_NET,
    generate_data.LatentSpace,
    'Which latent space to use to generate squiggles: "sine_net" or "taylor"',
)
_DATASET_CODE = flags.DEFINE_integer(
    'dataset_code',
    None,
    'Four-bit integer used in random seed. Required if `split` is anything '
    'other than "test", "train", or "validate".',
    lower_bound=0,
    upper_bound=15,
)

_SPLIT_TO_DEFAULT_DATASET_CODE: Dict[str, int] = {
    'train': 0,
    'test': 1,
    'validate': 2,
}


def main(argv):
  if len(argv) > 1:
    raise app.UsageError(
        f'Too many command-line arguments. Here\'s the excess: {argv[0:]}')
  split = _SPLIT_NAME.value
  dataset_code = _DATASET_CODE.value
  base_file = _BASE_PATH.value

  if dataset_code is None:
    try:
      dataset_code = _SPLIT_TO_DEFAULT_DATASET_CODE[split.casefold()]
    except KeyError:
      raise ValueError(
          textwrap.dedent(
              f"""Unable to generate datacode from split name "{split}".

              * If you intended your split to be one of "train", "test", or
                "validate", you probably misspelled it.
              * Otherwise, please use the `dataset_code` flag to specify a whole
                number less than 16. This number will be used in the random
                seeding to ensure this split has distinct data from other splits
                (with other datacodes)."""))
  basename = os.path.basename(base_file)
  if '_' in basename or '-' in basename or any(l.isupper() for l in basename):
    raise ValueError(
        textwrap.dedent(
            f"""Unable to use the basis for the filename "{basename}".

            * TFDS does not support underscores _.
            * TFDS does not support hyphens -.
            * TFDS changes the capitalization to lowercase."""
        ))

  shard_num = _SHARD_NUM.value
  num_shards = _NUM_SHARDS.value
  samples_per_shard = _SAMPLES_PER_SHARD.value
  latent_space = _SQUIGGLE_ALGORITHM.value
  latents, coords, labels = generate_data.generate_dataset(
      latent_space=latent_space,
      start_seed=samples_per_shard * shard_num,
      end_seed=samples_per_shard * (shard_num + 1),
      dataset_code=dataset_code,
      hidden_size=(
          generate_data.LATENT_SPACE_TO_DEFAULT_HIDDEN_SIZE[latent_space]),
      num_points=100,
  )
  taylor_latents = (
      latents if latent_space == generate_data.LatentSpace.TAYLOR else None)
  sinenet_latents = (
      latents if latent_space == generate_data.LatentSpace.SINE_NET else None)
  os.makedirs(os.path.dirname(base_file), exist_ok=True)
  generate_data.write_to_tfrecord(
      base_file=base_file,
      split=split,
      points=coords,
      labels=labels,
      shard_num=shard_num,
      num_shards=num_shards,
      taylor_latents=taylor_latents,
      sinenet_latents=sinenet_latents,
  )


if __name__ == '__main__':
  app.run(main)
