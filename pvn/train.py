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

# pyformat: disable
r"""The entry point for running a DSM module.

"""
# pyformat: enable

import functools

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
from ml_collections import config_dict
from ml_collections import config_flags
import tensorflow as tf

from pvn import offline
from pvn import online


MODULES = {'offline': offline.train, 'online': online.train}

# Parse jax flags, this will automatically update jax.config.
jax.config.parse_flags_with_absl()

WORKDIR = epath.DEFINE_path(
    'workdir',
    None,
    'Base working directory to host all required sub-directories.',
    required=True,
)
CHECKPOINT_DIR = epath.DEFINE_path(
    'checkpoint_dir',
    None,
    'Checkpoint directory. Optional.',
)

CONFIG = config_flags.DEFINE_config_file('config', lock_config=True)
MODULE = flags.DEFINE_enum('module', 'offline', list(MODULES.keys()), 'Module.')
# Useful debugging flags
# --pdb

# Useful Jax debugging flags
# --jax_debug_nans
# --jax_log_compiles
# --jax_numpy_rank_promotion=warn

# Useful logger level flags
# --logger_levels=DEBUG
# --alsologtostderr
# --stderrthreshold=DEBUG




def main(_):
  """Main method."""

  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX Process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX Process Count: %d', jax.process_count())
  logging.info('JAX Devices: %r', jax.devices())
  logging.info('JAX Device Count: %d', jax.device_count())
  logging.info('JAX Local Devices: %r', jax.local_devices())
  logging.info('JAX Local Device Count: %d', jax.local_device_count())
  # Log config
  logging.info(CONFIG.value)

  config = config_dict.FrozenConfigDict(CONFIG.value)

  # Create the working directory if it doesn't already exist
  WORKDIR.value.mkdir(parents=True, exist_ok=True)
  # Check that checkpoint_dir exists
  if CHECKPOINT_DIR.value is not None and not CHECKPOINT_DIR.value.exists():
    raise RuntimeError(
        f"Checkpoint directory {CHECKPOINT_DIR.value} doesn't exist"
    )


  if train := MODULES.get(MODULE.value, None):
    if CHECKPOINT_DIR.value:
      train = functools.partial(train, checkpoint_dir=CHECKPOINT_DIR.value)
    train(workdir=WORKDIR.value, config=config)
  else:
    raise AttributeError(f"Module {MODULE.value} doesn't exist.")


if __name__ == '__main__':
  app.run(main)
