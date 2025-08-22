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

"""Trainer binary."""
from absl import app
from absl import flags
import gin
import jax
import tensorflow.compat.v2 as tf

from utils import trainer_utils


_CONFIG_PATH = flags.DEFINE_string('config_path', None, 'paths to gin config.')
_CONFIG_OVERRIDES = flags.DEFINE_multi_string(
    'config_overrides',
    None,
    'Gin bindings to override the config given in config_path flag.',
)
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None,
                                  'Path to model checkpoints and summaries.')
_DEFAULT_PRECISION = flags.DEFINE_string(
    'default_precision', None, 'The default matmul precision for XLA.')
_MODE = flags.DEFINE_enum(
    'mode',
    'train_and_eval',
    ['train', 'eval', 'train_and_eval'],
    'Which mode to run the jobs.',
)


def main(_):
  jax.config.update('jax_default_matmul_precision', _DEFAULT_PRECISION.value)
  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')
  gin.parse_config_files_and_bindings(
      [_CONFIG_PATH.value], _CONFIG_OVERRIDES.value
  )
  if _MODE.value == 'train' or _MODE.value == 'train_and_eval':
    trainer_utils.train(_OUTPUT_DIR.value)
  if _MODE.value == 'eval' or _MODE.value == 'train_and_eval':
    trainer_utils.evaluate(_OUTPUT_DIR.value)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir', 'config_path'])
  app.run(main)
