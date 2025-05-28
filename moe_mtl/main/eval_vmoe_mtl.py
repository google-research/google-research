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

from moe_mtl.main import trainer_utils
_CONFIG_PATH = flags.DEFINE_string('config_path', None, 'paths to gin config.')
_CONFIG_PARAM = flags.DEFINE_multi_string(
    'gin_bindings', None, 'Newline separated list of Gin parameter bindings.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None,
                                  'Path to model checkpoints/summaries.')
_AES = flags.DEFINE_bool(
    'aes', False, 'launch with AES'
)


def main(_):

  # jax.config.update('jax_default_matmul_precision', _DEFAULT_PRECISION.value)
  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')
  config_params = _CONFIG_PARAM.value or []
  # enable relative paths within p5x configs.
  gin.add_config_file_search_path('third_party/py/t5x/configs')
  gin.parse_config_files_and_bindings([_CONFIG_PATH.value], config_params)
  if not _AES.value:
    trainer_utils.evaluate_vmoe_mtl(_OUTPUT_DIR.value)
  else:
    trainer_utils.evaluate_vmoe_mtl_with_aes(_OUTPUT_DIR.value)
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir', 'config_path'])
  app.run(main)
