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

"""Main file for inference evaluater."""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
import ml_collections
import tensorflow as tf

from sudoku_gpt.inference import inference_evaluater

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Directory to store model data.')
_CKPT_LOC = flags.DEFINE_string('ckpt_loc', None, 'Checkpoint location.')
flags.mark_flags_as_required(['workdir', 'ckpt_loc'])


def get_config():
  """Get the default hyperparameter configuration.

  Dataset choices:
  othello: For othello game

  sudoku: Sudoku game but fixed order (order: row-wise left to right)
  ordered-sudoku: Sudoku game data with the order of solver
  ordered-sudoku-wo-random-guessing-w-candidates-train-test: Uses sudoku games
              that can be solved with 7 human logics. It has train-test split.
              Does not contain examples with random guessing. Has penciling
              candidates for 10 locations and strategies used for each of the
              move.

  Returns:
    A ConfigDict with all the experiment related settings.
  """
  config = ml_collections.ConfigDict()
  config.dataset = 'ordered-sudoku'
  config.sampling_method = 'greedy-row-col'

  config.restore_checkpoint = False  # Not implemented yet

  ### Training related parameters
  config.max_steps = 2**22
  config.dtype = jax.numpy.bfloat16
  config.minibatch_size = 64

  if 'sudoku' in config.dataset:
    config.block_size = 81
    config.seq_len = 3 * config.block_size
    config.vocab_size = 11
    config.start_index = 32
    config.set_accuracy = 'top-k'  # Choice = "top-k", "all"
    config.set_accuracy_top_k = 20
  elif config.dataset == 'othello':
    config.block_size = 60
    config.seq_len = config.block_size
    config.vocab_size = 65
    config.start_index = 0  # Does not get used

  ### Model related parameters
  config.num_heads = 8
  config.num_layers = 8
  config.emb_dim = 576
  config.qkv_dim = 576
  config.mlp_dim = 6 * config.qkv_dim
  config.dropout_rate = 0.2
  config.attention_dropout_rate = 0.1
  config.optimizer = 'adamw'

  ### Training related parameters
  config.learning_rate = 1e-4  # Base learning rate.
  config.end_lr_factor = 0.2
  config.warmup_tokens = 2**10
  config.weight_decay = 5e-3
  config.seed = 9
  config.save_checkpoint = True
  config.save_every_steps = 2**13
  config.num_examples = 10

  ### Evaluation related parameters
  config.eval_every_steps = 500
  config.eval_epochs = 10
  config.beam_search_n = 3  ## Always use < 9

  if config.dataset == 'othello':
    config.dataset_path = None
  elif config.dataset == 'sudoku':
    config.dataset_path = None
  elif config.dataset == 'ordered-sudoku':
    config.dataset_path = None
  elif (
      config.dataset
      == 'ordered-sudoku-wo-random-guessing-w-candidates-train-test'
  ):
    config.test_puzzle_path = None
    config.test_cands_path = None

  return config


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f'process_index: {jax.process_index()}, '
      f'process_count: {jax.process_count()}'
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, _WORKDIR.value, 'workdir'
  )

  cfgs = get_config()
  logging.info(cfgs)

  inference_evaluater.inference_evaluate(cfgs, _WORKDIR.value, _CKPT_LOC.value)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
