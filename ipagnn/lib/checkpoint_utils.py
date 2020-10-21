# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Checkpoint utils for the Learned Interpreters framework."""

import os

from absl import logging  # pylint: disable=unused-import
from flax import serialization
from flax.training import checkpoints
import tensorflow as tf

from ipagnn.lib import path_utils

gfile = tf.io.gfile


CHECKPOINT_PREFIX = 'ckpt'


def build_checkpoint_dir(run_dir):
  """Builds a checkpoint_dir (str) for the given run_dir."""
  return os.path.join(run_dir, 'checkpoints')


def build_checkpoint_prefix(run_dir):
  """Builds a checkpoint_prefix (str) for the given run_dir."""
  checkpoint_dir = build_checkpoint_dir(run_dir)
  return os.path.join(checkpoint_dir, CHECKPOINT_PREFIX)


def build_checkpoint_path(run_dir, checkpoint_id):
  """Builds a checkpoint_path (str) from the given run_dir and checkpoint_id."""
  return os.path.join(build_checkpoint_dir(run_dir),
                      f'{CHECKPOINT_PREFIX}_{checkpoint_id}')


def build_success_path(run_dir):
  """Builds a path (str) indicating success from the given run_dir."""
  return os.path.join(run_dir, 'SUCCESS')


def get_checkpoint_id(checkpoint_path):
  """Gets the checkpoint_id for the given checkpoint_path."""
  basename = os.path.basename(checkpoint_path)  # e.g. ckpt_67
  index_str = basename.split('_')[-1]  # e.g. 67
  return int(index_str)


def get_checkpoint_dir(checkpoint_path):
  """Gets the checkpoint_dir containing the given checkpoint_path."""
  return os.path.dirname(checkpoint_path)


def get_run_dir(checkpoint_path):
  """Gets the run_dir containing the given checkpoint_path."""
  checkpoint_dir = get_checkpoint_dir(checkpoint_path)
  return os.path.dirname(checkpoint_dir)


def get_all_checkpoint_paths(checkpoint_dir):
  """Gets all checkpoint paths in the given checkpoint_dir."""
  glob_path = os.path.join(checkpoint_dir, f'{CHECKPOINT_PREFIX}_*')
  checkpoint_files = checkpoints.natural_sort(gfile.glob(glob_path))
  ckpt_tmp_path = os.path.join(checkpoint_dir, f'{CHECKPOINT_PREFIX}_tmp')
  checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
  return checkpoint_files


def latest_checkpoint(checkpoint_dir):
  all_checkpoint_paths = get_all_checkpoint_paths(checkpoint_dir)
  if all_checkpoint_paths:
    return all_checkpoint_paths[-1]


def get_specified_checkpoint_path(run_dir, checkpoint_config):
  """Gets a preexisting checkpoint_path if one is specified by the config."""
  assert is_checkpoint_specified(checkpoint_config)

  if checkpoint_config.path:
    assert not checkpoint_config.id
    assert not checkpoint_config.run_dir
    return path_utils.expand_glob(checkpoint_config.path)

  if checkpoint_config.run_dir:
    config_run_dir = path_utils.expand_glob(checkpoint_config.run_dir)

    if checkpoint_config.id:
      # Specific checkpoint id (e.g. ckpt_101) is specified.
      return build_checkpoint_path(config_run_dir,
                                   checkpoint_config.id)
    else:
      # run_dir specified but no checkpoint id specified.
      checkpoint_dir = build_checkpoint_dir(config_run_dir)
      checkpoint_path = latest_checkpoint(checkpoint_dir)
      if checkpoint_path is not None:
        return checkpoint_path
      else:
        raise RuntimeError('No checkpoint found in directory:', checkpoint_dir)

  if checkpoint_config.id:
    return build_checkpoint_path(run_dir, checkpoint_config.id)

  raise RuntimeError('Unexpected runtime error. No checkpoint specified.',
                     run_dir, checkpoint_config)


def is_checkpoint_specified(checkpoint_config):
  """Whether or not the checkpoint config specifies a checkpoint."""
  return (checkpoint_config.path
          or checkpoint_config.run_dir
          or checkpoint_config.id)


def save_checkpoint(checkpoint_dir, target, step):
  """Saves a checkpoint."""
  return checkpoints.save_checkpoint(checkpoint_dir, target, step,
                                     prefix=f'{CHECKPOINT_PREFIX}_',
                                     keep=3)


def restore_checkpoint(checkpoint_path, target):
  """Restores a checkpoint."""
  with gfile.GFile(checkpoint_path, 'rb') as f:
    return serialization.from_bytes(target, f.read())


def handle_restart_behavior(checkpoint_path, target, config):
  """Abort, restore, replace a checkpoint, or create a new checkpoint."""
  if checkpoint_path is None:
    pass  # Nothing to do here.
  elif config.runner.restart_behavior == 'abort':
    raise RuntimeError(
        'A preexisting checkpoint was found in the run directory. Aborting.')
  elif config.runner.restart_behavior == 'restore':
    logging.info('Restoring from checkpoint: %s', checkpoint_path)
    target = restore_checkpoint(checkpoint_path, target)
  else:
    raise ValueError('Unexpected restart_behavior config',
                     config.runner.restart_behavior)
  return target
