# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Implements file saving / loading."""
import os
import re
import time
from typing import Optional, Sequence, Union

from absl import logging
from flax import serialization
from flax import struct
from tensorflow.io import gfile

from autoregressive_diffusion.experiments.images import custom_train_state

# TODO(emielh) move to general folder.

# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
    r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')


def _checkpoint_path(ckpt_dir, step, prefix):
  """Construct full checkpoint path + filename."""
  return os.path.join(ckpt_dir, f'{prefix}{step}')


def latest_checkpoint_path(ckpt_dir, prefix):
  glob_path = os.path.join(ckpt_dir, f'{prefix}*')
  checkpoint_files = natural_sort(gfile.glob(glob_path))
  ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
  checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
  return checkpoint_files[-1] if checkpoint_files else None


def natural_sort(file_list, signed = True):
  """Natural sort for filenames with numerical substrings.

  Args:
    file_list: List of paths to sort containing numerical
      substrings.
    signed: If leading '-' (or '+') signs should be included in
      numerical substrings as a sign or treated as a separator.

  Returns:
    List of filenames sorted 'naturally', not lexicographically: any
    integer substrings are used to subsort numerically. e.g.
    file_1, file_10, file_2  -->  file_1, file_2, file_10
    file_0.1, file_-0.2, file_2.0  -->  file_-0.2, file_0.1, file_2.0
  """
  float_re = SIGNED_FLOAT_RE if signed else UNSIGNED_FLOAT_RE
  def maybe_num(s):
    if float_re.match(s):
      return float(s)
    else:
      return s
  def split_keys(s):
    return [maybe_num(c) for c in float_re.split(s)]
  return sorted(file_list, key=split_keys)


class SaveState(struct.PyTreeNode):
  """Custom savestate to include the epoch number, practical for restarting."""
  train_state: custom_train_state.TrainState
  step: int


def save_checkpoint(ckpt_dir,
                    state,
                    step,
                    keep = 10,
                    prefix = 'ckpt_'):
  """Save a checkpoint of the model.

  Attempts to be pre-emption safe by writing to temporary before
  a final rename and cleanup of past files.

  Args:
    ckpt_dir: Path to store checkpoint files in.
    state: Serializable flax object, usually a flax optimizer.
    step: Training step number or other metric number.
    keep: Number of checkpoints to keep.
    prefix: Checkepoint filename prefix.

  Returns:
    Filename of saved checkpoint.
  """
  # Write temporary checkpoint file.
  logging.info('Saving checkpoint at step: %s', step)
  ckpt_tmp_path = os.path.join(ckpt_dir, 'tmp')
  ckpt_destination_path = _checkpoint_path(ckpt_dir, step, prefix)
  gfile.makedirs(os.path.dirname(ckpt_destination_path))

  save_state = SaveState(state, step)
  with gfile.GFile(ckpt_tmp_path, 'wb') as fp:
    fp.write(serialization.to_bytes(save_state))

  # Rename once serialization and writing finished.
  gfile.rename(ckpt_tmp_path, ckpt_destination_path, overwrite=True)

  logging.info('Saved checkpoint at %s', ckpt_destination_path)

  # Remove old checkpoint files.
  base_path = os.path.join(ckpt_dir, f'{prefix}')
  checkpoint_files = natural_sort(gfile.glob(base_path + '*'))
  if len(checkpoint_files) > keep:
    old_ckpts = checkpoint_files[:-keep]
    for path in old_ckpts:
      logging.info('Removing checkpoint: %s', path)
      gfile.remove(path)

  return ckpt_destination_path


def restore_from_path(
    ckpt_dir,
    target,
    step = None,
    prefix = 'ckpt_'):
  """Restores a checkpoint from a directory path, if available."""
  ckpt_destination_path = latest_checkpoint_path(ckpt_dir, prefix)

  if ckpt_destination_path is None:
    logging.info('No checkpoints found, starting from the beginning.')
    return target, step

  logging.info('Restoring checkpoint: %s', ckpt_destination_path)
  save_state = SaveState(target, step)
  with gfile.GFile(ckpt_destination_path, 'rb') as fp:
    save_state = serialization.from_bytes(save_state, fp.read())
    return save_state.train_state, save_state.step


def wait_for_new_checkpoint(ckpt_dir,
                            last_ckpt_path = None,
                            seconds_to_sleep = 1,
                            timeout = None,
                            prefix = 'ckpt_'):
  """Waits until a new checkpoint file is found.

  Args:
    ckpt_dir: Directory in which checkpoints are saved.
    last_ckpt_path: Last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: Number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: Maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.
    prefix: Name prefix of checkpoint files.

  Returns:
    A new checkpoint path, or None if the timeout was reached.
  """
  logging.info('Waiting for new checkpoint in %s', ckpt_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    ckpt_path = latest_checkpoint_path(ckpt_dir, prefix)
    if ckpt_path is None or ckpt_path == last_ckpt_path:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info('Found new checkpoint: %s', ckpt_path)
      return ckpt_path


def checkpoints_iterator(ckpt_dir,
                         target,
                         timeout = None,
                         min_interval_secs = 0,
                         prefix = 'ckpt_'):
  """Repeatedly yield new checkpoints as they appear.

  Args:
    ckpt_dir: Directory in which checkpoints are saved.
    target: Matching object to rebuild via deserialized state-dict.
    timeout: Maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.
    min_interval_secs: Minimum number of seconds between yielding checkpoints.
    prefix: str: name prefix of checkpoint files.

  Yields:
    New checkpoint path if `target` is None, otherwise `target` updated from
    the new checkpoint path.
  """
  ckpt_path = None
  while True:
    new_ckpt_path = wait_for_new_checkpoint(
        ckpt_dir, ckpt_path, timeout=timeout, prefix=prefix)
    if new_ckpt_path is None:
      # Timed out.
      logging.info('Timed-out waiting for a checkpoint.')
      return
    start = time.time()
    ckpt_path = new_ckpt_path

    yield ckpt_path if target is None else restore_from_path(ckpt_path, target)

    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)
