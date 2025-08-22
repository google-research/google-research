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

r"""Computes the Successor Representation for a discrete PuddleWorld.

Example command to run locally:

python -m aux_tasks.puddle_world.compute_successor_representation \
  --output_dir=/tmp/puddle_world

"""

import fractions
import json
import time
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
import numpy as np
from shapely import geometry

from aux_tasks.puddle_world import arenas
from aux_tasks.puddle_world import puddle_world
from aux_tasks.puddle_world import utils as pw_utils

_ARENA_NAME = flags.DEFINE_enum(
    'arena_name', 'sutton', arenas.ARENA_NAMES,
    'The name of the arena to load.')
_NUM_BINS = flags.DEFINE_integer(
    'num_bins', 10,
    'The number of bins to use in both the width and height directions.')
_NUM_ROLLOUTS_PER_START_STATE = flags.DEFINE_integer(
    'num_rollouts_per_start_state', 100,
    'The number of rollouts to perform for each start state.')
_ROLLOUT_LENGTH = flags.DEFINE_integer(
    'rollout_length', 50,
    'The number of rollouts to perform for each start state.')
_GAMMA = flags.DEFINE_float(
    'gamma', 0.9,
    'The discount factor to use.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None,
    'The directory to store the results of the experiment.',
    required=True)
_NUM_SHARDS = flags.DEFINE_integer(
    'num_shards', 1, 'The number of shards to use.')
_SHARD_IDX = flags.DEFINE_integer(
    'shard_idx', 0, 'The index of the current shard.')


def _compute_start_and_end_indices(
    shard, num_shards, num_states):
  # We want to avoid rounding errors when calculating shards,
  # so we use fractions.
  percent_work_to_complete = fractions.Fraction(1, num_shards)
  start_idx = round(shard * percent_work_to_complete * num_states)
  end_idx = round((shard + 1) * percent_work_to_complete * num_states)
  return start_idx, end_idx


def _make_file_name(shard, num_shards):
  return f'sr.np-{shard:05d}-of-{num_shards:05d}'


def _maybe_save_metadata(output_dir):
  """Saves metadata if this is the first shard."""
  if _SHARD_IDX.value == 0:
    # Write some metadata to make analysis easier at the end.
    metadata = {
        'arena_name': _ARENA_NAME.value,
        'num_bins': _NUM_BINS.value,
        'num_shards': _NUM_SHARDS.value,
        'gamma': _GAMMA.value,
        'num_rollouts_per_start_state': _NUM_ROLLOUTS_PER_START_STATE.value,
        'rollout_length': _ROLLOUT_LENGTH.value,
    }

    json_file_path = output_dir / 'metadata.json'
    with json_file_path.open('w') as f:
      json.dump(metadata, f)


def _save_results(output_dir,
                  shard,
                  num_shards,
                  result):
  """Saves the data for this shard."""
  # Before saving, make sure the path exists.
  output_dir.mkdir(exist_ok=True)

  _maybe_save_metadata(output_dir)

  file_path = output_dir / _make_file_name(shard, num_shards)
  with file_path.open('wb') as f:
    np.save(f, result)


def _load_shard_data(
    output_dir, shard, num_shards):
  """Loads data for a particular shard."""
  path = output_dir / _make_file_name(shard, num_shards)

  # Only try 720 times (2 hours with 10 seconds per try).
  for _ in range(720):
    try:
      with path.open('rb') as f:
        data = np.load(f)
        return data
    except (FileNotFoundError, OSError, ValueError):
      # Couldn't load the file for some reason...
      time.sleep(10)
  raise RuntimeError(
      f'Couldn\'t load data for shard {shard} after 720 attempts.')


def _maybe_combine_shards(
    output_dir, shard, num_shards):
  """Combines shards if this is the first shard."""
  if shard != 0:
    logging.info('Not the first shard. Exiting.')
    return

  logging.info('Combining results...')
  shard_results = []
  for i in range(num_shards):
    shard_results.append(_load_shard_data(output_dir, i, num_shards))
  all_data = np.concatenate(shard_results)

  # Subtract the row average. This centers the data.
  all_data = all_data - np.mean(all_data, axis=1, keepdims=True)

  with (output_dir / 'sr.np').open('wb') as f:
    np.save(f, all_data)

  # Write the left singular vectors.
  left_singular_vectors, _, _ = np.linalg.svd(all_data)
  with (output_dir / 'svd.np').open('wb') as f:
    np.save(f, left_singular_vectors)

  # If this was successful, delete all shards.
  for i in range(num_shards):
    (output_dir / _make_file_name(i, num_shards)).unlink()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Set up the arena.
  puddles = arenas.get_arena(_ARENA_NAME.value)
  pw = puddle_world.PuddleWorld(
      puddles=puddles, goal_position=geometry.Point((1.0, 1.0)))
  dpw = pw_utils.DiscretizedPuddleWorld(pw, _NUM_BINS.value)
  num_states = dpw.num_states

  # Work out which work this shard should do.
  start_idx, end_idx = _compute_start_and_end_indices(
      _SHARD_IDX.value, _NUM_SHARDS.value, num_states)
  logging.info('start idx: %d, end idx: %d', start_idx, end_idx)

  result_matrices = list()
  for start_state in range(start_idx, end_idx):
    logging.info('Starting iteration %d', start_state)
    result_matrix = np.zeros(
        (_NUM_ROLLOUTS_PER_START_STATE.value, num_states), dtype=np.float32)

    for i in range(_NUM_ROLLOUTS_PER_START_STATE.value):
      s = dpw.sample_state_in_bin(start_state)
      rollout = pw_utils.generate_rollout(dpw, _ROLLOUT_LENGTH.value, s)
      result_matrix[i] = pw_utils.calculate_empricial_successor_representation(
          dpw, rollout, _GAMMA.value)

    result_matrices.append(np.mean(result_matrix, axis=0))

  output_dir = epath.Path(_OUTPUT_DIR.value)
  _save_results(
      output_dir,
      _SHARD_IDX.value,
      _NUM_SHARDS.value,
      np.stack(result_matrices, axis=0))
  _maybe_combine_shards(output_dir, _SHARD_IDX.value, _NUM_SHARDS.value)


if __name__ == '__main__':
  app.run(main)
