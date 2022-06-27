# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
import random
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

flags.DEFINE_enum('arena_name', 'hydrogen', arenas.ARENA_NAMES,
                  'The name of the arena to load.')
flags.DEFINE_integer(
    'num_bins', 10,
    'The number of bins to use in both the width and height directions.')
flags.DEFINE_integer(
    'num_rollouts_per_start_state', 100,
    'The number of rollouts to perform for each start state.')
flags.DEFINE_integer(
    'rollout_length', 50,
    'The number of rollouts to perform for each start state.')
flags.DEFINE_float(
    'gamma', 0.9,
    'The discount factor to use.')
flags.DEFINE_string(
    'output_dir', None,
    'The directory to store the results of the experiment.',
    required=True)
flags.DEFINE_integer('num_shards', 1, 'The number of shards to use.')
flags.DEFINE_integer('shard_idx', 0, 'The index of the current shard.')
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  puddles = arenas.get_arena(FLAGS.arena_name)
  pw = puddle_world.PuddleWorld(
      puddles=puddles, goal_position=geometry.Point((1.0, 1.0)))

  dpw = pw_utils.DiscretizedPuddleWorld(pw, FLAGS.num_bins)
  num_states = dpw.num_states

  # We want to avoid rounding errors when calculating shards,
  # so we use fractions.
  percent_work_to_complete = fractions.Fraction(1, FLAGS.num_shards)
  start_idx = int(FLAGS.shard_idx * percent_work_to_complete * num_states)
  end_idx = int((FLAGS.shard_idx + 1) * percent_work_to_complete * num_states)
  logging.info('start idx: %d, end idx: %d', start_idx, end_idx)

  result_matrices = list()

  # TODO(joshgreaves): utils has helpful functions for generating rollouts.
  for start_state in range(start_idx, end_idx):
    logging.info('Starting iteration %d', start_state)
    result_matrix = np.zeros(
        (FLAGS.num_rollouts_per_start_state, num_states), dtype=np.float32)

    for i in range(FLAGS.num_rollouts_per_start_state):
      current_gamma = 1.0

      s = dpw.sample_state_in_bin(start_state)

      for _ in range(FLAGS.rollout_length):
        action = random.randrange(pw_utils.NUM_ACTIONS)
        transition = dpw.transition(s, action)
        s = transition.next_state

        result_matrix[i, s.bin_idx] += current_gamma
        current_gamma *= FLAGS.gamma

    result_matrices.append(np.mean(result_matrix, axis=0))

  # Before saving, make sure the path exists.
  output_dir = epath.Path(FLAGS.output_dir)
  output_dir.mkdir(exist_ok=True)

  if FLAGS.shard_idx == 0:
    # Write some metadata to make analysis easier at the end.
    metadata = {
        'arena_name': FLAGS.arena_name,
        'num_bins': FLAGS.num_bins,
        'num_shards': FLAGS.num_shards,
    }
    json_file_path = output_dir / 'metadata.json'
    with json_file_path.open('w') as f:
      json.dump(metadata, f)

  file_path = output_dir / f'sr_{start_idx}-{end_idx}.np'
  with file_path.open('wb') as f:
    np.save(f, np.stack(result_matrices, axis=0))


if __name__ == '__main__':
  app.run(main)
