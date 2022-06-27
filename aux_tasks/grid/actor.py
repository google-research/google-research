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

r"""Successor Representation worker, part of distributed learning setup.

Local run command:

python -m aux_tasks.grid.actor \
  --config=aux_tasks/grid/config.py:implicit \
  --reverb_address=localhost:1234 \
  --eval=False

"""

from collections.abc import Sequence
import random

from absl import app
from absl import flags
from ml_collections import config_dict
from ml_collections import config_flags
import numpy as np
import reverb
from shapely import geometry

from aux_tasks.puddle_world import arenas
from aux_tasks.puddle_world import puddle_world
from aux_tasks.puddle_world import utils as pw_utils

flags.DEFINE_string(
    'reverb_address',
    None,
    'The address to use to connect to the reverb server.',
    required=True)
flags.DEFINE_boolean(
    'eval',
    None,
    'If True, this worker will gather mean discounted state visitations '
    'and insert them into the eval table.')

_CONFIG = config_flags.DEFINE_config_file('config', lock_config=True)

FLAGS = flags.FLAGS

_TRAIN_TABLE = 'successor_table'
_EVAL_TABLE = 'eval_table'


def train_worker(dpw,
                 reverb_client,
                 config):
  """Train worker loop.

  Collects rollouts and writes sampled discounted state visitation to reverb
  indefinitely (xmanager should kill this job when the learner job ends).

  Args:
    dpw: The discretized puddle world to use.
    reverb_client: The reverb client for writing data to reverb.
    config: Experiment configuration.
  """
  while True:
    rollout = pw_utils.generate_rollout(dpw, config.env.pw.rollout_length)
    sr = pw_utils.calculate_empricial_successor_representation(
        dpw, rollout, config.env.pw.gamma)
    pos = np.asarray(
        [rollout[0].state.true_state.x, rollout[0].state.true_state.y],
        dtype=np.float32)
    reverb_client.insert((pos, rollout[0].state.bin_idx, sr),
                         priorities={_TRAIN_TABLE: 1.0})


def eval_worker(dpw,
                reverb_client,
                config):
  """Eval worker loop.

  Collects mean discounts state visitation from a randomly sampled start state
  and writes the result to reverb.

  Args:
    dpw: The discretized puddle world to use.
    reverb_client: The reverb client for writing data to reverb.
    config: Experiment configuration.
  """
  max_table_size = config.num_eval_points

  while reverb_client.server_info()[_EVAL_TABLE].current_size < max_table_size:
    # This sampling should be equivalent to sampling (x, y) in [0, 1].
    initial_state = random.randrange(dpw.num_states)
    start_state = dpw.sample_state_in_bin(initial_state)
    all_discounted_visitations = list()

    for _ in range(config.num_eval_rollouts):
      rollout = pw_utils.generate_rollout(
          dpw, config.env.pw.rollout_length, start_state)
      sr = pw_utils.calculate_empricial_successor_representation(
          dpw, rollout, config.env.pw.gamma)
      all_discounted_visitations.append(sr)

    sr = np.mean(np.stack(all_discounted_visitations), axis=0)
    pos = np.asarray(
        [rollout[0].state.true_state.x, rollout[0].state.true_state.y],
        dtype=np.float32)

    reverb_client.insert((pos, sr), priorities={_EVAL_TABLE: 1.0})


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = _CONFIG.value

  # Connect to reverb
  reverb_client = reverb.Client(FLAGS.reverb_address)

  puddles = arenas.get_arena(config.env.pw.arena)
  pw = puddle_world.PuddleWorld(
      puddles=puddles, goal_position=geometry.Point((1.0, 1.0)))
  dpw = pw_utils.DiscretizedPuddleWorld(pw, config.env.pw.num_bins)

  if FLAGS.eval:
    eval_worker(dpw, reverb_client, config)
  else:
    train_worker(dpw, reverb_client, config)


if __name__ == '__main__':
  app.run(main)
