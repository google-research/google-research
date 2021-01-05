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

r"""Compute bisimulation metric on grid world.

Sample run:
  ```
python -m bisimulation_aaai2020/grid_world/compute_metric \
  --base_dir=/tmp/grid_world \
  --grid_file=bisimulation_aaai2020/grid_world/configs/mirrored_rooms.grid \
  --gin_files=bisimulation_aaai2020/grid_world/configs/mirrored_rooms.gin \
  --nosample_distance_pairs
  ```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
import gin.tf

from bisimulation_aaai2020.grid_world import grid_world

flags.DEFINE_string('grid_file', None,
                    'Path to file defining grid world MDP.')
flags.DEFINE_string('base_dir', None, 'Base directory to store stats.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')
flags.DEFINE_bool('exact_metric', True,
                  'Whether to compute the metric using the exact method.')
flags.DEFINE_bool('sampled_metric', True,
                  'Whether to compute the metric using sampling.')
flags.DEFINE_bool('learn_metric', True,
                  'Whether to compute the metric using learning.')
flags.DEFINE_bool('sample_distance_pairs', True,
                  'Whether to aggregate states (needs a learned metric.')
flags.DEFINE_integer('num_samples_per_cell', 100,
                     'Number of samples per cell when aggregating.')
flags.DEFINE_bool('verbose', False, 'Whether to print verbose messages.')

FLAGS = flags.FLAGS




def main(_):
  flags.mark_flag_as_required('base_dir')
  gin.parse_config_files_and_bindings(FLAGS.gin_files,
                                      bindings=FLAGS.gin_bindings,
                                      skip_unknown=False)
  grid = grid_world.GridWorld(FLAGS.base_dir, grid_file=FLAGS.grid_file)
  if FLAGS.exact_metric:
    grid.compute_exact_metric(verbose=FLAGS.verbose)
  if FLAGS.sampled_metric:
    grid.compute_sampled_metric(verbose=FLAGS.verbose)
  if FLAGS.learn_metric:
    grid.learn_metric(verbose=FLAGS.verbose)
    grid.save_statistics()
  if FLAGS.sample_distance_pairs:
    grid.sample_distance_pairs(num_samples_per_cell=FLAGS.num_samples_per_cell,
                               verbose=FLAGS.verbose)

if __name__ == '__main__':
  app.run(main)
