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

"""Some common flags."""

from absl import flags

# Common flags

flags.DEFINE_multi_string('gin_configs', [], 'A path to a config file.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')

flags.DEFINE_bool('debug_nans', False, 'When you want to find Nans problems.')
flags.DEFINE_bool('spoof_multi_device', False, 'Used for local debugging.')
flags.DEFINE_bool('disable_jit', False, 'Whether to disable jit and pmap. '
                  'Useful for debugging.')

flags.DEFINE_integer('seed', 1, 'Experiment seed.')

flags.DEFINE_integer('num_steps', 1_000_000, 'Number of training steps.')
flags.DEFINE_integer('eval_every_steps', 10_000,
                     'How many learner steps between evaluations.')
flags.DEFINE_integer('episodes_per_eval', 10,
                     'How many episodes to run per eval run.')

flags.DEFINE_string('algorithm', 'msg', 'The algorithm to use.')
flags.DEFINE_string('task_class', 'd4rl', 'The task class.')
flags.DEFINE_string('task_name', 'halfcheetah-medium-replay-v0', 'The task name.')
flags.DEFINE_bool('single_precision_env', False,
                  'Whether to make the env single precision.')

flags.DEFINE_integer('batch_size', 256, 'Number of training steps.')

flags.DEFINE_string('root_dir', '/tmp/test_msg', 'Experiment directory.')

flags.DEFINE_bool('create_saved_model_actor', False,
                  'Whether to create a saved model for the actor.')

flags.DEFINE_bool('eval_with_q_filter', False,
                  'If at evaluation time we want to also evaluate policies'
                  'by filtering their actions using their Q-value estimates.'
                  'Not supported by all algorithms.')

