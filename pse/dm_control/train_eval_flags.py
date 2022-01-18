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

"""All training and eval flags."""

import os
from absl import flags

flags.DEFINE_integer('trial_id', 0, 'The trial ID from 0 to num_trials-1.')
flags.DEFINE_integer('num_train_steps', 1000000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer(
    'eval_interval', 1000,
    'Number of train steps between evaluations. Set to 0 to skip.')
flags.DEFINE_string('method', 'drq', 'Which method to run. One of drq or qt')
flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'cartpole-swingup', 'Name of the environment.')
flags.DEFINE_string(
    'eval_log_dir', None,
    'Path to output summaries of the evaluations. If None a '
    'default directory relative to the root_dir will be used.')
flags.DEFINE_bool(
    'continuous', False,
    'If True all the evaluation will keep polling for new checkpoints.')
flags.DEFINE_multi_string('gin_files', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')
flags.DEFINE_boolean('from_pixels', True, 'Whether to train from pixel input.')
flags.DEFINE_integer(
    'policy_save_interval', 5000,
    'How often, in train_steps, the policy_save_interval trigger will save.')
flags.DEFINE_integer('checkpoint_interval', 10000,
                     'Number of train steps in between checkpoints.')
flags.DEFINE_integer('seed', None, 'Random seed.')

# FLAGS for PSEs
flags.DEFINE_bool('load_pretrained', False,
                  'If True, we initialize from the trained policy.')
flags.DEFINE_bool('image_encoder_representation', False,
                  'Use representation from image encoder similar to CURL.')

flags.DEFINE_string('pretrained_model_dir', None,
                    'Path to load pretrained model from.')
flags.DEFINE_float('contrastive_loss_weight', 1.0,
                   'Contrastive loss coefficient.')
flags.DEFINE_float('contrastive_loss_temperature', 0.1,
                   'Contrastive loss temperature.')

FLAGS = flags.FLAGS
