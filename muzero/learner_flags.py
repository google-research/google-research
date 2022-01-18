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

# python3
"""Learner command line flags."""

from absl import flags
from seed_rl.common import common_flags  # pylint: disable=unused-import

from muzero import learner_config

SAVE_CHECKPOINT_SECS = flags.DEFINE_integer(
    'save_checkpoint_secs', 1800, 'Checkpoint save period in seconds.')
TOTAL_ITERATIONS = flags.DEFINE_integer('total_iterations', int(1e6),
                                        'Total iterations to train for.')
BATCH_SIZE = flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
REPLAY_QUEUE_BLOCK = flags.DEFINE_integer(
    'replay_queue_block', 0, 'Whether actors block when enqueueing.')
RECURRENT_INFERENCE_BATCH_SIZE = flags.DEFINE_integer(
    'recurrent_inference_batch_size', 32,
    'Batch size for the recurrent inference.')
INITIAL_INFERENCE_BATCH_SIZE = flags.DEFINE_integer(
    'initial_inference_batch_size', 4, 'Batch size for initial inference.')
NUM_TRAINING_TPUS = flags.DEFINE_integer('num_training_tpus', 1,
                                         'Number of TPUs for training.')
INIT_CHECKPOINT = flags.DEFINE_string(
    'init_checkpoint', None,
    'Path to the checkpoint used to initialize the agent.')
NUM_ACTORS = flags.DEFINE_integer('num_actors', 10, 'Number of actors.')

REPLAY_BUFFER_SIZE = flags.DEFINE_integer('replay_buffer_size', 1000,
                                          'Size of the replay buffer.')
REPLAY_QUEUE_SIZE = flags.DEFINE_integer('replay_queue_size', 100,
                                         'Size of the replay queue.')
REPLAY_BUFFER_UPDATE_PRIORITY_AFTER_SAMPLING_VALUE = flags.DEFINE_float(
    'replay_buffer_update_priority_after_sampling_value', 1e-6,
    'After sampling an episode from the replay buffer, the corresponding '
    'priority is set to this value. For a value < 1, no priority update will '
    'be done.')
FLUSH_LEARNER_LOG_EVERY_N_S = flags.DEFINE_integer(
    'flush_learner_log_every_n_s', 60,
    'Size of the replay buffer (in number of batches stored).')
ENABLE_LEARNER_LOGGING = flags.DEFINE_integer(
    'enable_learner_logging', 1,
    'If true (1), logs are written to tensorboard.')
flags.DEFINE_integer('log_frequency', 100, 'in number of training steps')

IMPORTANCE_SAMPLING_EXPONENT = flags.DEFINE_float(
    'importance_sampling_exponent', 0.0,
    'Exponent used when computing the importance sampling '
    'correction. 0 means no importance sampling correction. '
    '1 means full importance sampling correction.')
PRIORITY_SAMPLING_EXPONENT = flags.DEFINE_float(
    'priority_sampling_exponent', 0.0,
    'For sampling from priority queue. 0 for uniform. The higher this value '
    'the more likely it is to sample an instance for which the model predicts '
    'a wrong value.')
LEARNER_SKIP = flags.DEFINE_integer('learner_skip', 0,
                                    'How many batches the learner skips.')
EXPORT_AGENT = flags.DEFINE_integer('export_agent', 0,
                                    'Save the agent in ExportAgent format.')

WEIGHT_DECAY = flags.DEFINE_float('weight_decay', 1e-5, 'l2 penalty')
POLICY_LOSS_SCALING = flags.DEFINE_float('policy_loss_scaling', 1.0,
                                         'Scaling for the policy loss term.')
REWARD_LOSS_SCALING = flags.DEFINE_float('reward_loss_scaling', 1.0,
                                         'Scaling for the policy loss term.')
POLICY_LOSS_ENTROPY_REGULARIZER = flags.DEFINE_float(
    'policy_loss_entropy_regularizer', 0.0,
    'Entropy loss for the policy loss term.')
GRADIENT_NORM_CLIP = flags.DEFINE_float('gradient_norm_clip', 0.0,
                                        'Gradient norm clip (0 for no clip).')

DEBUG = flags.DEFINE_boolean('debug', False, '')

FLAGS = flags.FLAGS


def learner_config_from_flags():
  """Returns the learner config based on command line flags."""
  return learner_config.LearnerConfig(
      save_checkpoint_secs=FLAGS.save_checkpoint_secs,
      total_iterations=FLAGS.total_iterations,
      batch_size=FLAGS.batch_size,
      replay_queue_block=FLAGS.replay_queue_block,
      recurrent_inference_batch_size=FLAGS.recurrent_inference_batch_size,
      initial_inference_batch_size=FLAGS.initial_inference_batch_size,
      num_training_tpus=FLAGS.num_training_tpus,
      init_checkpoint=FLAGS.init_checkpoint,
      replay_buffer_size=FLAGS.replay_buffer_size,
      replay_queue_size=FLAGS.replay_queue_size,
      replay_buffer_update_priority_after_sampling_value=(
          FLAGS.replay_buffer_update_priority_after_sampling_value),
      flush_learner_log_every_n_s=FLAGS.flush_learner_log_every_n_s,
      enable_learner_logging=FLAGS.enable_learner_logging,
      log_frequency=FLAGS.log_frequency,
      importance_sampling_exponent=FLAGS.importance_sampling_exponent,
      priority_sampling_exponent=FLAGS.priority_sampling_exponent,
      learner_skip=FLAGS.learner_skip,
      export_agent=FLAGS.export_agent,
      weight_decay=FLAGS.weight_decay,
      policy_loss_scaling=FLAGS.policy_loss_scaling,
      reward_loss_scaling=FLAGS.reward_loss_scaling,
      policy_loss_entropy_regularizer=FLAGS.policy_loss_entropy_regularizer,
      gradient_norm_clip=FLAGS.gradient_norm_clip,
      debug=FLAGS.debug,
      logdir=FLAGS.logdir,
      server_address=FLAGS.server_address)
