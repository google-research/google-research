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

# Lint as: python3
# pylint: disable=logging-format-interpolation
# pylint: disable=g-complex-comprehension
r"""SEED actor."""

import collections
import os
import random

from absl import flags
from absl import logging
import numpy as np
from seed_rl import grpc
from seed_rl.common import common_flags  # pylint: disable=unused-import
from seed_rl.common import profiling
from seed_rl.common import utils
import tensorflow as tf

from muzero import core
from muzero import utils as mzutils

FLAGS = flags.FLAGS

TASK = flags.DEFINE_integer('task', 0, 'Task id.')
USE_SOFTMAX_FOR_TARGET = flags.DEFINE_integer(
    'use_softmax_for_target', 0,
    'If True (1), use a softmax for the child_visit count distribution that '
    'is used as a target for the policy.')
NUM_TEST_ACTORS = flags.DEFINE_integer(
    'num_test_actors', 2, 'Number of actors that are used for testing.')
NUM_ACTORS_WITH_SUMMARIES = flags.DEFINE_integer(
    'num_actors_with_summaries', 1,
    'Number of actors that will log debug/profiling TF '
    'summaries.')
ACTOR_LOG_FREQUENCY = flags.DEFINE_integer('actor_log_frequency', 10,
                                           'in number of training steps')
MCTS_VIS_FILE = flags.DEFINE_string(
    'mcts_vis_file', None, 'File in which to log the mcts visualizations.')
FLAG_FILE = flags.DEFINE_string('flag_file', None,
                                'File in which to log the parameters.')
ENABLE_ACTOR_LOGGING = flags.DEFINE_boolean('enable_actor_logging', True,
                                            'Verbose logging for the actor.')
MAX_NUM_ACTION_EXPANSION = flags.DEFINE_integer(
    'max_num_action_expansion', 0,
    'Maximum number of new nodes for a node expansion. 0 for no limit. '
    'This is important for the full vocabulary.')
ACTOR_ENQUEUE_EVERY = flags.DEFINE_integer(
    'actor_enqueue_every', 0,
    'After how many steps the actor enqueues samples. 0 for at episode end.')
ACTOR_SKIP = flags.DEFINE_integer('actor_skip', 0,
                                  'How many target samples the actor skips.')


def is_training_actor():
  return TASK.value >= NUM_TEST_ACTORS.value


def are_summaries_enabled():
  return is_training_actor(
  ) and TASK.value < NUM_TEST_ACTORS.value + NUM_ACTORS_WITH_SUMMARIES.value


def actor_loop(create_env_fn,
               mzconfig,
               share_of_supervised_episodes_fn=lambda _: 0.):
  """Main actor loop.

  Args:
    create_env_fn: Callable (taking the task ID as argument) that must return a
      newly created environment.
    mzconfig: MuZeroConfig instance.
    share_of_supervised_episodes_fn: Function that specifies the share of
      episodes that should be supervised based on the learner iteration.
  """

  logging.info('Starting actor loop')

  actor_log_dir = os.path.join(FLAGS.logdir, 'actor_{}'.format(TASK.value))
  if are_summaries_enabled():
    summary_writer = tf.summary.create_file_writer(
        actor_log_dir, flush_millis=20000, max_queue=1000)
    timer_cls = profiling.ExportingTimer
    if FLAG_FILE.value:
      mzutils.write_flags(FLAGS.__flags, FLAG_FILE.value)  # pylint: disable=protected-access
  else:
    summary_writer = tf.summary.create_noop_writer()
    timer_cls = utils.nullcontext

  batch_queue = collections.deque()

  actor_step = tf.Variable(0, dtype=tf.int64)
  num_episodes = tf.Variable(0, dtype=tf.int64)

  # We use the checkpoint to keep track of the actor_step and num_episodes.
  actor_checkpoint = tf.train.Checkpoint(
      actor_step=actor_step, num_episodes=num_episodes)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=actor_checkpoint, directory=actor_log_dir, max_to_keep=1)
  if ckpt_manager.latest_checkpoint:
    logging.info('Restoring actor checkpoint: %s',
                 ckpt_manager.latest_checkpoint)
    actor_checkpoint.restore(ckpt_manager.latest_checkpoint).assert_consumed()

  reward_agg, length_agg = profiling.Aggregator(), profiling.Aggregator()
  with summary_writer.as_default():
    tf.summary.experimental.set_step(actor_step)
    while True:
      try:
        # Client to communicate with the learner.
        client = grpc.Client(FLAGS.server_address)

        def _create_training_samples(episode, start_idx=0):
          start_idx += random.choice(range(ACTOR_SKIP.value + 1))
          for i in range(start_idx, len(episode.history), ACTOR_SKIP.value + 1):
            target = episode.make_target(
                state_index=i,
                num_unroll_steps=mzconfig.num_unroll_steps,
                td_steps=mzconfig.td_steps,
                rewards=episode.rewards,
                policy_distributions=episode.child_visits,
                discount=episode.discount,
                value_approximations=episode.root_values)
            priority = np.float32(1e-2)  # preventing all zero priorities
            if len(episode) > 0:  # pylint: disable=g-explicit-length-test
              last_value_idx = min(len(episode) - 1 - i, len(target.value) - 1)
              priority = np.maximum(
                  priority,
                  np.float32(
                      np.abs(episode.root_values[i + last_value_idx] -
                             target.value[last_value_idx])))

            # This will be batched and given to add_to_replay_buffer on the
            # learner.
            sample = (
                priority,
                episode.make_image(i),
                tf.stack(
                    episode.history_range(i, i + mzconfig.num_unroll_steps)),
            ) + tuple(map(lambda x: tf.cast(tf.stack(x), tf.float32), target))
            batch_queue.append(sample)
          if ENABLE_ACTOR_LOGGING.value:
            logging.info(
                'Added %d samples to the batch_queue. Size: %d of needed %d',
                len(episode.history) - start_idx, len(batch_queue),
                mzconfig.train_batch_size)

        def _add_queue_to_replay_buffer():
          with timer_cls('actor/elapsed_add_to_buffer_s',
                         10 * ACTOR_LOG_FREQUENCY.value):
            while len(batch_queue) >= mzconfig.train_batch_size:
              batch = [
                  batch_queue.popleft()
                  for _ in range(mzconfig.train_batch_size)
              ]
              flat_batch = [tf.nest.flatten(b) for b in batch]
              stacked_batch = list(map(tf.stack, zip(*flat_batch)))
              structured_batch = tf.nest.pack_sequence_as(
                  batch[0], stacked_batch)
              client.add_to_replay_buffer(*structured_batch)
              if ENABLE_ACTOR_LOGGING.value:
                logging.info('Added batch of size %d into replay_buffer.',
                             len(batch))

        env = create_env_fn(TASK.value, training=is_training_actor())

        def recurrent_inference_fn(*args, **kwargs):
          with timer_cls('actor/elapsed_recurrent_inference_s',
                         100 * ACTOR_LOG_FREQUENCY.value):
            output = client.recurrent_inference(*args, **kwargs)
            output = tf.nest.map_structure(lambda t: t.numpy(), output)
          return output

        def get_legal_actions_fn(episode):

          def legal_actions_fn(*args, **kwargs):
            with timer_cls('actor/elapsed_get_legal_actions_s',
                           100 * ACTOR_LOG_FREQUENCY.value):
              output = episode.legal_actions(*args, **kwargs)
            return output

          return legal_actions_fn

        while True:
          episode = mzconfig.new_episode(env)
          is_supervised_episode = is_training_actor() and \
              random.random() < share_of_supervised_episodes_fn(
                  client.learning_iteration().numpy())

          if is_supervised_episode:
            if ENABLE_ACTOR_LOGGING.value:
              logging.info('Supervised Episode.')
            try:
              with timer_cls('actor/elapsed_load_supervised_episode_s',
                             ACTOR_LOG_FREQUENCY.value):
                episode_example = env.load_supervised_episode()
              with timer_cls('actor/elapsed_run_supervised_episode_s',
                             ACTOR_LOG_FREQUENCY.value):
                targets, samples = env.run_supervised_episode(episode_example)
              episode.rewards = samples['reward']
              episode.history = samples['to_predict']
              for target in targets:
                batch_queue.append(target)
            except core.RLEnvironmentError as e:
              logging.warning('Environment not ready %s', str(e))
              # restart episode
              continue
            except core.BadSupervisedEpisodeError as e:
              logging.warning('Abort supervised episode: %s', str(e))
              # restart episode
              continue
          else:
            if ENABLE_ACTOR_LOGGING.value:
              logging.info('RL Episode.')
            try:
              last_enqueued_idx = 0
              legal_actions_fn = get_legal_actions_fn(episode)
            except core.RLEnvironmentError as e:
              logging.warning('Environment not ready: %s', str(e))
              # restart episode
              continue
            except core.SkipEpisode as e:
              logging.warning('Episode is skipped due to: %s', str(e))
              # restart episode
              continue
            while (not episode.terminal() and
                   len(episode.history) < mzconfig.max_moves):
              # This loop is the agent playing the episode.
              current_observation = episode.make_image(-1)

              # Map the observation to hidden space.
              with timer_cls('actor/elapsed_initial_inference_s',
                             10 * ACTOR_LOG_FREQUENCY.value):
                initial_inference_output = client.initial_inference(
                    current_observation)
                initial_inference_output = tf.nest.map_structure(
                    lambda t: t.numpy(), initial_inference_output)

              # Run MCTS using recurrent_inference_fn.
              with timer_cls('actor/elapsed_mcts_s',
                             10 * ACTOR_LOG_FREQUENCY.value):
                legal_actions = legal_actions_fn()
                root = core.prepare_root_node(mzconfig, legal_actions,
                                              initial_inference_output)
                with timer_cls('actor/elapsed_run_mcts_s',
                               10 * ACTOR_LOG_FREQUENCY.value):
                  core.run_mcts(mzconfig, root, episode.action_history(),
                                legal_actions_fn, recurrent_inference_fn,
                                episode.visualize_mcts)
                action = core.select_action(
                    mzconfig,
                    len(episode.history),
                    root,
                    train_step=actor_step.numpy(),
                    use_softmax=mzconfig.use_softmax_for_action_selection,
                    is_training=is_training_actor())

              try:
                # Perform chosen action.
                with timer_cls('actor/elapsed_env_step_s',
                               10 * ACTOR_LOG_FREQUENCY.value):
                  training_steps = client.learning_iteration().numpy()
                  episode.apply(action=action, training_steps=training_steps)
              except core.RLEnvironmentError as env_error:
                logging.warning('Environment failed: %s', str(env_error))
                episode.failed = True
                # terminate episode
                break

              episode.store_search_statistics(
                  root, use_softmax=(USE_SOFTMAX_FOR_TARGET.value == 1))
              actor_step.assign_add(delta=1)
              if is_training_actor() and ACTOR_ENQUEUE_EVERY.value > 0 and (
                  len(episode.history) -
                  last_enqueued_idx) >= ACTOR_ENQUEUE_EVERY.value:
                _create_training_samples(episode, start_idx=last_enqueued_idx)
                last_enqueued_idx = len(episode.history)
                _add_queue_to_replay_buffer()

            if episode.failed:
              # restart episode
              logging.warning('Episode failed, restarting.')
              continue
            # Post-episode stats
            num_episodes.assign_add(delta=1)
            reward_agg.add(episode.total_reward())
            length_agg.add(len(episode))
            if ENABLE_ACTOR_LOGGING.value:
              logging.info(
                  'Episode done. Length: %d, '
                  'Total Reward: %d, Min Reward: %d, Max Reward: %d',
                  len(episode), episode.total_reward(), min(episode.rewards),
                  max(episode.rewards))
            if reward_agg.count % ACTOR_LOG_FREQUENCY.value == 0:
              tf.summary.experimental.set_step(actor_step)
              tf.summary.scalar('actor/total_reward', reward_agg.average())
              tf.summary.scalar('actor/episode_length', length_agg.average())
              tf.summary.scalar('actor/num_episodes', num_episodes)
              tf.summary.scalar('actor/step', actor_step)
              tf.summary.scalar(
                  'actor/share_of_supervised_episodes',
                  share_of_supervised_episodes_fn(
                      client.learning_iteration().numpy()))
              if episode.mcts_visualizations:
                tf.summary.text('actor/mcts_vis',
                                '\n\n'.join(episode.mcts_visualizations))
                if are_summaries_enabled() and MCTS_VIS_FILE.value is not None:
                  # write it also into a txt file
                  with tf.io.gfile.GFile(MCTS_VIS_FILE.value, 'a') as f:
                    f.write('Step {}\n{}\n\n\n\n'.format(
                        actor_step, '\n\n'.join(episode.mcts_visualizations)))

              special_stats = episode.special_statistics()
              for stat_name, stat_value in special_stats.items():
                if isinstance(stat_value, float) or isinstance(stat_value, int):
                  tf.summary.scalar('actor/{}'.format(stat_name), stat_value)
                elif isinstance(stat_value, str):
                  tf.summary.text('actor/{}'.format(stat_name), stat_value)
                else:
                  logging.warning(
                      'Special statistic %s could not be tracked. '
                      'Type %s is not supported.', stat_name, type(stat_value))

              ckpt_manager.save()
              reward_agg.reset()
              length_agg.reset()

            if is_training_actor():
              # Create samples for training.
              _create_training_samples(episode, start_idx=last_enqueued_idx)

          # Send training samples to the learner after the episode is finished
          if is_training_actor():
            _add_queue_to_replay_buffer()

          summary_name = 'train' if is_training_actor() else 'test'
          if is_supervised_episode:
            summary_name += ' (supervised)'
          with timer_cls('actor/elapsed_add_to_reward_s',
                         10 * ACTOR_LOG_FREQUENCY.value):
            # This is just for statistics.
            client.add_to_reward_queue(summary_name,
                                       np.float32(episode.total_reward()),
                                       np.int64(len(episode)),
                                       *episode.special_statistics_learner())
          del episode

      except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
        logging.exception(e)
        env.close()
