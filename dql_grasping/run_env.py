# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Library function for stepping/evaluating a policy in a Gym environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import os
import numpy as np
import PIL.Image as Image
import six
import tensorflow as tf
import gin.tf


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


@gin.configurable(blacklist=['task', 'num_episodes', 'global_step', 'tag'])
def run_env(env,
            policy=None,
            explore_schedule=None,
            episode_to_transitions_fn=None,
            replay_writer=None,
            root_dir=None,
            task=0,
            global_step=0,
            num_episodes=100,
            tag='collect'):
  """Runs agent+env loop num_episodes times and log performance + collect data.

  Interpolates between an exploration policy and greedy policy according to a
  explore_schedule. Run this function separately for collect/eval.

  Args:
    env: Gym environment.
    policy: Policy to collect/evaluate.
    explore_schedule: Exploration schedule that defines a `value(t)` function
      to compute the probability of exploration as a function of global step t.
    episode_to_transitions_fn: Function that converts episode data to transition
      protobufs (e.g. TFExamples).
    replay_writer: Instance of a replay writer that writes a list of transition
      protos to disk (optional).
    root_dir: Root directory of the experiment summaries and data collect. If
      replay_writer is specified, data is written to the `policy_*` subdirs.
      Setting root_dir=None results in neither summaries or transitions being
      saved to disk.
    task: Task number for replica trials for a given experiment.
    global_step: Training step corresponding to policy checkpoint.
    num_episodes: Number of episodes to run.
    tag: String prefix for evaluation summaries and collect data.
  """

  episode_rewards = []
  episode_q_values = collections.defaultdict(list)

  if root_dir and replay_writer:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    record_prefix = os.path.join(root_dir, 'policy_%s' % tag,
                                 'gs%d_t%d_%s' % (global_step, task, timestamp))
  if root_dir:
    summary_dir = os.path.join(root_dir, 'live_eval_%d' % task)
    summary_writer = tf.summary.FileWriter(summary_dir)

  if replay_writer:
    replay_writer.open(record_prefix)

  for ep in range(num_episodes):
    done, env_step, episode_reward, episode_data = (False, 0, 0.0, [])
    policy.reset()
    obs = env.reset()
    if explore_schedule:
      explore_prob = explore_schedule.value(global_step)
    else:
      explore_prob = 0
    while not done:
      action, policy_debug = policy.sample_action(obs, explore_prob)
      if policy_debug and 'q' in policy_debug:
        episode_q_values[env_step].append(policy_debug['q'])
      new_obs, rew, done, env_debug = env.step(action)
      env_step += 1
      episode_reward += rew

      episode_data.append((obs, action, rew, new_obs, done, env_debug))
      obs = new_obs
      if done:
        tf.logging.info('Episode %d reward: %f' % (ep, episode_reward))
        episode_rewards.append(episode_reward)
        if replay_writer:
          transitions = episode_to_transitions_fn(episode_data)
          replay_writer.write(transitions)
    if episode_rewards and len(episode_rewards) % 10 == 0:
      tf.logging.info('Average %d collect episodes reward: %f' %
                      (len(episode_rewards), np.mean(episode_rewards)))

  if replay_writer:
    replay_writer.close()

  if root_dir:
    summary_values = [
        tf.Summary.Value(
            tag='%s/episode_reward' % tag,
            simple_value=np.mean(episode_rewards)),
        # TODO(konstantinos): Move this out of task-agnostic code
        tf.Summary.Value(
            tag='%s/input_image' % tag,
            image=tf.Summary.Image(
                encoded_image_string=encode_image_array_as_png_str(obs)))
    ]
    for step, q_values in episode_q_values.items():
      summary_values.append(
          tf.Summary.Value(
              tag='%s/Q/%d' % (tag, step), simple_value=np.mean(q_values)))
    summary = tf.Summary(value=summary_values)
    summary_writer.add_summary(summary, global_step)
