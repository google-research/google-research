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
"""Hindsight Instruction Relabeling with Random Network Distillation."""
# pylint: disable=unused-variable
# pylint: disable=g-explicit-length-test
# pylint: disable=unused-import
# pylint: disable=line-too-long
from __future__ import absolute_import
from __future__ import division

import random
import time

import numpy as np

from hal.learner.hir import HIR
from hal.learner.language_utils import get_vocab_path
from hal.learner.language_utils import instruction_type
from hal.learner.language_utils import negate_unary_sentence
from hal.learner.language_utils import pad_to_max_length
from hal.learner.language_utils import paraphrase_sentence
from hal.utils.video_utils import add_text
from hal.utils.video_utils import pad_image
from hal.utils.video_utils import save_json
# from hal.utils.video_utils import save_video
import hal.utils.word_vectorization as wv


class RndHIR(HIR):
  """Learner that executes Hindsight Instruction Relabeling."""

  def reset(self, env, agent, rnd_agent, sample_new_scene=False):
    """Reset at the episode boundary.

    Args:
      env: the RL environment
      agent: the RL agent
      rnd_agent: the second agent responsible for doing RND
      sample_new_scene: sample a brand new set of objects for the scene

    Returns:
      the reset state of the environment
    """
    if self.cfg.reset_mode == 'random_action':
      for _ in range(20):
        s, _, _, _ = env.step(env.sample_random_action())
    elif self.cfg.reset_mode == 'none':
      s = env.get_obs()
    elif self.cfg.reset_mode == 'rnd':
      s = env.get_obs()
      for _ in range(20):
        s, _, _, _ = env.step(rnd_agent.step(s, env, epsilon=0.1))
    else:
      s = env.reset(sample_new_scene)
    return s

  def learn(self, env, agent, replay_buffer, rnd_model, rnd_agent):
    """Run learning for 1 cycle with consists of num_episode of episodes.

    Args:
      env: the RL environment
      agent: the RL agent
      replay_buffer: the experience replay buffer
      rnd_model: the rnd model for computing the pseudocount of a state
      rnd_agent: the second agent responsible for doing RND

    Returns:
      statistics of the training episode
    """
    average_per_ep_reward = []
    average_per_ep_achieved_n = []
    average_per_ep_relabel_n = []
    average_batch_loss = []
    average_rnd_loss = []
    average_rnd_agent_loss = []

    curr_step = agent.get_global_step()
    self.update_epsilon(curr_step)
    tic = time.time()
    for _ in range(self.cfg.num_episode):
      curr_step = agent.increase_global_step()

      sample_new_scene = random.uniform(0, 1) < self.cfg.sample_new_scene_prob
      s = self.reset(env, agent, sample_new_scene)
      episode_experience = []
      episode_reward = 0
      episode_achieved_n = 0
      episode_relabel_n = 0

      # rollout
      g_text, p = env.sample_goal()
      if env.all_goals_satisfied:
        s = self.reset(env, agent, True)
        g_text, p = env.sample_goal()
      g = np.squeeze(self.encode_fn(g_text))

      for t in range(self.cfg.max_episode_length):
        rnd_model.update_stats(s)  # Update the moving statistics of rnd model
        a = agent.step(s, g, env, self.epsilon)
        s_tp1, r, _, _ = env.step(
            a,
            record_achieved_goal=True,
            goal=p,
            atomic_goal=self.cfg.record_atomic_instruction)
        ag = env.get_achieved_goals()
        ag_text = env.get_achieved_goal_programs()
        ag_total = ag  # TODO(ydjiang): more can be stored in ag
        episode_experience.append((s, a, r, s_tp1, g, ag_total))
        episode_reward += r
        s = s_tp1
        if r > env.shape_val:
          episode_achieved_n += 1
          g_text, p = env.sample_goal()
          if env.all_goals_satisfied:
            break
          g = np.squeeze(self.encode_fn(g_text))

      average_per_ep_reward.append(episode_reward)
      average_per_ep_achieved_n.append(episode_achieved_n)

      # processing trajectory
      episode_length = len(episode_experience)
      for t in range(episode_length):
        s, a, r, s_tp1, g, ag = episode_experience[t]
        episode_relabel_n += float(len(ag) > 0)
        g_text = self.decode_fn(g)
        if self.cfg.paraphrase:
          g_text = paraphrase_sentence(
              g_text, delete_color=self.cfg.diverse_scene_content)
        g = self.encode_fn(g_text)
        replay_buffer.add((s, a, r, s_tp1, g))
        if self.cfg.relabeling:
          self.hir_relabel(episode_experience, t, replay_buffer, env)

      average_per_ep_relabel_n.append(episode_relabel_n / float(episode_length))

      if not self.is_warming_up(curr_step):
        state_trajectory = []
        for t in range(episode_length):
          state_trajectory.append(episode_experience[t][0])
        state_trajectory = np.stack(state_trajectory)
        curiosity_loss = 0
        for _ in range(self.cfg.optimization_steps):
          curiosity_loss += rnd_model.train(state_trajectory)['prediction_loss']

        average_rnd_loss.append(curiosity_loss / self.cfg.optimization_steps)

      # training
      if not self.is_warming_up(curr_step):
        batch_loss, rnd_batch_loss = 0, 0
        for _ in range(self.cfg.optimization_steps):
          experience = replay_buffer.sample(self.cfg.batchsize)
          s, a, r, s_tp1, g = [
              np.squeeze(elem, axis=1) for elem in np.split(experience, 5, 1)
          ]
          s = np.stack(s)
          s_tp1 = np.stack(s_tp1)
          g = np.array(list(g))
          if self.cfg.instruction_repr == 'language':
            g = np.array(pad_to_max_length(g, self.cfg.max_sequence_length))
          batch = {
              'obs': np.asarray(s),
              'action': np.asarray(a),
              'reward': np.asarray(r),
              'obs_next': np.asarray(s_tp1),
              'g': np.asarray(g)
          }
          loss_dict = agent.train(batch)
          batch_loss += loss_dict['loss']
          # update rnd agent
          batch['reward'] = rnd_model.compute_intrinsic_reward(
              batch['obs_next'])
          batch['done'] = np.zeros(self.cfg.batchsize)
          rnd_agent_loss = rnd_agent.train(batch)
          rnd_batch_loss += rnd_agent_loss['loss']
        average_batch_loss.append(batch_loss / self.cfg.optimization_steps)
        average_rnd_agent_loss.append(rnd_batch_loss /
                                      self.cfg.optimization_steps)

    time_per_episode = (time.time() - tic) / self.cfg.num_episode

    # Update the target network
    agent.update_target_network()
    rnd_agent.update_target_network()

    ################## Debug ##################
    sample = replay_buffer.sample(min(10000, len(replay_buffer.buffer)))
    sample_s, _, sample_r, _, _ = [
        np.squeeze(elem, axis=1) for elem in np.split(sample, 5, 1)
    ]
    sample_intrinsic_r = rnd_model.compute_intrinsic_reward(sample_s)
    print('n one:', np.sum(np.float32(sample_r == 1.0)), 'n zero',
          np.sum(np.float32(sample_r == 0.0)), 'n buff',
          len(replay_buffer.buffer))
    ################## Debug ##################
    stats = {
        'loss': np.mean(average_batch_loss) if average_batch_loss else 0,
        'reward': np.mean(average_per_ep_reward),
        'achieved_goal': np.mean(average_per_ep_achieved_n),
        'average_relabel_goal': np.mean(average_per_ep_relabel_n),
        'epsilon': self.epsilon,
        'global_step': curr_step,
        'time_per_episode': time_per_episode,
        'replay_buffer_reward_avg': np.mean(sample_r),
        'replay_buffer_reward_var': np.var(sample_r),
        'intrinsic_reward_avg': np.mean(sample_intrinsic_r),
        'intrinsic_reward_var': np.var(sample_intrinsic_r)
    }
    return stats
