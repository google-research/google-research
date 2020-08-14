# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
from __future__ import absolute_import
from __future__ import division

import random
import time

import numpy as np
import tensorflow as tf

from hal.learner.hir import HIR
from hal.learner.language_utils import instruction_type
from hal.learner.language_utils import pad_to_max_length
from hal.learner.language_utils import paraphrase_sentence


class MaxentIrlHIR(HIR):
  """Learner that executes Hindsight Instruction Relabeling."""

  def reset(self, env, agent, sample_new_scene=False, **kwargs):
    """Reset at the episode boundary.

    Args:
      env: the RL environment
      agent: the RL agent
      sample_new_scene: sample a brand new set of objects for the scene
      **kwargs: other potential arguments

    Returns:
      the reset state of the environment
    """
    if self.cfg.reset_mode == 'random_action':
      for _ in range(20):
        s, _, _, _ = env.step(env.sample_random_action())
    elif self.cfg.reset_mode == 'none':
      s = env.get_obs()
    else:
      s = env.reset(sample_new_scene)
    return s

  def learn(self, env, agent, replay_buffer, **kwargs):
    """Run learning for 1 cycle with consists of num_episode of episodes.

    Args:
      env: the RL environment
      agent: the RL agent
      replay_buffer: the experience replay buffer
      **kwargs: other potential arguments

    Returns:
      statistics of the training episode
    """
    average_per_ep_reward = []
    average_per_ep_achieved_n = []
    average_per_ep_relabel_n = []
    average_batch_loss = []

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
        replay_buffer.add((s, a, r, s_tp1, g, ag))
        if self.cfg.relabeling:
          self.hir_relabel(episode_experience, t, replay_buffer, env)

      average_per_ep_relabel_n.append(episode_relabel_n / float(episode_length))

      # training
      if not self.is_warming_up(curr_step):
        batch_loss = 0
        for _ in range(self.cfg.optimization_steps):
          experience = replay_buffer.sample(self.cfg.batchsize)
          s, a, r, s_tp1, g = self.max_ent_relabel(experience, agent, env)
          batch = {
              'obs': s,
              'action': a,
              'reward': r,
              'obs_next': s_tp1,
              'g': g
          }
          loss_dict = agent.train(batch)
          batch_loss += loss_dict['loss']
        average_batch_loss.append(batch_loss / self.cfg.optimization_steps)

    time_per_episode = (time.time() - tic) / self.cfg.num_episode

    # Update the target network
    agent.update_target_network()

    ################## Debug ##################
    sample = replay_buffer.sample(min(10000, len(replay_buffer.buffer)))
    sample_s, _, sample_r, _, _, _ = [
        np.squeeze(elem, axis=1) for elem in np.split(sample, 6, 1)
    ]
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
    }
    return stats

  def max_ent_relabel(self, experience, agent, env):
    """Perform maximum entropy relabeling.

    Args:
      experience: experience to be re-labeled
      agent: the RL agent
      env: the RL environment

    Returns:
      relabeled experience
    """
    relabel_proportion = self.cfg.relabel_proportion
    exp_o = experience[int(len(experience) * relabel_proportion):]
    exp_n = experience[:int(len(experience) * relabel_proportion)]
    s_o, a_o, r_o, s_tp1_o, g_o, ag_o = [
        np.squeeze(elem, axis=1) for elem in np.split(exp_o, 6, 1)
    ]
    s_n, a_n, r_n, s_tp1_n, g_n, ag_n = [
        np.squeeze(elem, axis=1) for elem in np.split(exp_n, 6, 1)
    ]
    chosen_q_idx = np.random.choice(
        np.arange(len(env.all_questions)), self.cfg.irl_sample_goal_n)
    g_candidate = np.array(env.all_questions)[chosen_q_idx]
    g_candidate = [q for q, p in g_candidate]
    if self.cfg.instruction_repr == 'language':
      g_o = np.array(pad_to_max_length(g_o, self.cfg.max_sequence_length))
      if self.cfg.paraphrase:
        for i, g_text in enumerate(g_candidate):
          g_candidate[i] = paraphrase_sentence(
              g_text, delete_color=self.cfg.diverse_scene_content)
      g_candidate = [self.encode_fn(g) for g in g_candidate]
      g_candidate = np.array(
          pad_to_max_length(g_candidate, self.cfg.max_sequence_length))
    soft_q = agent.compute_q_over_all_g(s_n, a_n, g_candidate)
    normalized_soft_q = tf.nn.softmax(soft_q, axis=-1).numpy()
    chosen_g = []
    for sq in normalized_soft_q:
      chosen_g.append(np.random.choice(np.arange(sq.shape[0]), 1, p=sq)[0])
    g_n = g_candidate[chosen_g]
    s = np.concatenate([np.stack(s_o), np.stack(s_n)], axis=0)
    a = np.concatenate([a_o, a_n], axis=0)
    r = np.concatenate([r_o, r_n], axis=0)
    s_tp1 = np.concatenate([np.stack(s_tp1_o), np.stack(s_tp1_n)], axis=0)
    g = np.concatenate([g_o, g_n])
    if self.cfg.instruction_repr == 'language':
      g = np.array(pad_to_max_length(g, self.cfg.max_sequence_length))
    return s, a, r, s_tp1, g

  def hir_relabel(self, episode_experience, current_t, replay_buffer, env):
    """Relabeling trajectories.

    Args:
      episode_experience: the RL environment
      current_t: time time step at which the experience is relabeled
      replay_buffer:  the experience replay buffer
      env: the RL environment

    Returns:
      the reset state of the environment
    """
    s, a, _, s_tp1, g, ag = episode_experience[current_t]
    if not ag:
      return
    for _ in range(min(self.cfg.k_immediate, len(ag) + 1)):
      ag_text_single = random.choice(ag)
      g_type = instruction_type(ag_text_single)
      if self.cfg.paraphrase and g_type != 'unary':
        ag_text_single = paraphrase_sentence(
            ag_text_single, delete_color=self.cfg.diverse_scene_content)
      replay_buffer.add(
          (s, a, env.reward_scale, s_tp1, self.encode_fn(ag_text_single), ag))
