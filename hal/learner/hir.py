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
"""Hindsight Instruction Relabeling."""
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
from __future__ import absolute_import
from __future__ import division

import random
import time

import numpy as np

from hal.learner.language_utils import get_vocab_path
from hal.learner.language_utils import instruction_type
from hal.learner.language_utils import negate_unary_sentence
from hal.learner.language_utils import pad_to_max_length
from hal.learner.language_utils import paraphrase_sentence
from hal.utils.video_utils import add_text
from hal.utils.video_utils import pad_image
from hal.utils.video_utils import save_json
from hal.utils.video_utils import save_video
import hal.utils.word_vectorization as wv


class HIR:
  """Learner that executes Hindsight Instruction Relabeling.

  Attributes:
    cfg: configuration of this learner
    step: current training step
    epsilon: value of the epsilon for sampling random action
    vocab_list: vocabulary list used for the instruction labeler
    encode_fn: function that encodes a instruction
    decode_fn: function that converts encoded instruction back to text
    labeler: object that generates labels for transitions
  """

  def __init__(self, cfg):
    # making session
    self.cfg = cfg
    self.step = 0
    self.epsilon = 1.0

    # Vocab loading
    vocab_path = get_vocab_path(cfg)
    self.vocab_list = wv.load_vocab_list(vocab_path)
    v2i, i2v = wv.create_look_up_table(self.vocab_list)
    self.encode_fn = wv.encode_text_with_lookup_table(
        v2i, max_sequence_length=self.cfg.max_sequence_length)
    self.decode_fn = wv.decode_with_lookup_table(i2v)

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
    curiosity_loss = 0

    curr_step = agent.get_global_step()
    self.update_epsilon(curr_step)
    tic = time.time()
    time_rolling_out, time_training = 0.0, 0.0
    for _ in range(self.cfg.num_episode):
      curr_step = agent.increase_global_step()

      sample_new_scene = random.uniform(0, 1) < self.cfg.sample_new_scene_prob
      s = self.reset(env, agent, sample_new_scene)
      episode_experience = []
      episode_reward = 0
      episode_achieved_n = 0
      episode_relabel_n = 0

      # rollout
      rollout_tic = time.time()
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
      time_rolling_out += time.time() - rollout_tic

      average_per_ep_reward.append(episode_reward)
      average_per_ep_achieved_n.append(episode_achieved_n)

      # processing trajectory
      train_tic = time.time()
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

      # training
      if not self.is_warming_up(curr_step):
        batch_loss = 0
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
          if 'prediction_loss' in loss_dict:
            curiosity_loss += loss_dict['prediction_loss']
        average_batch_loss.append(batch_loss / self.cfg.optimization_steps)
      time_training += time.time()-train_tic

    time_per_episode = (time.time() - tic) / self.cfg.num_episode
    time_training_per_episode = time_training / self.cfg.num_episode
    time_rolling_out_per_episode = time_rolling_out / self.cfg.num_episode

    # Update the target network
    agent.update_target_network()
    ################## Debug ##################
    sample = replay_buffer.sample(min(10000, len(replay_buffer.buffer)))
    _, _, sample_r, _, _ = [
        np.squeeze(elem, axis=1) for elem in np.split(sample, 5, 1)
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
        'time_training_per_episode': time_training_per_episode,
        'time_rolling_out_per_episode': time_rolling_out_per_episode,
        'replay_buffer_reward_avg': np.mean(sample_r),
        'replay_buffer_reward_var': np.var(sample_r)
    }
    return stats

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
    ep_len = len(episode_experience)
    s, a, _, s_tp1, _, ag = episode_experience[current_t]
    if ag:
      for _ in range(min(self.cfg.k_immediate, len(ag) + 1)):
        ag_text_single = random.choice(ag)
        g_type = instruction_type(ag_text_single)
        if self.cfg.paraphrase and g_type != 'unary':
          ag_text_single = paraphrase_sentence(
              ag_text_single, delete_color=self.cfg.diverse_scene_content)
        replay_buffer.add(
            (s, a, env.reward_scale, s_tp1, self.encode_fn(ag_text_single)))
        if g_type == 'unary' and self.cfg.negate_unary:
          negative_ag = negate_unary_sentence(ag_text_single)
          if negative_ag:
            replay_buffer.add((s, a, 0., s_tp1, self.encode_fn(negative_ag)))
    goal_count, repeat = 0, 0
    while goal_count < self.cfg.future_k and repeat < (ep_len - current_t) * 2:
      repeat += 1
      future = np.random.randint(current_t, ep_len)
      _, _, _, _, _, ag_future = episode_experience[future]
      if not ag_future:
        continue
      random.shuffle(ag_future)
      for single_g in ag_future:
        if instruction_type(single_g) != 'unary':
          discount = self.cfg.discount**(future - current_t)
          if self.cfg.paraphrase:
            single_g = paraphrase_sentence(
                single_g, delete_color=self.cfg.diverse_scene_content)
          replay_buffer.add((s, a, discount * env.reward_scale, s_tp1,
                             self.encode_fn(single_g)))
          goal_count += 1
          break

  def update_epsilon(self, step):
    new_epsilon = self.cfg.epsilon_decay**(step // self.cfg.num_episode)
    self.epsilon = max(new_epsilon, self.cfg.min_epsilon)

  def is_warming_up(self, step):
    return step <= self.cfg.collect_cycle * self.cfg.num_episode

  def rollout(self,
              env,
              agent,
              directory,
              record_video=False,
              timeout=8,
              num_episode=10,
              record_trajectory=False):
    """Rollout and save.

    Args:
      env: the RL environment
      agent: the RL agent
      directory: directory where the output of the rollout is saved
      record_video: record the video
      timeout: timeout step if the agent is stuck
      num_episode: number of rollout episode
      record_trajectory: record the ground truth trajectory

    Returns:
      percentage of success during this rollout
    """
    print('#######################################')
    print('Rolling out...')
    print('#######################################')
    all_frames = []
    ep_observation, ep_action, ep_agn = [], [], []
    black_frame = pad_image(env.render(mode='rgb_array')) * 0.0
    goal_sampled = 0
    timeout_count, success = 0, 0
    for ep in range(num_episode):
      s = self.reset(env, agent, self.cfg.diverse_scene_content)
      all_frames += [black_frame] * 10
      g_text, p = env.sample_goal()
      if env.all_goals_satisfied:
        s = self.reset(env, agent, True)
        g, p = env.sample_goal()
      goal_sampled += 1
      g = np.squeeze(self.encode_fn(g_text))
      current_goal_repetition = 0
      for t in range(self.cfg.max_episode_length):
        prob = self.epsilon if record_trajectory else 0.0
        action = agent.step(s, g, env, explore_prob=prob)
        s_tp1, r, _, _ = env.step(
            action,
            record_achieved_goal=True,
            goal=p,
            atomic_goal=self.cfg.record_atomic_instruction)
        ag = env.get_achieved_goals()
        s = s_tp1
        all_frames.append(
            add_text(pad_image(env.render(mode='rgb_array')), g_text))
        current_goal_repetition += 1

        if record_trajectory:
          ep_observation.append(env.get_direct_obs().tolist())
          ep_action.append(action)
          ep_agn.append(len(ag))

        sample_new_goal = False
        if r > env.shape_val:
          for _ in range(5):
            all_frames.append(
                add_text(
                    pad_image(env.render(mode='rgb_array')),
                    g_text,
                    color='green'))
          success += 1
          sample_new_goal = True

        if current_goal_repetition >= timeout:
          all_frames.append(
              add_text(pad_image(env.render(mode='rgb_array')), 'time out :('))
          timeout_count += 1
          sample_new_goal = True

        if sample_new_goal:
          g, p = env.sample_goal()
          if env.all_goals_satisfied:
            break
          g_text = g
          g = np.squeeze(self.encode_fn(g))
          current_goal_repetition = 0
          goal_sampled += 1

    print('Rollout finished')
    print('{} instrutctions tried given'.format(goal_sampled))
    print('{} instructions timed out'.format(timeout_count))
    if record_video:
      save_video(np.uint8(all_frames), directory, fps=5)
      print('Video saved...')
    if record_trajectory:
      print('Recording trajectory...')
      datum = {
          'obs': ep_observation,
          'action': ep_action,
          'achieved goal': ep_agn,
      }
      save_json(datum, directory[:-4] + '_trajectory.json')
    return success / float(num_episode)
