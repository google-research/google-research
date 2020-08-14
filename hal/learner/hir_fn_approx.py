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

from hal.labeler.labelers import Labeler
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

_SYNONYM_TABLES = [{
    'on the left side': ['on the lefthand side', 'in the lefthand side'],
    'on the right side': ['on the righthand side', 'in the righthand side'],
    'in front of': ['on the front'],
    'behind the': ['rear of']
}, {
    'matte': ['dull', 'muted', 'lackluster'],
    'rubber': ['plastic', 'rubbery', 'soft'],
    'is': ['exists', 'be']
}, {
    'object': ['item', 'thing'],
    'sphere': [
        'round',
        'rounds',
        'circle',
        'circles',
    ],
    'spheres': [
        'rounds',
        'round',
        'circle',
        'circles',
    ],
    'ball': [
        'rounds',
        'round',
        'circle',
        'circles',
    ],
    'balls': [
        'rounds',
        'round',
        'circle',
        'circles',
    ],
}, {
    'red': ['cardinal', 'crimson', 'pink'],
    'green': ['olive', 'emerald'],
    'blue': [
        'cobalt',
        'azure',
    ],
    'purple': ['lavender', 'violet', 'grape'],
}]


def get_labeler_config(cfg, vocab_size):
  """Get the configuration dictionary for the labeler."""
  if cfg.obs_type == 'state':
    out = {
        'generated_label_num': cfg.generated_label_num,
        'max_sequence_length': cfg.max_sequence_length,
        'sampling_temperature': cfg.sampling_temperature,
        'captioning_encoder': {
            'name': 'state',
            'embedding_dim': 32
        },
        'captioning_decoder': {
            'name': 'state',
            'word_embedding_dim': 16,
            'hidden_units': 100,
            'vocab_size': vocab_size,
        },
        'answering_encoder': {
            'name': 'state',
            'embedding_dim': 64
        },
        'answering_decoder': {
            'name': 'state',
            'word_embedding_dim': 64,
            'hidden_units': 512,
            'vocab_size': vocab_size,
        },
        'captioning_weight_path': None,
        'answering_weight_path': None
    }
    return out
  else:
    return {
        'generated_label_num': cfg.generated_label_num,
        'max_sequence_length': 15,
        'sampling_temperature': cfg.sampling_temperature,
        'captioning_encoder': {
            'name': 'image',
            'embedding_dim': 32
        },
        'captioning_decoder': {
            'name': 'attention',
            'word_embedding_dim': 64,
            'hidden_units': 256,
            'vocab_size': vocab_size,
        },
        'answering_encoder': {
            'name': 'state',
            'embedding_dim': 64
        },
        'answering_decoder': {
            'name': 'state',
            'word_embedding_dim': 64,
            'hidden_units': 512,
            'vocab_size': vocab_size,
        }
    }


class FnApproxHIR:
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
    self._use_labeler_as_reward = cfg.use_labeler_as_reward
    self._use_oracle_instruction = cfg.use_oracle_instruction
    self._use_synonym_for_rollout = cfg.use_synonym_for_rollout

    # Vocab loading
    vocab_path = get_vocab_path(cfg)
    self.vocab_list = wv.load_vocab_list(vocab_path)
    self.vocab_list = ['eos', 'sos', 'nothing'] + self.vocab_list[1:]
    v2i, i2v = wv.create_look_up_table(self.vocab_list)
    self.encode_fn = wv.encode_text_with_lookup_table(
        v2i, max_sequence_length=self.cfg.max_sequence_length)
    self.decode_fn = wv.decode_with_lookup_table(i2v)

    labeler_config = get_labeler_config(cfg, self.vocab_list)

    if self._use_labeler_as_reward or not self._use_oracle_instruction:
      self.labeler = Labeler(labeler_config=labeler_config)
      self.labeler.set_captioning_model(
          labeler_config, labeler_config['captioning_weight_path'])
      self.labeler.set_answering_model(labeler_config,
                                       labeler_config['answering_weight_path'])

  def learn(self, env, agent, replay_buffer):
    """Run learning for 1 cycle with consists of num_episode of episodes.

    Args:
      env: the RL environment
      agent: the RL agent
      replay_buffer: the experience replay buffer

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
      s = env.reset(sample_new_scene)
      episode_experience = []
      episode_reward = 0
      episode_achieved_n = 0
      episode_relabel_n = 0

      # rollout
      g_text, p = env.sample_goal()
      if env.all_goals_satisfied:
        s = env.reset(True)
        g_text, p = env.sample_goal()
      g = self.encode_fn(g_text)
      g = np.squeeze(pad_to_max_length([g], self.cfg.max_sequence_length)[0])
      _ = agent.step(s, g, env, 0.0)  # taking a step to create weights

      for t in range(self.cfg.max_episode_length):
        a = agent.step(s, g, env, self.epsilon)
        s_tp1, r, _, _ = env.step(
            a,
            record_achieved_goal=self._use_oracle_instruction,
            goal=p,
            atomic_goal=self.cfg.record_atomic_instruction)
        if self._use_labeler_as_reward:
          labeler_answer = self.labeler.verify_instruction(
              env.convert_order_invariant_to_direct(s_tp1), g)
          r = float(labeler_answer > 0.5)
        if self._use_oracle_instruction:
          ag = env.get_achieved_goals()
        else:
          ag = [None]
        episode_experience.append((s, a, r, s_tp1, g, ag))
        episode_reward += r
        s = s_tp1
        if r > env.shape_val:
          episode_achieved_n += 1
          g_text, p = env.sample_goal()
          if env.all_goals_satisfied:
            break
          g = self.encode_fn(g_text)
          g = np.squeeze(
              pad_to_max_length([g], self.cfg.max_sequence_length)[0])

      average_per_ep_reward.append(episode_reward)
      average_per_ep_achieved_n.append(episode_achieved_n)

      # processing trajectory
      episode_length = len(episode_experience)

      if not self._use_oracle_instruction:  # generate instructions from traj
        transition_pair = []
        if self.cfg.obs_type == 'order_invariant':
          for t in episode_experience:
            transition_pair.append([
                env.convert_order_invariant_to_direct(t[0]),
                env.convert_order_invariant_to_direct(t[3])
            ])
          transition_pair = np.stack(transition_pair)
        else:
          for t in episode_experience:
            transition_pair.append([t[0], t[3]])

        all_achieved_goals = self.labeler.label_trajectory(
            transition_pair, null_token=2)
        for i in range(len(episode_experience)):
          s, a, r, s_tp1, g, ag = episode_experience[i]
          step_i_text = []
          for inst in all_achieved_goals[i]:
            decoded_inst = self.decode_fn(inst)
            step_i_text.append(decoded_inst)
          episode_experience[i] = [s, a, r, s_tp1, g, step_i_text]

      non_null_future_idx = [[] for _ in range(episode_length)]
      for t in range(episode_length):
        _, _, _, _, _, ag = episode_experience[t]
        if ag:
          for u in range(t):
            non_null_future_idx[u].append(t)

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
          self.hir_relabel(non_null_future_idx, episode_experience, t,
                           replay_buffer, env)

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
        average_batch_loss.append(batch_loss / self.cfg.optimization_steps)

    time_per_episode = (time.time() - tic) / self.cfg.num_episode

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
        'replay_buffer_reward_avg': np.mean(sample_r),
        'replay_buffer_reward_var': np.var(sample_r)
    }
    return stats

  def hir_relabel(self, non_null_future_idx, episode_experience, current_t,
                  replay_buffer, env):
    """Relabeling trajectories.

    Args:
      non_null_future_idx: list of time step where something happens
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
      # TODO(ydjiang): k_immediate logic needs improvement
      for _ in range(self.cfg.k_immediate):
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
            replay_buffer.add((s, a, 0.0, s_tp1, self.encode_fn(negative_ag)))
    # TODO(ydjiang): repeat logit needs improvement
    goal_count, repeat = 0, 0
    while goal_count < self.cfg.future_k and repeat < (ep_len - current_t) * 4:
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
    print('\n#######################################')
    print('Rolling out...')
    print('#######################################')

    # randomly change subset of embedding
    if self._use_synonym_for_rollout and self.cfg.embedding_type == 'random':
      original_embedding = agent.randomize_partial_word_embedding(10)

    all_frames = []
    ep_observation, ep_action, ep_agn = [], [], []
    black_frame = pad_image(env.render(mode='rgb_array')) * 0.0
    goal_sampled = 0
    timeout_count, success = 0, 0
    for ep in range(num_episode):
      s = env.reset(self.cfg.diverse_scene_content)
      all_frames += [black_frame] * 10
      g_text, p = env.sample_goal()
      if env.all_goals_satisfied:
        s = env.reset(True)
        g, p = env.sample_goal()
      goal_sampled += 1
      g = self.encode_fn(g_text)
      g = np.squeeze(pad_to_max_length([g], self.cfg.max_sequence_length)[0])
      if self._use_synonym_for_rollout and self.cfg.embedding_type != 'random':
        # use unseen lexicons for test
        g = paraphrase_sentence(
            self.decode_fn(g), synonym_tables=_SYNONYM_TABLES)
      current_goal_repetition = 0
      for t in range(self.cfg.max_episode_length):
        prob = self.epsilon if record_trajectory else 0.0
        action = agent.step(s, g, env, explore_prob=prob)
        s_tp1, r, _, _ = env.step(
            action,
            record_achieved_goal=False,
            goal=p,
            atomic_goal=self.cfg.record_atomic_instruction)
        s = s_tp1
        all_frames.append(
            add_text(pad_image(env.render(mode='rgb_array')), g_text))
        current_goal_repetition += 1

        if record_trajectory:
          ep_observation.append(env.get_direct_obs().tolist())
          ep_action.append(action)

        sample_new_goal = False
        if r > env.shape_val:
          img = pad_image(env.render(mode='rgb_array'))
          for _ in range(5):
            all_frames.append(add_text(img, g_text, color='green'))
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
          g = self.encode_fn(g_text)
          g = np.squeeze(
              pad_to_max_length([g], self.cfg.max_sequence_length)[0])
          if self._use_synonym_for_rollout and self.cfg.embedding_type != 'random':
            g = paraphrase_sentence(
                self.decode_fn(g), synonym_tables=_SYNONYM_TABLES)
          current_goal_repetition = 0
          goal_sampled += 1

    # restore the original embedding
    if self._use_synonym_for_rollout and self.cfg.embedding_type == 'random':
      agent.set_embedding(original_embedding)

    print('Rollout finished')
    print('{} instrutctions tried given'.format(goal_sampled))
    print('{} instructions timed out'.format(timeout_count))
    print('{} success rate\n'.format(1 - float(timeout_count) / goal_sampled))
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
    return 1 - float(timeout_count) / goal_sampled
