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

import copy
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.optim as optim
import torch
from collections import defaultdict, deque
from gtd.ml.torch.utils import try_gpu
from strategic_exploration.hrl import abstract_state as AS
from strategic_exploration.hrl.bonus import RewardBonus
from strategic_exploration.hrl.graph import DirectedEdge
from strategic_exploration.hrl.policy import Policy
from strategic_exploration.hrl.replay import ReplayBuffer
from strategic_exploration.hrl.rl import Experience
from strategic_exploration.hrl.utils import mean_with_default
from scipy.misc import imread
from torch.nn.utils import clip_grad_norm


class Worker(object):
  """Worker policy interface.

  Owns a bunch of skills and calls them at
    the appropriate times.
    """

  @classmethod
  def from_config(cls, config, num_actions, room_dir):
    skill_pool = SkillPool(config.skill, num_actions,
                           config.max_combined_buffer_size)
    return cls(skill_pool, config.max_steps, config.max_worker_reward,
               config.debug_stats, room_dir)

  def __init__(self, skill_pool, max_steps, max_worker_reward, debug_stats,
               room_dir):
    """Constructs.

        Args: skill_pool (SkillPool)
            max_steps (int): maximum number of steps a worker can be active for
            max_worker_reward (float): worker episode terminates when it
            accumulates this much reward
            debug_stats (bool): if True, logs stats from Skills
            room_dir (str): path to directory containing room visualizations
            named room-1.png, room-2.png, ...
    """
    self._skills = skill_pool
    self._max_steps = max_steps
    self._max_worker_reward = max_worker_reward
    self._debug_stats = debug_stats
    self._room_dir = room_dir

  def act(self, current_state, goal_edge, step, cum_reward):
    """Given the goal edge, and current state, returns an action by

        calling the appropriate skill.

        Args: current_state (State)
            goal_edge (DirectedEdge): goal is goal_edge.end
            step (int): how many steps the worker has been active for
    """
    skill, _ = self._skills.get_skill(goal_edge)
    worker_state = self._skill_state(current_state, goal_edge, step, cum_reward)
    epsilon = self._epsilon(goal_edge)
    return skill.act(worker_state, epsilon=epsilon)

  def add_experience(self, edge, experience, step, cum_reward, success=False):
    """Adds an experience for updating to the skill associated with the

        edge. Reward in the experience should be the env extrinsic reward.

        Args: edge (DirectedEdge) experience (Experience)
            step (int): step at the beginning of the experience
            success (bool): True if the experience was part of a successful
            trajectory
    """
    # Experience doesn't apply
    if step > self.max_steps(edge):
      return

    current_skill_state = self._skill_state(experience.state, edge, step,
                                            cum_reward)
    reward = self.reward(experience.next_state, edge, experience.reward,
                         experience.done)
    next_skill_state = self._skill_state(experience.next_state, edge, step + 1,
                                         cum_reward + reward)
    skill_done = experience.done or step + 1 >= self.max_steps(edge) \
        or cum_reward + reward >= self._max_worker_reward
    skill_experience = Experience(current_skill_state, experience.action,
                                  reward, next_skill_state, skill_done)
    skill, _ = self._skills.get_skill(edge)
    skill.add_experience(
        skill_experience, success, allow_updates=edge.training())

  def mark_reliable(self, edge):
    """Marks the edge as reliable.

    Under the hood, this saves the
        corresponding skill.

        Args: edge (DirectedEdge)
    """
    logging.info("Saving worker associated with: {}".format(edge))
    self._skills.save_skill(edge)
    for parent_edge in edge.end.parents:
      if parent_edge.training():
        self._skills.remove_skill(parent_edge)

  def mark_failed_evaluation(self, edge):
    """Marks a failed evaluation.

    Worker may start new evaluation.

        Args:
            edge (DirectedEdge): the edge whose evaluation was failed.
    """
    next_skill_exists = self._skills.try_next_skill(edge)
    if next_skill_exists:
      edge.start_evaluation()

  def max_steps(self, edge):
    """Returns the maximum allowable steps the worker can be active on

        this edge.

        Args: edge (DirectedEdge)

        Returns:
            int
        """
    return self._max_steps * edge.degree

  def reward(self, next_state, edge, env_reward, done):
    """Defines the worker's intrinsic reward for reaching next_state

        while trying to traverse the edge.

        Args: next_state (State) edge (DirectedEdge)
            env_reward (float): environment extrinsic reward
            done (bool): True if overall episode ended

        Returns:
            float
        """
    if AS.AbstractState(next_state) == edge.end.abstract_state and \
            not done and env_reward >= 0:
      return 1.
    else:
      return 0.

  def stats(self):
    stats = {
        "SKILLS/Learned": self._skills.num_saved_skills,
        "SKILLS/Volatile": self._skills.num_volatile_skills
    }

    if self._debug_stats:
      stats.update(self._skills.stats())
    return stats

  def visualize(self, save_dir):
    self._skills.visualize(save_dir, self._room_dir)

  def __str__(self):
    return "Worker: \n{}".format(str(self._skills))

  def _skill_state(self, state, goal_edge, step, cum_reward):
    """Adds the goal and step to the state.

        Args: state (State)
            goal_edge (DirectedEdge): goal_edge.end is goal
            step (int): number of steps the skill has been active

        Returns:
            State
        """
    goal_abstract_state = goal_edge.end.abstract_state
    abstract_state_diff = \
        goal_abstract_state.numpy - AS.AbstractState(state).unbucketed
    worker_step_frac = float(step) / self.max_steps(goal_edge)
    on_goal = AS.AbstractState(state) == goal_edge.end.abstract_state
    goal = Goal(abstract_state_diff, worker_step_frac, on_goal, cum_reward)

    # Shallow copy OK. Copy references to the np.arrays. Getters don't
    # expose the underlying arrays directly
    state_copy = copy.copy(state)
    state_copy.set_goal(goal)
    return state_copy

  def _epsilon(self, edge):
    """Returns epsilon value to use on this edge.

        Args: edge (DirectedEdge)

        Returns:
            float in [0, 1]
        """
    train_max = 1 if edge.train_count == 0 \
        else 1 << (int(math.ceil(edge.train_count / 100.)) - 1).bit_length()
    train_max *= 75.
    epsilon = 1. - min(edge.train_count / train_max, 1.)

    if not edge.training():
      epsilon = 0.
    return epsilon

  def partial_state_dict(self):
    """Returns partial information to reload with load_state_dict.

        Returns:
            dict
        """
    return {"skills": self._skills.partial_state_dict()}

  def load_state_dict(self, state_dict, edges):
    """Given the partial state_dict from partial_state_dict() and missing

        information, reloads.

        Args:
            state_dict (dict): should come from partial_state_dict
            edges (list[DirectedEdge]): all of the edges in the AbstractGraph at
              the time of serialization
    """
    self._skills.load_state_dict(state_dict["skills"], edges)


class Goal(object):
  """Goal that worker conditions on."""

  def __init__(self, abstract_state_diff, worker_step_frac, on_goal,
               cum_reward):
    """Constructs goal.

        Args:
            abstract_state_diff (np.array): goal abstract state (bucketed) -
              current abstract state (unbucketed)
            worker_step_frac (float): current step / max worker steps
            on_goal (bool): True if current abstract state = goal abstract state
            cum_reward (float): Cumulative worker reward on current trajectory.
    """
    self._numpy = np.zeros(self.size())
    self._numpy[:AS.AbstractState.size()] = abstract_state_diff
    self._numpy[AS.AbstractState.size()] = worker_step_frac
    self._numpy[AS.AbstractState.size() + 1] = on_goal
    self._numpy[AS.AbstractState.size() + 2] = cum_reward

  @property
  def cum_reward(self):
    return self._numpy[-1]

  @property
  def all_but_cum_reward(self):
    return self._numpy[:-1]

  @property
  def on_goal(self):
    return self._numpy[-2]

  @property
  def worker_step_frac(self):
    return self._numpy[-3]

  @property
  def abstract_state_diff(self):
    return self._numpy[:AS.AbstractState.size()]

  @classmethod
  def size(cls):
    return AS.AbstractState.size() + 3

  def numpy(self):
    return self._numpy


class Skill(object):

  @classmethod
  def from_config(cls, config, num_actions, name):
    dqn = try_gpu(Policy.from_config(config.policy, num_actions))
    replay_buffer = ReplayBuffer.from_config(config.buffer)
    imitation_buffer = None
    if config.imitation:
      imitation_buffer = ReplayBuffer.from_config(config.buffer)
    optimizer = optim.Adam(dqn.parameters(), lr=config.learning_rate)
    return cls(dqn, replay_buffer, imitation_buffer, optimizer, name,
               config.sync_target_freq, config.min_buffer_size,
               config.batch_size, config.grad_steps_per_update,
               config.max_grad_norm, num_actions, config.adaptive_update,
               config.epsilon_clipping, config.max_worker_reward,
               config.dqn_vmax, config.dqn_vmin, config)

  def __init__(self, dqn, replay_buffer, imitation_buffer, optimizer, name,
               sync_freq, min_buffer_size, batch_size, grad_steps_per_update,
               max_grad_norm, num_actions, adaptive_update, epsilon_clipping,
               max_worker_reward, dqn_vmax, dqn_vmin, config):
    """
        Args: dqn (DQNPolicy) replay_buffer (ReplayBuffer)
            imitation_buffer (ReplayBuffer): replay buffer for self-imitation
            loss. None to disable self-imitation loss optimizer
            (torch.Optimizer) name (string)
            sync_freq (int): number of updates between syncing the DQN target Q
            network
            min_buffer_size (int): replay buffer must be at least this large
            before taking grad updates
            batch_size (int): number of experience to sample per grad step
            grad_steps_per_update (int): number of grad steps to take per call
            to update
            max_grad_norm (float): gradient is clipped to this norm on each
            update
            adaptive_update (bool): if True, adaptively changes the updates per
            timestep based on successes
            epsilon_clipping (bool): if True, clips epsilon if there have been
            many successes in the past
            max_worker_reward (float): if worker reward hits this, episode is a
            success and terminates
            dqn_vmax (float): vmax term in update_from_experiences
            dqn_vmin (float): vmin term in update_from_experiences
            config (Config): the config with which this Skill was created
    """
    self._dqn = dqn
    self._replay_buffer = replay_buffer
    self._imitation_buffer = imitation_buffer
    self._optimizer = optimizer
    self._frozen = False
    self._config = config
    self.name = name
    self._sync_freq = sync_freq
    self._min_buffer_size = min_buffer_size
    self._batch_size = batch_size
    self._grad_steps_per_update = grad_steps_per_update
    self._max_grad_norm = max_grad_norm
    self._updates = 0
    self._num_actions = num_actions
    self._reward_bonus = RewardBonus()
    self._episode_reward = 0.
    self._episode_rewards = deque(maxlen=10)
    self._success_rate = deque(maxlen=10)
    self._epsilon = 0.
    self._adaptive_update = adaptive_update
    self._epsilon_clipping = epsilon_clipping
    self._max_worker_reward = max_worker_reward
    self._dqn_vmax = dqn_vmax
    self._dqn_vmin = dqn_vmin

  def add_experience(self, experience, success=False, allow_updates=True):
    """Adds the experience to the skill's replay buffer.

        Args: experience (Experience)
            success (bool): see SkillPool
            allow_updates (bool): if True, takes an update
    """
    if not self._frozen:
      # Memory optimization
      self._episode_reward += experience.reward
      if experience.done:
        self._success_rate.append(
            experience.reward +
            experience.state.goal.cum_reward >= self._max_worker_reward)
        self._episode_rewards.append(self._episode_reward)
        self._episode_reward = 0.
      experience.state.drop_teleport()
      experience.next_state.drop_teleport()
      self._reward_bonus.observe(experience)
      if success and self._imitation_buffer is not None:
        self._imitation_buffer.add(experience)
      self._replay_buffer.add(experience)

      if allow_updates:
        self.update()

  def update(self):
    """Takes gradient steps by sampling from replay buffer."""

    def take_grad_step(loss):
      self._optimizer.zero_grad()
      loss.backward()

      # clip according to the max allowed grad norm
      grad_norm = clip_grad_norm(
          self._dqn.parameters(), self._max_grad_norm, norm_type=2)

      # TODO: Fix
      finite_grads = True

      # take a step if the grads are finite
      if finite_grads:
        self._optimizer.step()
      return finite_grads, grad_norm

    if self._frozen:
      return

    # Adaptive success: w/ prob 1 - current success rate, take update
    success_rate = mean_with_default(self._success_rate, 0.)
    update = not self._adaptive_update or np.random.random() > success_rate
    if len(self._replay_buffer) >= self._min_buffer_size and update:
      for _ in range(self._grad_steps_per_update):
        self._updates += 1
        if self._updates % self._sync_freq == 0:
          self._dqn.sync_target()
        experiences = self._replay_buffer.sample(self._batch_size)
        experiences = [self._reward_bonus(e) for e in experiences]
        td_error = self._dqn.update_from_experiences(
            experiences,
            np.ones(self._batch_size),
            take_grad_step,
            vmax=self._dqn_vmax,
            vmin=self._dqn_vmin)
        max_td_error = torch.max(td_error)[0]

        if (max_td_error > 4).any():
          logging.warning("Large error: {} on skill: {}".format(
              max_td_error, self))

    imitation_update = update and self._imitation_buffer is not None
    if imitation_update and len(self._imitation_buffer) > 0:
      imitation_experiences = self._imitation_buffer.sample(self._batch_size)
      self._dqn.update_from_imitation(imitation_experiences, take_grad_step,
                                      self._max_worker_reward)

  def freeze(self):
    """Freezes the skill's parameters, freeing all possible memory.

        Subsequent calls to update are effectively no-ops.
        """
    # Free replay buffer memory
    self._replay_buffer = None
    self._imitation_buffer = None
    self._frozen = True
    self._reward_bonus.clear()

  def act(self, state, epsilon=None, **kwargs):
    """Given the current state, returns an action.

    Supports all the
        keyword args as DQNPolicy.

        Args: state (State)

        Returns:
            action (int)
        """
    if self._epsilon_clipping and epsilon is not None:
      epsilon -= mean_with_default(self._success_rate, 0.)
      epsilon = max(epsilon, 0.)
    self._epsilon = epsilon or 0.
    return self._dqn.act(state, epsilon=epsilon, **kwargs)

  def clone(self):
    config = self._config
    dqn = try_gpu(Policy.from_config(config.policy, self._num_actions))
    dqn._Q.load_state_dict(self._dqn._Q.state_dict())
    dqn._target_Q.load_state_dict(self._dqn._target_Q.state_dict())
    replay_buffer = ReplayBuffer(config.buffer_max_size)
    optimizer = optim.Adam(dqn.parameters(), lr=config.learning_rate)
    return Skill(dqn, replay_buffer, optimizer, self.name + "-clone",
                 config.sync_target_freq, config.min_buffer_size,
                 config.batch_size, config.grad_steps_per_update,
                 config.max_grad_norm, self._num_actions, config)

  @property
  def frozen(self):
    return self._frozen

  @property
  def replay_buffer_size(self):
    if self._replay_buffer is None:
      return 0
    else:
      return len(self._replay_buffer)

  def stats(self):
    stats = {}
    for k, v in self._dqn.stats().items():
      stats["{}_{}".format(self.name, k)] = v
    stats["{}_avg_reward".format(self.name)] = mean_with_default(
        self._episode_rewards, 0.)
    stats["{}_success_rate".format(self.name)] = mean_with_default(
        self._success_rate, 0.)
    stats["{}_epsilon".format(self.name)] = self._epsilon
    return stats

  def __str__(self):
    return "Skill({}, frozen={})".format(self.name, self.frozen)

  __repr__ = __str__


class SkillPool(object):
  """Maintains associations between edges and skills.

  Each skill is in
    either a volatile or saved state. Saved skills may not be updated and
    may be shared amongst many edges. Volatile skills are associated with a
    unique edge and can be deleted / updated.
    """

  def __init__(self, skill_config, num_actions, max_combined_buffer_size):
    """
        Args:
            skill_config (Config): the config for creating new skills
            num_actions (int): the number of actions for each skill
            max_combined_buffer_size (int): maximum number of entries in the
              replay buffer amongst all skills
    """
    # bucket_key --> list[Skill]
    self._saved_skills = defaultdict(list)
    # index --> Skill, indices may not be contiguous
    self._volatile_skills = {}
    # key --> (index, volatile)
    self._edge_to_metadata = defaultdict(lambda: (0, False))

    self._skill_config = skill_config
    self._num_actions = num_actions

    # key --> timestamp (int)
    # if key corresponds to saved skill, then timestamp is np.inf
    self._timestamps = {}
    self._curr_time = 0
    self._max_combined_buffer_size = max_combined_buffer_size

  def get_skill(self, edge):
    """Returns the associated skill and whether or not it is saved.

        Args: edge (DirectedEdge)

        Returns:
            Skill
            saved (bool)
        """
    self._timestamps[edge] = self._curr_time
    self._tick()
    return self._get_skill(edge)

  def remove_skill(self, edge):
    """Removes the skill associated with this edge, if there is one.

    If
        the skill is not volatile, raises an error. The edge is placed in
        evaluating mode afterwards.

        Args: edge (DirectedEdge)
    """
    logging.info("Removing skill associated with edge: {}".format(edge))
    if edge not in self._edge_to_metadata:
      raise ValueError("No skill found for: {}".format(edge))

    index, volatile = self._edge_to_metadata[edge]
    if not volatile:
      raise ValueError(
          "Removing skill for {}, skill not volatile.".format(edge))

    logging.info("Removing from ({}, {})".format(index, volatile))
    logging.info("Removing skill: {}".format(self._volatile_skills[index]))
    del self._volatile_skills[index]
    del self._edge_to_metadata[edge]
    del self._timestamps[edge]
    edge.start_evaluation()

  def try_next_skill(self, edge):
    """Associates the edge with the next skill with the same edge

        difference. If there are more skills with the same edge difference,
        returns True. Otherwise, effectively a no-op and returns False.

        Args: edge (DirectedEdge)

        Returns:
            bool
        """
    name = "{} - {}".format(edge.start.uid, edge.end.uid)
    index, volatile = self._edge_to_metadata[edge]
    if volatile:
      return False
    else:
      bucket = self._saved_skills[tuple(edge.state_difference)]
      if index < len(bucket) - 1:
        self._edge_to_metadata[edge] = (index + 1, volatile)
      else:
        index = max(list(self._volatile_skills.keys()) or [0]) + 1
        if edge.degree < 3:
          self._volatile_skills[index] = Skill.from_config(
              self._skill_config, self._num_actions, name)
        else:
          config_copy = copy.deepcopy(self._skill_config)
          config_copy.put("policy.observation_type",
                          config_copy.alternate_observation_type)
          self._volatile_skills[index] = Skill.from_config(
              config_copy, self._num_actions, name + "-pixel")
        self._edge_to_metadata[edge] = (index, True)
      return True

  def save_skill(self, edge):
    """Marks the skill associated with this edge as saved.

    Freezes the
        skill.

        Args: edge (DirectedEdge)
    """
    index, volatile = self._edge_to_metadata[edge]
    logging.info("Saving skill edge={} index={}, volatile={}".format(
        edge, index, volatile))
    if volatile:
      skill = self._volatile_skills[index]
      bucket = self._saved_skills[tuple(edge.state_difference)]
      bucket.append(skill)
      self._edge_to_metadata[edge] = (len(bucket) - 1, False)
      del self._volatile_skills[index]
      skill.freeze()

  def visualize(self, save_dir, room_dir):
    skill_to_edge = defaultdict(list)  # Skill --> DirectedEdge
    buckets = defaultdict(list)  # (room, edge type) --> list[Skill]
    for edge in self._timestamps.keys():
      if edge.reliable():
        skill, _ = self._get_skill(edge)
        skill_to_edge[skill.name].append(edge)
        attrs = [int(edge.start.abstract_state.room_number)]
        attrs.extend(edge.start.abstract_state.match_attributes)
        edge_type = tuple(edge.state_difference)
        buckets[(tuple(attrs), edge_type)].append(skill)

    skill_to_color = defaultdict(lambda: np.random.random((3, 1)))
    fig = plt.figure()
    for (attrs, edge_type), skill_list in \
            buckets.items():
      room_num = attrs[0]
      room_path = os.path.join(room_dir, "room-{}.png".format(room_num))
      if not os.path.exists(room_path):
        continue
      plt.imshow(imread(room_path))

      arrow_xs = []
      arrow_ys = []
      arrow_us = []
      arrow_vs = []
      colors = []
      for skill in skill_list:
        for edge in skill_to_edge[skill.name]:
          match = \
              room_num == int(
                  edge.start.abstract_state.room_number) and \
              np.array_equal(
                      attrs[1:],
                      edge.start.abstract_state.match_attributes)

          if match:
            arrow_us.append(edge.end.abstract_state.pixel_x -
                            edge.start.abstract_state.pixel_x)
            arrow_vs.append(edge.end.abstract_state.pixel_y -
                            edge.start.abstract_state.pixel_y)
            arrow_xs.append(edge.start.abstract_state.pixel_x)
            arrow_ys.append(edge.start.abstract_state.pixel_y)
            colors.append(skill_to_color[skill])

      if len(arrow_xs) > 0:
        plt.quiver(
            arrow_xs,
            arrow_ys,
            arrow_us,
            arrow_vs,
            color=colors,
            scale=1,
            scale_units="xy",
            angles="xy")

        save_path = os.path.join(save_dir, "{}-{}.png".format(attrs, edge_type))
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight")
      plt.clf()
    plt.close(fig)

  def stats(self):
    stats = {}
    for skill_list in self._saved_skills.values():
      for skill in skill_list:
        for k, v in skill.stats().items():
          stats["WORKER/{}".format(k)] = v

    for skill in self._volatile_skills.values():
      for k, v in skill.stats().items():
        stats["WORKER/{}".format(k)] = v
    return stats

  def _tick(self):
    """Increments the current time and evicts skills based on LRU, if

        the combined buffer sizes exceeds the max.
        """
    self._curr_time += 1

    if self._curr_time % 200000 == 0:
      logging.info("{} {} {}".format("=" * 20, self._curr_time, "=" * 20))
      combined_buffer_size = 0
      for edge in self._timestamps:
        skill, saved = self._get_skill(edge)
        if saved:
          self._timestamps[edge] = np.inf
          assert skill.frozen
          assert skill.replay_buffer_size == 0
        combined_buffer_size += skill.replay_buffer_size
      logging.info("Combined buffer size: {}".format(combined_buffer_size))

      lru_skills = sorted(
          self._timestamps.items(), key=lambda x: x[1], reverse=True)

      while combined_buffer_size > \
              self._max_combined_buffer_size * 0.75:
        edge, timestamp = lru_skills.pop()
        skill, saved = self._get_skill(edge)
        logging.info("Evicting {}, edge={}, timestamp={}".format(
            skill, edge, timestamp))
        combined_buffer_size -= skill.replay_buffer_size
        logging.info("Freed {} buffer entries".format(skill.replay_buffer_size))
        assert not saved
        self.remove_skill(edge)
      logging.info("{} skills left".format(len(lru_skills)))

  @property
  def num_saved_skills(self):
    return sum(len(skill_list) for skill_list in self._saved_skills.values())

  @property
  def num_volatile_skills(self):
    return len(self._volatile_skills)

  def partial_state_dict(self):
    """Returns partial information used to reload in load_state_dict.

        Returns:
            dict
        """
    # skill_config, _num_actions, _max_combined_buffer_size reloaded from
    # config
    edge_to_metadata_dict = {
        edge.summary(): (index, volatile)
        for edge, (index, volatile) in self._edge_to_metadata.items()
    }

    timestamps_dict = {
        edge.summary(): timestamp
        for edge, timestamp in self._timestamps.items()
    }

    return {
        "saved_skills": self._saved_skills,
        "volatile_skills": self._volatile_skills,
        "edge_to_metadata": edge_to_metadata_dict,
        "timestamps": timestamps_dict,
        "curr_time": self._curr_time,
    }

  def load_state_dict(self, state_dict, edges):
    """Given a partial state dict and additional missing information,

        reloads.

        Args:
            state_dict (dict): from partial_state_dict
            edges (list[DirectedEdge]): all edges in the AbstractGraph at the
              time of serialization
    """
    self._saved_skills = state_dict["saved_skills"]
    self._volatile_skills = state_dict["volatile_skills"]
    self._curr_time = state_dict["curr_time"]

    summary_to_edge = {edge.summary(): edge for edge in edges}
    for summary, metadata in state_dict["edge_to_metadata"].items():
      edge = summary_to_edge[summary]
      self._edge_to_metadata[edge] = metadata

    for summary, timestamp in state_dict["timestamps"].items():
      edge = summary_to_edge[summary]
      self._timestamps[edge] = timestamp

  def _get_skill(self, edge):
    # Must be called after tick, otherwise might get evicted
    index, volatile = self._edge_to_metadata[edge]
    if not volatile:
      bucket = self._saved_skills[tuple(edge.state_difference)]
      if len(bucket) == 0:
        index = max(list(self._volatile_skills.keys()) or [0]) + 1
        volatile = True
        self._edge_to_metadata[edge] = (index, volatile)
        name = "{} - {}".format(edge.start.uid, edge.end.uid)
        if edge.degree < 3:
          self._volatile_skills[index] = Skill.from_config(
              self._skill_config, self._num_actions, name)
        else:
          config_copy = copy.deepcopy(self._skill_config)
          config_copy.put("policy.observation_type",
                          config_copy.alternate_observation_type)
          self._volatile_skills[index] = Skill.from_config(
              config_copy, self._num_actions, name + "-pixel")
      else:
        return bucket[index], not volatile
    return self._volatile_skills[index], not volatile

  def __str__(self):
    s = "Saved skills:\n"
    for bucket_key, bucket in self._saved_skills.items():
      s += "{}: {}\n".format(bucket_key, bucket)
    s += "=" * 30 + "\n"
    s += "Volatile skills:\n"
    for _, skill in self._volatile_skills.items():
      s += "{}\n".format(skill)
    s += "=" * 30 + "\n"
    for edge in self._timestamps:
      skill, saved = self._get_skill(edge)
      if saved:
        s += "{}: {}\n".format(edge, skill)
    return s
