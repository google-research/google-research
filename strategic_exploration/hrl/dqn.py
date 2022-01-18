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

import logging
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import strategic_exploration.hrl.embeddings as E
from collections import deque
from gtd.ml.torch.utils import GPUVariable
from strategic_exploration.hrl.policy import Policy
from strategic_exploration.hrl.schedule import LinearSchedule
from strategic_exploration.hrl.utils import mean_with_default
from strategic_exploration.hrl.worker import Goal


class DQNPolicy(Policy):

  @classmethod
  def from_config(cls, config, num_actions):
    if config.observation_type == "ram":
      embedder_factory = E.RAMStateEmbedder
    elif config.observation_type == "pixel":
      embedder_factory = E.PixelStateEmbedder
    elif config.observation_type == "local-pixel":
      embedder_factory = E.LocalPixelStateEmbedder
    elif config.observation_type == "extra-local-pixel":
      embedder_factory = E.ExtraLocalPixelStateEmbedder
    elif config.observation_type == "ram-goal":
      embedder_factory = lambda: E.ConcatenateEmbedder(E.RAMStateEmbedder(),
                                                       E.GoalEmbedder())
    elif config.observation_type == "pixel-goal":
      embedder_factory = lambda: E.ConcatenateEmbedder(E.PixelStateEmbedder(),
                                                       E.GoalEmbedder())
    elif config.observation_type == "pixel-reward-embed-goal":
      embedder_factory = lambda: E.ConcatenateEmbedder(E.PixelStateEmbedder(),
                                                       E.GoalRewardEmbedder())
    elif config.observation_type == "pixel-no-reward-goal":
      embedder_factory = lambda: E.ConcatenateEmbedder(
          E.PixelStateEmbedder(), E.IgnoreRewardGoalEmbedder())
    elif config.observation_type == "local-pixel-goal":
      embedder_factory = lambda: E.ConcatenateEmbedder(
          E.LocalPixelStateEmbedder(), E.GoalEmbedder())
    elif config.observation_type == "extra-local-pixel-goal":
      embedder_factory = lambda: E.ConcatenateEmbedder(
          E.ExtraLocalPixelStateEmbedder(), E.GoalEmbedder())
    elif config.observation_type == "goal-only":
      embedder_factory = E.GoalEmbedder
    elif config.observation_type == "reward-embed-goal-only":
      embedder_factory = E.GoalRewardEmbedder
    elif config.observation_type == "no-reward-goal-only":
      embedder_factory = E.IgnoreRewardGoalEmbedder
    else:
      raise ValueError("{} not a supported type of observation type".format(
          config.observation_type))

    return cls(num_actions, LinearSchedule.from_config(config.epsilon_schedule),
               config.test_epsilon, embedder_factory, config.debug_stats,
               config.gamma)

  def __init__(self,
               num_actions,
               epsilon_schedule,
               test_epsilon,
               state_embedder_factory,
               debug_stats,
               gamma=0.99):
    """
        DQNPolicy should typically be constructed via from_config, and not
        through the constructor.

        Args:
            num_actions (int): the number of possible actions to take at each
              state
            epsilon_schedule (Schedule): defines rate at which epsilon decays
            test_epsilon (float): epsilon to use during test time (when test is
              True in act)
            state_embedder_factory (Callable --> StateEmbedder): type of state
              embedder to use
            debug_stats (bool): if True, logs Q values
            gamma (float): discount factor
    """
    super(DQNPolicy, self).__init__()
    self._Q = DuelingNetwork(num_actions, state_embedder_factory())
    self._target_Q = DuelingNetwork(num_actions, state_embedder_factory())
    self._num_actions = num_actions
    self._epsilon_schedule = epsilon_schedule
    self._test_epsilon = test_epsilon
    self._gamma = gamma

    self._debug_stats = debug_stats
    # Used for generating statistics about the policy
    # Average of max Q values
    self._max_q = deque(maxlen=1000)
    self._min_q = deque(maxlen=1000)
    # Average grad norm during loss steps
    self._grad_norms = deque(maxlen=1000)
    # Average loss on update_from_experiences
    self._td_losses = deque(maxlen=1000)
    self._max_target = deque(maxlen=1000)
    self._min_target = deque(maxlen=1000)

    # Stats from update_from_imitation
    self._imitation_grad_norms = deque(maxlen=1000)
    self._regression_losses = deque(maxlen=1000)
    self._margin_losses = deque(maxlen=1000)

  def act(self, state, test=False, epsilon=None):
    """
        Args: state (State)
            test (bool): if True, takes on the test epsilon value
            epsilon (float | None): if not None, overrides the epsilon greedy
            schedule with this epsilon value. Mutually exclusive with test flag

        Returns:
            int: action
        """
    q_values = self._Q([state])
    if test:
      assert epsilon is None
      epsilon = self._test_epsilon
    elif epsilon is None:
      epsilon = self._epsilon_schedule.step()

    if self._debug_stats:
      self._max_q.append(torch.max(q_values)[0].cpu().data.numpy()[0])
      self._min_q.append(torch.min(q_values)[0].cpu().data.numpy()[0])
    return epsilon_greedy(q_values, epsilon)[0]

  def update_from_experiences(self,
                              experiences,
                              weights,
                              take_grad_step,
                              vmax=None,
                              vmin=None):
    """Updates parameters from a batch of experiences

        Minimizing the loss:

            (target - Q(s, a))^2

            target = r if done
                     r + \gamma * max_a' Q(s', a')

            target is clamped between [vmin, vmax - G_t] if vmin, vmax provided

        Args:
            experiences (list[Experience]): batch of experiences, state and
              next_state may be LazyFrames or np.arrays
            weights (list[float]): importance weights on each experience
            take_grad_step (Callable(loss)): takes the loss and updates
              parameters
            vmax (float | None): if None, no clamping
            vmin (float | None): if None, no clamping

        Returns:
            td_error (GPUVariable[FloatTensor]): (batch_size), error per
                experience
        """
    batch_size = len(experiences)
    states = [e.state for e in experiences]
    actions = GPUVariable(
        torch.LongTensor(np.array([np.array(e.action) for e in experiences])))
    next_states = [e.next_state for e in experiences]
    rewards = GPUVariable(
        torch.FloatTensor(np.array([e.reward for e in experiences])))

    # (batch_size,) 1 if was not done, otherwise 0
    not_done_mask = GPUVariable(
        torch.FloatTensor(np.array([1 - e.done for e in experiences])))
    weights = GPUVariable(torch.FloatTensor(np.array(weights)))

    current_state_q_values = self._Q(states).gather(1, actions.unsqueeze(1))

    # DDQN
    next_state_q_values = self._Q(next_states)
    best_q_values, best_actions = torch.max(next_state_q_values, 1)
    best_actions = best_actions.unsqueeze(1)
    target_q_values = self._target_Q(next_states).gather(
        1, best_actions).squeeze(1)
    targets = rewards + self._gamma * (target_q_values * not_done_mask)

    if vmax is not None:
      # targets - (targets - G_t)_+
      max_reward_to_go = GPUVariable(
          torch.FloatTensor(
              np.array([vmax - e.state.goal.cum_reward for e in experiences])))
      clip_amount = torch.clamp(targets - max_reward_to_go, min=0.)
      targets = targets - clip_amount
    if vmin is not None:
      targets = torch.clamp(targets, min=vmin)

    targets.detach_()  # Don't backprop through targets
    td_error = current_state_q_values.squeeze() - targets
    loss = torch.mean((td_error**2) * weights)
    grad_norm = take_grad_step(loss)[1]

    if grad_norm > 100:
      logging.warning("Large grad norm: {}".format(grad_norm))
      logging.warning("TD Errors: {}".format(td_error))
      logging.warning("Predicted Q-values: {}".format(current_state_q_values))
      logging.warning("Targets: {}".format(targets))

    if self._debug_stats:
      max_target = torch.max(targets)[0]
      min_target = torch.min(targets)[0]
      self._max_target.append(max_target)
      self._min_target.append(min_target)
      self._td_losses.append(loss)
      self._grad_norms.append(grad_norm)
    return td_error

  def update_from_imitation(self, experiences, take_grad_step, vmax):
    """Updates the Q values to match the reward to go and a margin loss.

            regression_loss = ||G_t - Q(s_t, a_t)||_2
            G_t = return of the episode from timestep t and onward

            margin_loss = max_a (Q(s, a) + l(a_e, a)) - Q(s, a_e)
            a_e is "expert" action from the experience
            l(a_e, a) = 0 if a_e = a,
                      = K otherwise. (following DQfD)

        Args:
            experiences (list[Experience]): batch of experiences, state and
              next_state may be LazyFrames or np.arrays
            take_grad_step (Callable(loss)): takes the loss and updates
            vmax (float): optimal value of best state
    """
    states = [e.state for e in experiences]
    actions = GPUVariable(
        torch.LongTensor(np.array([np.array(e.action) for e in experiences])))

    # Computes V-max (trajectories assumed to achieve V-max reward)
    rewards_to_go = GPUVariable(
        torch.FloatTensor(
            np.array([vmax - e.state.goal.cum_reward for e in experiences])))
    q_values = self._Q(states)
    expert_q_values = q_values.gather(1, actions.unsqueeze(1))
    regression_loss = torch.mean((rewards_to_go - expert_q_values)**2)

    margin = 0.5 * torch.ones_like(q_values).scatter(1, actions.unsqueeze(1),
                                                     0.)
    max_margin_q_values, _ = torch.max(q_values + margin, dim=1)
    margin_loss = torch.mean(max_margin_q_values - expert_q_values)
    grad_norm = take_grad_step(regression_loss + margin_loss)[1]

    if self._debug_stats:
      self._margin_losses.append(margin_loss)
      self._regression_losses.append(regression_loss)
      self._imitation_grad_norms.append(grad_norm)

  def sync_target(self):
    """Syncs the target Q values with the current Q values"""
    self._target_Q.load_state_dict(self._Q.state_dict())

  def stats(self):
    """See comments in constructor for more details about what these stats

        are
    """
    return {
        "epsilon":
            self._epsilon_schedule.step(take_step=False),
        "Max Q":
            mean_with_default(self._max_q, None),
        "Min Q":
            mean_with_default(self._min_q, None),
        "Avg grad norm":
            mean_with_default(self._grad_norms, None),
        "Max target":
            mean_with_default(self._max_target, None),
        "Min target":
            mean_with_default(self._min_target, None),
        "TD loss":
            mean_with_default(self._td_losses, None),
        "Imitation grad norm":
            mean_with_default(self._imitation_grad_norms, None),
        "Margin loss":
            mean_with_default(self._margin_losses, None),
        "Regression loss":
            mean_with_default(self._regression_losses, None)
    }


class DQN(nn.Module):
  """Implements the Q-function."""

  def __init__(self, num_actions, state_embedder):
    """
        Args:
            num_actions (int): the number of possible actions at each state
            state_embedder (StateEmbedder): the state embedder to use
    """
    super(DQN, self).__init__()
    self._state_embedder = state_embedder
    self._q_values = nn.Linear(self._state_embedder.embed_dim, num_actions)

  def forward(self, states):
    """Returns Q-values for each of the states.

        Args:
            states (FloatTensor): shape (batch_size, 84, 84, 4)

        Returns:
            FloatTensor: (batch_size, num_actions)
        """
    return self._q_values(self._state_embedder(states))


class DuelingNetwork(DQN):
  """Implements the following Q-network:

        Q(s, a) = V(s) + A(s, a) - avg_a' A(s, a')
    """

  def __init__(self, num_actions, state_embedder):
    super(DuelingNetwork, self).__init__(num_actions, state_embedder)
    self._V = nn.Linear(self._state_embedder.embed_dim, 1)
    self._A = nn.Linear(self._state_embedder.embed_dim, num_actions)

  def forward(self, states):
    state_embedding = self._state_embedder(states)
    V = self._V(state_embedding)
    advantage = self._A(state_embedding)
    mean_advantage = torch.mean(advantage)
    return V + advantage - mean_advantage


def epsilon_greedy(q_values, epsilon):
  """Returns the index of the highest q value with prob 1 - epsilon,
    otherwise uniformly at random with prob epsilon.

    Args:
        q_values (Variable[FloatTensor]): (batch_size, num_actions)
        epsilon (float)

    Returns:
        list[int]: actions
    """
  batch_size, num_actions = q_values.size()
  _, max_indices = torch.max(q_values, 1)
  max_indices = max_indices.cpu().data.numpy()
  actions = []
  for i in range(batch_size):
    if random.random() > epsilon:
      actions.append(max_indices[i])
    else:
      actions.append(random.randint(0, num_actions - 1))
  return actions
