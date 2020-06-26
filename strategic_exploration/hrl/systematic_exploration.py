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

import copy
import numpy as np
import torch
from collections import deque, namedtuple
from gtd.ml.torch.utils import assert_tensor_equal
from strategic_exploration.hrl.policy import Policy
import matplotlib.pyplot as plt


class SystematicExploration(Policy):
  """Systematically explores the state space.

  Assumes that the transition
    dynamics are deterministic over state histories:

        p(s_{t + 1} | h_t, a_t) \in {0, 1},
            where h_t = [s_t, s_{t - 1}, ..., s_{t - k}] and
                    k = history length (default 2)

    At each state history, tries to learn the transition dynamics of each of
    the possible actions by expanding "frontiers."

    Usage:
        policy = SystematicExploration(num_actions)
        state = env.reset()
        policy.reset(state)  # Call at beginning of episode

        while True:  # Does one episode
            action = policy.act(state)
            next_state, reward, done = env.step(action)
            policy.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    """

  def __init__(self, num_actions, history_length=1):
    self._num_actions = num_actions
    self._history_length = history_length  # k in the docs

    self._exploration_graph = ExplorationGraph(num_actions)
    self._current_node = self._exploration_graph.get_node(
        History([None] * history_length), False)

    # list[(ExplorationNode, action)] actions to take
    # act follows these actions if planned_path is not None
    self._planned_path = None

    self._current_path = []
    self._best_path = None

    self._current_reward = 0.
    self._best_reward = 0.
    self._best_node_so_far = None  # Node with highest reward observed

  def act(self, state):
    """If the current state is a "frontier", then expands the "frontier" by

        taking an action that has not been taken before. Otherwise, takes
        actions to go to a "frontier."

        Args: states (np.array)

        Returns:
            action (int)
        """
    assert_tensor_equal(state, self._current_node.history[0])
    unexplored_actions = self._current_node.unexplored_actions

    if self._planned_path is not None:
      expected_state, action = self._planned_path.pop(0)
      assert expected_state == self._current_node
      if len(self._planned_path) == 0:
        self._planned_path = None
      return action
    elif len(unexplored_actions) > 0:  # On a frontier
      action = np.random.choice(unexplored_actions)
      self._planned_path = None  # Give up on any earlier plans
      return action
    else:
      print("Planning")
      self._planned_path = self._exploration_graph.reachable_frontier(
          self._current_node)
      if self._planned_path is not None:
        expected_state, action = self._planned_path.pop(0)
        assert expected_state == self._current_node
        if len(self._planned_path) == 0:
          self._planned_path = None
        return action

    # No known way to get to frontier
    print("No reachable frontiers")
    return np.random.choice(range(self._num_actions))

  def observe(self, state, action, reward, next_state, done):
    """Records the transition dynamics of taking the action and

        transitioning the next_state, where the current history ends with
        STATE.

        Args: state (FloatTensor) action (int) reward (float) next_state
        (FloatTensor) done (bool)
    """
    new_history = self._current_node.history.clone()
    new_history.update(next_state)

    self._current_path.append((self._current_node, action))

    next_node = self._exploration_graph.get_node(new_history, done)
    self._current_node.observe(action, reward, next_node)
    self._current_node = next_node

    self._current_reward += reward

    if self._current_reward > self._best_reward:
      self._best_reward = self._current_reward
      self._best_node_so_far = next_node
      self._best_path = list(self._current_path)
      print("Setting best: {}".format(len(self._best_path)))
      print("Best path: {}".format(self._best_path[:2]))

  def reset(self, state):
    """Must be called at the beginning of every new episode.

    Resets the
        current history and sets the current state to be state.

        Args:
            state (FloatTensor): The first state of the episode
    """
    self._current_node = self._exploration_graph.get_node(
        History([state] + [None] * (self._history_length - 1)), False)
    self._planned_path = None

    self._current_reward = 0.

    self._current_path = []

    if self._best_node_so_far is not None:  # Return to the best node you've seen
      print("Returning to reward {} node".format(self._best_reward))
      self._planned_path = list(self._best_path)
      print("Path length: {}".format(len(self._planned_path)))

  def stats(self):
    return {
        "Best reward": self._best_reward,
        "Number of states discovered": self._exploration_graph.num_nodes
    }


class ExplorationNode(object):
  SAVE_INDEX = 0
  """A node in the exploration graph that the SystematicExploration policy

    builds. Each node represents a history of states [s_t, ..., s_{t - k - 1}].
    Edges from nodes represent transitions from taking actions, as well as the
    reward dynamics.
    """

  def __init__(self, history, done, num_actions):
    """
        Args:
            history (History): list of past k states ordered as [s_t, ..., s_{t
              - k - 1}]
            done (bool): If this is a terminal history
            num_actions (int): Number of possible actions to take in this domain
    """
    self._history = history
    self._done = done

    # self._transitions[i] = (reward, next_node) for action (i)
    self._transitions = [(None, None)] * num_actions

  def observe(self, action, reward, next_node):
    """Creates a transition edge to the next node.

    If the edge already
        exists, the next_node and the reward should match the existing edge.
        Currently overwrites the current edge, if the edge already exists,
        printing a message if the edge information does not match.

        Effectively logs:
            p(reward | action, current_history) = 1
            p(next_node | action, current_history) = 1

        Args: action (int) reward (float) next_node (ExplorationNode)
    """
    if self._transitions[action][0] is not None:
      recorded_reward, recorded_next_node = self._transitions[action]
      if recorded_reward != reward:
        print("Recorded: p({} | {}, {}) = 1".format(recorded_reward, action,
                                                    self))
        print("New: p({} | {}, {}) = 1".format(reward, action, self))
      if recorded_next_node != next_node:
        plt.imsave(
            "current-{}-{}.png".format(action, ExplorationNode.SAVE_INDEX),
            self.history[0].numpy().reshape((84, 84)),
            cmap="gray")
        plt.imsave(
            "recorded-{}-{}.png".format(action, ExplorationNode.SAVE_INDEX),
            recorded_next_node.history[0].numpy().reshape((84, 84)),
            cmap="gray")
        plt.imsave(
            "new-{}-{}.png".format(action, ExplorationNode.SAVE_INDEX),
            next_node.history[0].numpy().reshape((84, 84)),
            cmap="gray")
        ExplorationNode.SAVE_INDEX += 1
        print("Recorded: p({} | {}, {}) = 1".format(recorded_next_node, action,
                                                    self))
        print("New: p({} | {}, {}) = 1".format(next_node, action, self))
    self._transitions[action] = (reward, next_node)

  def next_node(self, action):
    """Returns the ExplorationNode that this action transitions to, if it

        has been observed before. Otherwise, returns None.

        Args: action (int)

        Returns:
            ExplorationNode
        """
    return self._transitions[action][1]

  def reward(self, action):
    """Returns the reward that you would get if you took this action at the

        current history, if it has been observed before. Otherwise returns
        None.

        Args: action (int)

        Returns:
            float
        """
    return self._transitions[action][0]

  @property
  def unexplored_actions(self):
    """Returns a list[int] of actions which have not been played at the

        current history.
        """
    return [
        action for action, (reward, next_node) in enumerate(self._transitions)
        if reward is None
    ]

  @property
  def done(self):
    """Returns if this node is a terminal history."""
    return self._done

  @property
  def history(self):
    """Returns history, see constructor"""
    return self._history

  def __str__(self):
    return "ExplorationNode({})".format(self._history)

  __repr__ = __str__


# The action that was taken to get from prev_path_node to node
PathNode = namedtuple("PathNode",
                      ["exploration_node", "action_taken", "prev_path_node"])


class ExplorationGraph(object):
  """Collection of ExplorationNodes arranged into a graph."""

  def __init__(self, num_actions):
    """
        Args:
            num_actions (int): Number of possible actions to take in this task
    """
    self._nodes = {}  # (history, done) --> ExplorationNode
    self._num_actions = num_actions

  @property
  def num_nodes(self):
    return len(self._nodes)

  def get_node(self, history, done):
    """Returns the ExplorationNode(history, done) in the graph if it

        exists. Otherwise, creates the node, adds it to the graph and returns
        it.

        Args:
            history (History): see constructor of ExplorationNode
            done (bool): see constructor of ExplorationNode
    """
    return self._nodes.setdefault((history, done),
                                  ExplorationNode(history, done,
                                                  self._num_actions))

  def reachable_frontier(self, current_node):
    """Returns a path to a frontier ExplorationNode (node with unexplored

        actions) if one exists. Otherwise, returns None.

        Args:
            current_node (ExplorationNode): start of path

        Returns:
            path (list[(ExplorationNode, int)]) | None:
                [(current_node, action_to_take), (next_node, action_to_take),
                 ...], should lead to a frontier node
        """
    # Does a BFS for a frontier node
    bfs_queue = deque([PathNode(current_node, None, None)])  # deque[PathNode]
    visited = set()  # visited ExplorationNode
    visited.add(current_node)
    while len(bfs_queue) > 0:
      node = bfs_queue.popleft()
      if len(node.exploration_node.unexplored_actions) > 0:
        # Reverse the path
        path = []
        while node.action_taken is not None:
          prev_path_node = node.prev_path_node
          path.append((prev_path_node.exploration_node, node.action_taken))
          node = prev_path_node
        return list(reversed(path))

      actions = range(self._num_actions)
      np.random.shuffle(actions)
      for action in actions:
        next_node = node.exploration_node.next_node(action)
        if next_node not in visited:
          visited.add(next_node)
          bfs_queue.append(PathNode(next_node, action, node))

    return None


"""Past k states, in reverse time order:

e.g. [s_t, s_{t - 1}, ..., s_{t - k + 1}]
"""


class History(object):

  def __init__(self, states):
    """
        Args:
            states (list[FloatTensor | None]): the states in the correct order
    """
    self._states = deque(states)

  def update(self, new_state):
    """Updates from the current history: [s_t, ..., s_{t - k + 1}]

                                         to: [s_{t + 1}, ..., s_{t - k}]

        Args:
            new_state (FloatTensor): s_{t + 1}
    """
    self._states.appendleft(new_state)
    self._states.pop()

  # TODO: Probably should just implement __deepcopy__
  def clone(self):
    """Returns a deep copy of self"""
    return History(copy.deepcopy(self._states))

  def __getitem__(self, index):
    return self._states[index]

  def __hash__(self):
    return hash(
        tuple(
            np.sum(state) if state is not None else None
            for state in self._states))

  def __eq__(self, other):
    if not isinstance(other, History):
      return False

    for self_state, other_state in zip(self._states, other._states):
      if self_state is None and other_state is None:
        continue
      elif self_state is None and other_state is not None:
        return False
      elif other_state is None and self_state is not None:
        return False
      elif not np.allclose(self_state, other_state):
        return False

    return True

  def __str__(self):
    return "History({})".format([
        torch.sum(state) if state is not None else None
        for state in self._states
    ])

  __repr__ = __str__
