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

"""Monte Carlo Tree Search (MCTS) player implementation."""
import random
import time
from absl import logging
import gin
import numpy as np
from scipy.stats import multivariate_normal
from polish.mcts import mcts_node as m_node
from polish.utils import performance


@gin.configurable(denylist=['call_policy'])
class MCTSPlayer(object):
  """Monte Carlo Tree Search (MCTS) player class.

  This class is the core of performing Monte Carlo Tree Search. The tree search
  consists of performing the following steps:
  (1) Suggest a move. That involves multiple iterations of MCTS simulation
      and then suggesting a preferable move based on the collected statistics.
  (2) Play the suggested move on the environment.

  Attributes:
    tree_env: The environment used for MCTS simulations (batched).
    max_num_actions: Maximum number of actions for the tree search (specifies
      the branching factor for each node in the tree). Usually this value should
      be equal to the maximum possible actions for each environment. For the
      environments with large action space (or continuous), you may need to
      limit this value.
    num_mcts_sim: Number of MCTS simulations to be performed before suggesting
      an action.
    timed_match: If set, the tree search is performed in a `timed` manner. That
      is, after a predefined limit defined in seconds (`seconds_per_move`), the
      MCTS simulations ends, regardless of the number of MCTS simulations.
    seconds_per_move: If `timed_match` is enabled, this indicates the maximum
      seconds to be spent per action.
    temp_threshold: Defines the `threshold` term in MCTS. The moves before
      reaching this limit are performed non-deterministically. After that, all
      the moves are chosen deterministically. That is, the moves with higher
      number of visits are preferred.
    root: Root node in the MCTS tree. The root node gets updated after taking
      the `suggested` move.
    move_counts: Counts the number of moves taken in the environment so far.
    init_observ: Initial observation for the tree environment.
    init_state: Initial state for the tree environment.
    game_observs: An array that holds the observations that have been observed
      in the player environment (for the suggested moves).
    game_rewards: An array that holds the rewards for the actions taken on the
      player environment (for the suggested moves).
    game_actions: An array that holds the actions taken on the player
      environment (for the suggested moves).
    game_probs: An array that holds the probability associated with the
      observations in the player environment.
    game_values: An array that holds the state-values associated with the
      observations in the player environment.
    game_dones: An array that holds the status (done/not done) associated with
      the observations in the player environment.
    game_means: An array that holds the estimated mean of the tree policy
      distribution given a state. The estimated mean is computed for all the
      states in the trajectory. Currently, the estimated mean is computed as the
      weighted average of the normalized number of visit counts of a node's
      children.
    game_logstd: An array that holds the log of estimated standard deviation of
      the tree policy distribution given a state. Similar to `game_means`, the
      estimated standard deviation is computed according to the normalized
      number of visit counts of a node's children.
  """

  def __init__(self,
               tree_env,
               max_episode_steps=0,
               call_policy=None,
               env_action_space=2,
               max_q_enable=True,
               max_num_actions=128,
               num_mcts_sim=0,
               timed_match=False,
               seconds_per_move=5,
               temperature=5,
               parallel_readouts=1,
               num_envs=32,
               num_moves_per_search=10,
               build_dist_enable=False,
               debug=False):
    """Creats an MCTS player.

    Args:
      tree_env: The environment used for MCTS simulations.
      max_episode_steps: Defines the maximum length of the episodes for a given
        environment. We use this for the environments in which the number of
        steps defines the episode length. That is, after taking this numnber of
        steps, the environment sets `done` to be true.
      call_policy: A function which calls the tf.estimator in prediction mode to
        retrieve values from policy/value network.
      env_action_space: A scalar defining the size (first dimension) of the
        action space.
      max_q_enable: If set, state value of a node is equal to the max Q value of
        its children.
      max_num_actions: Maximum number of actions for the tree search (specifies
        the branching factor for each node in the tree). Usually this value
        should be equal to the maximum possible actions for each environment.
        For the environments with large action space (or continuous), you may
        need to limit this value.
      num_mcts_sim: If `timed_match` is enabled, this indicates the maximum
        seconds to be spent per action.
      timed_match: If set, the tree search is performed in a `timed` manner.
        That is, after a predefined limit defined in seconds
        (`seconds_per_move`), the MCTS simulations end, regardless of the number
        of MCTS simulations.
      seconds_per_move: If `timed_match` is enabled, this indicates the maximum
        seconds to be spent per action.
      temperature: Defines the `threshold` term in MCTS. The moves before
        reaching this limit are performed non-deterministically. After that, all
        the moves are chosen deterministically. That is, the moves with higher
        number of visits are preferred.
      parallel_readouts: Specifies the number of simultaneous threads/processes
        that perform tree search. Currently, we only support value one. That is,
        in each tree only one process performs search.
      num_envs: Number of environments used for tree search. This value
        basically defines the number of parallel environments in `tree_env`.
      num_moves_per_search: Defines how many moves to choose after a set of MCTS
        simulations are performed.
      build_dist_enable: if True, we build a distribution around the selected
        action by MCTS and find log of probability distribution function (pdf).
      debug: If True, it makes a set of debugging texts visible that are helpful
        to debug the code.
    """
    if max_num_actions < num_envs:
      raise ValueError('max_num_actions should be greater than or '
                       'equal to num_envs.')

    self.tree_env = tree_env

    self._max_episode_steps = max_episode_steps
    self._call_policy = call_policy
    self._env_action_space = env_action_space
    self._max_q_enable = max_q_enable
    self.max_num_actions = max_num_actions
    self.num_mcts_sim = num_mcts_sim
    self.timed_match = timed_match
    self.seconds_per_move = seconds_per_move
    self.temp_threshold = temperature
    self._parallel_readouts = parallel_readouts
    self._num_envs = num_envs
    self._num_moves_per_search = num_moves_per_search
    self._build_dist_enable = build_dist_enable
    self._debug = debug

    self.root = None
    self.move_counts = 0
    self.init_observ = None
    self.init_state = None
    self._num_searches = 0

    self.game_observs = []
    self.game_rewards = []
    self.game_actions = []
    self.game_probs = []
    self.game_values = []
    self.game_dones = []
    self.game_means = []
    self.game_logstd = []

    self.initialize_game()

  def initialize_game(self, init_state=None):
    """Initialize a MCTS tree search.

    Args:
      init_state: start MCTS from init_state.

      (1) Put all the parallel environments in `tree_env` (batched)
          at the same state.
      (2) Creates a MCTS node as the root of the tree.
    """
    # initialize other members
    self.move_counts = 0
    self._num_searches = 0
    self.game_observs = []
    self.game_rewards = []
    self.game_actions = []
    self.game_probs = []
    self.game_values = []
    self.game_dones = []
    self.game_means = []
    self.game_logstd = []

    if init_state is None:
      self.init_observ = self.tree_env[0].reset()
      self.init_state = self.tree_env[0].sim.get_state()
      for env in self.tree_env:
        env.set_state(self.init_state.qpos, self.init_state.qvel)
    else:
      self.init_state = init_state
      for env in self.tree_env:
        env.set_state(self.init_state.qpos, self.init_state.qvel)
      # pylint: disable=protected-access
      self.init_observ = self.tree_env[0].unwrapped._get_obs()

    # Creates the root node in MCTS player
    self.root = m_node.MCTSNode(
        state=self.init_state,
        observ=self.init_observ,
        max_episode_steps=self._max_episode_steps,
        max_num_actions=self.max_num_actions,
        env_action_space=self._env_action_space,
        temperature=self.temp_threshold)

  def sample_actions(self, mcts_dist):
    """Sample a set of actions per node.

    The sampled actions become the children of the given node.

    Args:
      mcts_dist: This distribution is built using the RL Policy (PPO) by passing
        the state of the current node to the policy network.

    Returns:
      A set of actions sampled from the policy distribution.
    """
    with performance.timer('sample multivariate normal distribution',
                           self._debug):
      sampled_actions = mcts_dist.rvs(size=self.max_num_actions)
    return sampled_actions

  def suggest_move(self):
    """Suggest a single action after performing a number of MCTS simulations.

    Args: None

    Returns:
      An action to be taken in the environment.
    """
    start = time.time()

    if self.timed_match:
      while time.time() - start < self.seconds_per_move:
        self.tree_search()
    else:
      current_readouts = self.root.n
      with performance.timer('Searched %d' % self.num_mcts_sim, True):
        if (self._num_searches %
            self._num_moves_per_search == 0) or (not bool(self.root.children)):
          while self.root.n < current_readouts + self.num_mcts_sim:
            # perform MCTS searches in the following two scenarios:
            # (1) we've already taken the designated number of moves per search
            # (2) or, the root does not have any children
            self.tree_search()
    # Increments the number of searches performed so far.
    self._num_searches += 1
    # Retrieve the suggested move based on the collected statistics in the tree.
    suggested_move = self.pick_move()
    return suggested_move

  def play_move(self, move):
    """Play the suggested action in the player environment.

      Notable side effects:

      (1) finalizes the probability distribution according to
          this roots visit counts.
      (2) Makes the node associated with this move the root, for future
            `inject_noise` calls.

    Args:
      move: This is an index which maps into the actual action to be taken in
        the environment.

    Returns:
      value, action probability, done, reward (after taking the action)
    """
    # Map the move index to an actual action in the environment.
    action = self.root.move_to_action[move]
    if self._debug:
      logging.info('=' * 64)
      logging.info('Played action index: %s (%s).', move, action)
      logging.info('=' * 64)
    # Note that, we have already stored the next observation, next state,
    #   reward, and status (done/not done) in each tree node.
    #   As such, instead of calling the environment again, we just retrieve
    #   these values from the corresponding arrays.
    next_observ = self.root.children[move].observ
    next_state = self.root.children[move].state
    reward = self.root.child_reward[move]
    done = self.root.children[move].is_done()

    # Reset all the parallel environments in the tree_env and set the
    # state of all the environments to the next state.
    for env in self.tree_env:
      env.reset()
      env.set_state(next_state[0], next_state[1])

    # Compute the children probabilities and update probability
    # of the action taken on the environment. The visit counts are normalized.
    normalized_children_visits = self.root.get_children_probs(is_beginning=True)
    normalized_count = normalized_children_visits[move]
    # Compute the state-value of the state (after taking the action)
    value = self.root.child_w[move]
    # Add the new state to the state buffer.
    self.game_observs.append(next_observ)
    # Add the action to the action buffer.
    self.game_actions.append(action)
    # Add the reward to the reward buffer.
    self.game_rewards.append(reward)

    action_probs = (self.root.child_n + 1.) / float(
        np.sum(self.root.child_n + 1.))
    weighted_action_probs = [
        np.multiply(act, prob)
        for act, prob in zip(self.root.move_to_action.values(), action_probs)
    ]
    estimator_mean = np.sum(weighted_action_probs, axis=0)
    self.game_means.append(estimator_mean)

    actions_squared = np.power(
        list(self.root.move_to_action.values()) - estimator_mean, 2.0)
    weighted_actions_squared = [
        np.multiply(act2, prob)
        for act2, prob in zip(actions_squared, action_probs)
    ]
    estimator_var = np.sum(weighted_actions_squared, axis=0)
    self.game_logstd.append(np.log(np.sqrt(estimator_var)))

    # Create a MultivariateNormal Distribution and find the probability
    # of the selected action.
    mcts_dist = multivariate_normal(
        mean=estimator_mean, cov=np.diag(estimator_var))
    self.game_probs.append(-mcts_dist.logpdf(action))

    # Find the state-value of the node.
    if self._max_q_enable:
      self.game_values.append(np.max(self.root.child_q))
    else:
      self.game_values.append(self.root.network_value)

    # Add the status of the new state (done/not-done) to the done buffer.
    self.game_dones.append(done)

    # Update root and increment the number of moves.
    self.root = self.root.children[move]
    self.move_counts += 1

    # After taking the action, you do not need the other nodes in the tree,
    # simply delete them
    del self.root.parent.children

    return value, normalized_count, done, reward

  def pick_move(self):
    """Picks a move to play, based on MCTS simulations statistics.

      Highest N is most robust indicator.
      In the early stage of the game (less than temperature), pick
        a move weighted by visit count; later on, pick the absolute max.

    Returns:
      The move (an index) to be taken in the environment. We still need to map
      this move to the actual action that can be taken in the environment.
    """
    # Deterministic action selection after number of taken actions are greater
    # than temperature.
    if self.move_counts >= self.temp_threshold:
      while True:
        sorted_moves = self.root.best_child()
        # If there is no children, we do not have a basis to select a children.
        # That is, we have not collected any statistics for this node.
        if not bool(self.root.children):
          return -1
        for m in sorted_moves:
          if m in self.root.children:
            return m
    else:
      # Non-deterministic action selection before reaching to
      # the temperature threshold.
      while True:
        # If there are no children, we do not have a basis to select a child.
        # That is, we have not collected any statistics for this node.
        if not bool(self.root.children):
          return -1
        cdf = self.root.get_children_probs(is_beginning=True).cumsum()
        selection = random.random()
        move = cdf.searchsorted(selection)
        if move in self.root.children:
          return move

  def tree_search(self, parallel_readouts=None):
    """Main tree search (MCTS simulations).

    Args:
      parallel_readouts: Number of parallel searches (threads) that are
        performed in the tree.

    Returns:
      Selected leaf node.
    """
    if parallel_readouts is None:
      parallel_readouts = self._parallel_readouts

    # The array that holds the selected leaf nodes for expand and evaluate
    #   If parallel_readouts is one, len(leaves) = 1
    leaves = []
    failsafe = 0
    while len(leaves) < parallel_readouts and failsafe < parallel_readouts * 2:

      failsafe += 1

      with performance.timer('select a leaf', self._debug):
        # Select a leaf for expansion in the tree
        leaf = self.root.select_leaf()

      # If this is a terminal node, directly call backup_value
      # We pass `zero` as the backup value on this node. The meaning of this
      # backup value is that `the expected total reward for an agent starting
      # from this terminal node is ZERO.` You may need to change this based
      # on the target environment.
      if leaf.is_done():
        # Everything should be a tensor, as we are working in a batched mode.
        policy_out = self._call_policy(
            np.asarray([leaf.observ]), only_normalized=True)
        leaf.network_value = policy_out['value'][0]
        leaf.backup_value(value=0, up_to=self.root)
        continue

      # Append the leaf to the list for further evaluation.
      leaves.append(leaf)

    if parallel_readouts == 1:
      assert (len(leaves) <= 1), 'Only select one leaf or less!'

    # Calculate the state-value and probabilities of the children
    if leaves:
      # Calls the policy on the leaf`s observation (NOT STATE) and retrieves
      # the `value` for these observations from the value network.
      with performance.timer('call policy', self._debug):
        policy_out = self._call_policy(
            np.asarray([leaves[0].observ]), only_normalized=True)
      mean = policy_out['mean'][0]
      logstd = policy_out['logstd'][0]
      value = policy_out['value'][0]
      leaves[0].network_value = value
      with performance.timer('create multivariate normal distribution',
                             self._debug):
        # Based on the returned policy (mean and logstd), we create
        # a multivariate normal distribution. This distribution is later used
        # to sample actions from the environment.
        mcts_dist = multivariate_normal(
            mean=mean, cov=np.diag(np.power(np.exp(logstd), 2)))
      sampled_actions = self.sample_actions(mcts_dist=mcts_dist)
      child_probs = mcts_dist.pdf(sampled_actions)
      # Update `move_to_action` for the selected leaf
      for i, a in enumerate(sampled_actions):
        leaves[0].move_to_action[i] = a

      # In case the number of parallel environments are not sufficient,
      # we process the actions in chunks. Since the last chunk may have fewer
      # elements, we pad it so as to all chunks have the same size.
      # This restrication is necessary bc of BatchEnv environment.
      first_iteration = True
      child_reward = np.zeros(0)
      child_observ = np.zeros(0)
      child_state_qpos = np.zeros(0)
      child_state_qvel = np.zeros(0)
      child_done = np.zeros(0)

      for mcts_env, mcts_action in zip(self.tree_env, sampled_actions):
        mcts_env.reset()
        mcts_env.set_state(leaves[0].state[0], leaves[0].state[1])
        observ, reward, done, _ = mcts_env.step(mcts_action)
        state = mcts_env.sim.get_state()

        if first_iteration:
          child_reward = np.array([reward])
          child_observ = np.array([observ])
          child_state_qpos = np.array([state.qpos])
          child_state_qvel = np.array([state.qvel])
          child_done = np.array([done])
          first_iteration = False
        else:
          child_reward = np.concatenate((child_reward, np.array([reward])))
          child_observ = np.concatenate((child_observ, [observ]))
          child_state_qpos = np.concatenate((child_state_qpos, [state.qpos]))
          child_state_qvel = np.concatenate((child_state_qvel, [state.qvel]))
          child_done = np.concatenate((child_done, np.array([done])))

      # Updates the rewards/observs/states for the selected leaf's children and
      # performs backup step.
      leaves[0].child_reward = child_reward[:self.max_num_actions]
      leaves[0].move_to_observ = child_observ[:self.max_num_actions]
      leaves[0].move_to_state = [(qpos, qvel) for qpos, qvel in zip(
          child_state_qpos[:self.max_num_actions],
          child_state_qvel[:self.max_num_actions])]
      leaves[0].move_to_done = child_done[:self.max_num_actions]

      # Update the values for all the children by calling the value network.
      # We set `only_normalized` to True as we only need to normalize the
      # observations whenever we call the policy/value network without updating
      # the running mean and standard deviation. We only update running mean
      # and standard deviation for observations in the trajectory.
      # Note that, this is a design decision and is based on the intuition
      # that the MCTS search may visit some states that may not be `good`.
      # Using this approach, we avoid updating the running mean/std for
      # the observations that may not be good.
      network_children = self._call_policy(
          leaves[0].move_to_observ, only_normalized=True)
      leaves[0].child_w = network_children['value']

      with performance.timer('incorporating rewards', self._debug):
        leaves[0].incorporate_results(
            child_probs=child_probs, node_value=value, up_to=self.root)
    return leaves

  def is_done(self):
    """Returns the state (done/not done) of the root node."""
    return self.root.is_done()

  def print_tree(self):
    """Print the Monte-Carlo tree (only connection between the nodes)."""
    nodes = []
    nodes.append(self.root)
    # pylint: disable=g-explicit-length-test
    while len(nodes) != 0:
      print('Node: ', nodes[0].move)
      for _, value in nodes[0].children.items():
        print('Children: ', value.move)
        nodes.append(value)
      print('-' * 64)
      del nodes[0]
