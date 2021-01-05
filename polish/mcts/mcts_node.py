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

"""This file implements tree node class for a Monte Carlo Tree Search (MCTS)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
from absl import logging
import gin
import numpy as np
from polish.utils import performance

np.set_printoptions(threshold=np.inf)
# For more elaborate description of variables, check:
#   https://www.gwern.net/docs/rl/2017-silver.pdf
# N(s, a): number of time action a is taken from state s
# W(s, a): total action value
# Q(s, a): mean action value
# P(s, a): prior probability of selecting action 'a'

# Change this value according to maximum value in your application.
VIRTUAL_LOSS_VALUE = 9000.0
ILLEGAL_MOVE_WEIGHT = 9000000.


class DummyNode(object):
  """A fake node of a MCTS search tree.

    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler.
  """

  def __init__(self):
    self.parent = None
    self.child_n = collections.defaultdict(float)
    self.child_w = collections.defaultdict(float)
    self.child_q = collections.defaultdict(float)
    self.child_reward = collections.defaultdict(float)

  def add_virtual_loss(self, up_to):
    pass

  def revert_virtual_loss(self, up_to):
    pass

  def backup_value(self, value, up_to):
    pass


@gin.configurable(
    blacklist=['num_steps', 'reward', 'move', 'parent', 'mask', 'done'])
class MCTSNode(object):
  """A node of a MCTS search tree.

    An MCTSNode stores the state representation in an OpenAI gym-like
      environment. A node can take an action in the environment using
      self.env.step.

      A node knows how to compute the action scores of all of its children,
      so that a decision can be made about which move to explore next. Upon
      selecting a move, a new MCTS node is added to the children dictionary.

    Attributes:
      state: The states of the environment that this node represent.
      max_num_actions: Maximum number of actions per each node.
      current_reward: The reward value from parent node to current node.
      move: A move (action index) that led to this state.
      parent: A parent MCTSNode.
      is_expanded: If True, this node is expanded (the children are evaluated).
      losses_applied: Use for virtual loss calculation. If True, the virtual
        loss is applied to this node.
      mask: An array indicating which actions are feasible
            (1: feasible, 0: invalid). If None, all the actions are feasible. If
              an array is given, the element of the array indicates which
              actions are feasible. Suppose the max_num_actions are 4. Then,
              mask = [2, 3] means second and third actions are feasible and
              zeroth and first actions are invalid.
      illegal_moves: An array indicating the illegal moves from the current
        node.
      child_n: An array indicating number of times each children is visited.
      child_w: An array indicating the state value for each children.
      child_reward: An array indicating the reward from current node to each of
        its children.
      original_prior: An array indicating the original probability for each
        children.
      child_prior: An array indicating the current probability for each children
        (after zeroing out the illegal moves).
      children: A dictionary with key representing the move and the value
        referencing the children MCTSNode.
      network_value: the value network result for the current node.
      done: Indicating whether this is a terminal node.
  """

  def __init__(self,
               state,
               observ,
               max_episode_steps=50,
               num_steps=0,
               dirichlet_noise_alpha=2.,
               dirichlet_noise_weight=0.35,
               mcts_value_discount=0.99,
               c_puct=3.,
               temperature=20,
               max_num_actions=128,
               reward=0,
               move=None,
               parent=None,
               mask=None,
               done=False,
               env_action_space=2,
               debug=False):
    """Initialize the MCTSNode attributes.

    MCTSNode does not need to instantiate BatchEnv, as it does not need to
    simulate environments in parallel.

    Args:
      state: The physics states of the environment that this node represent.
      observ: The environment observation of the current state.
      max_episode_steps:
      num_steps: number of steps taken so far
      dirichlet_noise_alpha: Concentrated-ness of the noise being injected into
        priors.
      dirichlet_noise_weight: How much to weight the priors vs. dirichlet noise
        when mixing.
      mcts_value_discount: The discount value for backup step in MCTS.
      c_puct: Exploration constant balancing priors vs. value net output.
      temperature: Exploration temperature for action selection during MCTS
        simulation.
      max_num_actions: Maximum number of actions per each node.
      reward: the reward of getting to this node from parent
      move: A move (action index) that led to this state.
      parent: A MCTSNode that is parent of this node.
      mask: An array indicating which actions are feasible
              (1: feasible, 0: invalid).
      done: Indicating whether this is a terminal node.
      env_action_space: the size of actions
      debug: If True, a set of prints are shown during the execution that may be
        used for debugging.
    """
    if parent is None:
      parent = DummyNode()

    # Parent node
    self.parent = parent

    self.max_num_actions = max_num_actions
    self.env_action_space = env_action_space
    self._max_episode_steps = max_episode_steps

    # Action that led to this position
    self.move = move

    self.num_steps = num_steps
    # State consists of [qpos, qvel]
    self.state = state
    self.observ = observ
    self.current_reward = reward

    # If True, this node's children have been explored.
    self.is_expanded = False

    # Number of virtual losses on this node (used for parallel MCTS)
    self.losses_applied = 0

    # Mask value for the current node (1: legal move, 0: illegal move)
    if mask is None:
      self.mask = np.ones([self.max_num_actions], dtype=np.float32)
    else:
      self.mask = np.zeros([self.max_num_actions], dtype=np.float32)
      self.mask[mask] = 1

    # An array which defines the illegal moves
    self.illegal_moves = 1 - np.asarray(self.mask)

    # N(s, a): Visit count
    self.child_n = np.zeros(self.max_num_actions, dtype=np.float32)
    # W(s, a): Total action value
    self.child_w = np.zeros(self.max_num_actions, dtype=np.float32)
    # R(s, a): Reward for the action
    self.child_reward = np.zeros(self.max_num_actions, dtype=np.float32)
    # Save a copy of the original prior before it gets mutated by d-noise.
    self.original_prior = np.zeros(self.max_num_actions, dtype=np.float32)
    # May be different from original_prior b/c of injecting noise
    self.child_prior = np.zeros(self.max_num_actions, dtype=np.float32)
    # Map of moves (action index) to the resulting MCTSNodes
    self.children = {}
    self.network_value = 0
    self.done = done

    # Private members
    self._dirichlet_noise_alpha = dirichlet_noise_alpha
    self._dirichlet_noise_weight = dirichlet_noise_weight
    self._mcts_value_discount = mcts_value_discount
    self._c_puct = c_puct
    self._temperature = temperature
    self._max_num_actions = max_num_actions
    # Map of moves (action index) to the actual action in the environment.
    self.move_to_action = {}
    # Map of moves (action index) to the children's observations
    self.move_to_observ = {}
    # Map of moves (action index) to the children's states
    self.move_to_state = {}
    # Map of moves (action index) to the children's done status
    self.move_to_done = {}

    self._debug = debug

  @property
  def child_action_score(self):
    """Calculate action score for all the children.

    Assign a big loss for illegal move so MCTS never chooses illegal moves
    In Gym environment, we may not need this mechanism.

    Returns:
      An array of action scores for all the children.
    """
    if self._debug:
      logging.info('=' * 64)
      logging.info('Move: %s', self.move)
      logging.info('.' * 64)
      logging.info('Child Reward: %s', self.child_reward)
      logging.info('Child W: %s', self.child_w)
      logging.info('Child N: %s', self.child_n)
      logging.info('Root N: %s', self.n)
      logging.info('Child Priors: %s', self.child_prior)
      logging.info('.' * 64)
      logging.info('Illegal moves: %s', self.illegal_moves)
      logging.info('Child Q: %s', self.child_q)
      logging.info('Child U: %s', self.child_u)
      logging.info('Child Action Score: %s', self.child_q + self.child_u)
      logging.info('=' * 64)
    return (self.child_q + self.child_u -
            ILLEGAL_MOVE_WEIGHT * self.illegal_moves)

  # For all the children
  @property
  def child_q(self):
    """Exploitation term in Polynomial Upper Confidence Trees (PUCT).

    Returns:
      exploitation scores for each children.
    """
    return self.child_reward + (
        self._mcts_value_discount * (self.child_w /
                                     (np.maximum(1, self.child_n))))

  # PUCT algorithm (for all the children)
  @property
  def child_u(self):
    """Exploration term in Polynomial Upper Confidence Trees (PUCT).

    The less we have tried an action, the greater U will be, which
    encourages `exploration`. Increasing c_puct puts more weight toward
    this exploration term.

    Returns:
      exploration scores for each children.
    """
    # return (self._c_puct * math.sqrt(max(1, self.n - 1)) * (self.child_prior /
    #                                                       (1 + self.child_n)))

    return (3 * self._c_puct * self.child_prior) * np.sqrt(
        math.log(self.n + 1) / (1 + self.child_n))

  @property
  def q(self):
    """Q(s, a).

    Returns:
      Mean action value for the edge from parent to the current node.
    """
    return self.parent.child_q[self.move]

  @property
  def n(self):
    """N(s, a).

    Returns:
      Visit count for the edge from parent to the current node
    """
    return self.parent.child_n[self.move]

  @n.setter
  def n(self, value):
    self.parent.child_n[self.move] = value

  @property
  def w(self):
    """W(s, a).

    Returns:
      Total action value for the edge from parent to the current node
    """
    return self.parent.child_w[self.move]

  @w.setter
  def w(self, value):
    self.parent.child_w[self.move] = value

  @property
  def reward(self):
    """R(s, a).

    Returns:
      Reward from environment for taking action a from state s
    """
    return self.parent.child_reward[self.move]

  @reward.setter
  def reward(self, value):
    self.parent.child_reward[self.move] = value

  def best_child(self):
    """sort by child_N tie break with action score.

    Returns:
      sorted args.
    """
    if self._debug:
      logging.info('Best Children: %s', self.child_n)
    sorted_args = np.argsort(-(self.child_n + self.child_action_score / 10000.))
    if self._debug:
      logging.info('Sorted Args: %s', sorted_args)
      logging.info('-' * 64)
    return sorted_args

  def select_leaf(self):
    """Select leaves from MCTS.

    Args: None

    Returns:
      A leaf node.
    """
    current = self
    while True:
      # if a node has never been evaluated (leaf node),
      # we have no basis to select a child.
      if (not current.is_expanded) or current.is_done():
        break

      with performance.timer('calculating action score', self._debug):
        action_scores = current.child_action_score
      with performance.timer('calculating argmax', self._debug):
        best_move = np.argmax(action_scores)
      with performance.timer('add child', self._debug):
        current = current.add_child_if_absent(best_move)

    return current

  def add_child_if_absent(self, move):
    """Adds child node for fcoord if not already exist, and returns it.

    Args:
      move: The index of an action to play

    Returns:
      The children node corresponding to action_index
    """
    if move not in self.children:
      child_state = self.move_to_state[move]
      child_observ = self.move_to_observ[move]
      child_reward = self.child_reward[move]
      child_done = self.move_to_done[move]

      self.children[move] = MCTSNode(
          state=child_state,
          observ=child_observ,
          max_episode_steps=self._max_episode_steps,
          num_steps=self.num_steps + 1,
          dirichlet_noise_alpha=self._dirichlet_noise_alpha,
          dirichlet_noise_weight=self._dirichlet_noise_weight,
          mcts_value_discount=self._mcts_value_discount,
          c_puct=self._c_puct,
          temperature=self._temperature,
          max_num_actions=self._max_num_actions,
          reward=child_reward,
          move=move,
          parent=self,
          # all the moves are valid in Gym
          mask=None,
          done=child_done,
          env_action_space=self.env_action_space)
    return self.children[move]

  def add_virtual_loss(self, up_to):
    """Propagate a virtual loss up to the root node.

    Args:
       up_to: The node to propagate until. (This is needed to reverse the
         virtual loss later.)
    """
    self.losses_applied += 1
    loss = -VIRTUAL_LOSS_VALUE
    self.w += loss
    if self.parent is None or self is up_to:
      return
    self.parent.add_virtual_loss(up_to)

  def revert_virtual_loss(self, up_to):
    """Reverting applied virtual loss for the trajectory (parallel MCTS).

    Args:
      up_to: revert virtual loss recursively up to this node

    Returns:
      None
    """
    self.losses_applied -= 1
    revert = VIRTUAL_LOSS_VALUE
    self.w += revert
    if self.parent is None or self is up_to:
      return
    self.parent.revert_virtual_loss(up_to)

  def incorporate_results(self, child_probs, node_value, up_to):
    """Backup prior probabilities and values backward to up_to MCTSNode.

    Args:
      child_probs: children move probabilities
      node_value: current node value
      up_to: up to a MCTSNode
    """
    # If a node was picked multiple times, we shouldn't
    # expand it more than once.
    if self.is_expanded:
      return

    self.is_expanded = True

    # `node_value` is the output of `value_network` for the current state
    self.network_value = node_value

    # Zero out illegal moves and rescale probs
    move_probs = np.asarray(child_probs)
    self.original_prior = move_probs
    # Normalize the child proabilities in order to make the coordinates summed
    # to one. This is important as we later inject Dirichlet noise to these
    # probabilities and we want to preserve this property of coordinates summed
    # to one and make it a VALID probability distribution.
    scale = np.sum(move_probs)
    if scale > 0:
      move_probs *= 1.0 / float(scale)

    self.child_prior = move_probs

    # Recursively backup the values up to the root node
    self.backup_value(self.network_value, up_to=up_to, first_call=True)

  def backup_value(self, value, up_to, first_call=True):
    """Propagates a value estimation up to the root node.

    Args:
         value: the value to be propagated
         up_to: the node to propagate until.
         first_call: True means that we have already updated the value of the
           current node. As such, no need to add it again.
    """
    self.n += 1
    if not first_call:
      self.w += value
    if self.parent is None or self is up_to:
      return
    self.parent.backup_value(
        value=self.reward + self._mcts_value_discount * value,
        up_to=up_to,
        first_call=False)

  def is_done(self):
    """True if the position is at a move greater than the max horizon.

    Args: None

    Returns:
      Boolean
    """
    # pylint: disable=protected-access
    if self.done or (self._max_episode_steps <= self.num_steps):
      return True
    else:
      return False

  def inject_noise(self):
    """Add noise to root to promote exploration.

    Add Dirichlet noise to the prior probabilities in the root node.

    https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5'

    The support of Dirichlet (i.e., the values where it is defined) is
    the set of vectors (x_1, x_2, ..., x_n) whose coordinates are nonnegative
    and sum to 1. If n=3, then some examples would be (0.1, 0.5, 0.4) and
    (0.3, 0, 0.7). This set is also called the (n-dimensional) simplex.

    Now note that Dir(1) is uniform. This means that sampling from it will
    produce either of the above two vectors (or any other valid vector)
    with equal chance. As `alpha` gets smaller, Dirichlet starts to
    prefer vectors near the basis vectors: (0.98, 0.01, 0.01) or (0.1, 0.9, 0)
    will be more likely to be drawn than (0.3, 0.3, 0.4).
    For values greater than 1, the opposite is true - the basis vectors
    are de-emphasized, and more-balanced vectors are preferred.

    (1) When `alpha` > 1, this tends to `flatten` p, making us
    less strongly prefer any one move.
    (2) For `alpha` < 1, it will cause a random component to become more
    emphasized.

    In both cases, MCTSPlayer is encourged to explore AWAY from the prior.

    It is suggested to choose `alpha` = 10 / (number of moves).
    However, in general `alpha` = 1 seems a reasonable value.
    """
    a = [self._dirichlet_noise_alpha] * (self.max_num_actions)
    dirichlet = np.random.dirichlet(a)
    if self._debug:
      logging.info('=' * 64)
      logging.info('Dirichlet Noise Alpha: %s', a)
      logging.info('Dirichlet Noise Value: %s', dirichlet)
      logging.info('Child Prior BEFORE Injecting Noise: %s', self.child_prior)
    self.child_prior = (
        self.child_prior * (1 - self._dirichlet_noise_weight) +
        dirichlet * self._dirichlet_noise_weight)
    if self._debug:
      logging.info('Child Prior AFTER Injecting Noise: %s', self.child_prior)
      logging.info('=' * 64)

  def get_children_probs(self, is_beginning=True):
    """Returns the child visit counts as a probability distribution (pi).

    Args:
        is_beginning: If true, exponentiate the probabilities by a temperature
          slightly larger than unity to encourage diversity in early play and
          normalize the visit count so they look like probability.
    """
    probs = self.child_n
    if is_beginning:  # beginning of the game
      probs = probs**.98
      sum_probs = np.sum(probs)
      if sum_probs == 0:
        return probs
      return probs.astype(float) / sum_probs.astype(float)
    else:
      return probs.astype(float)
