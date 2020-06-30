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
"""Code for running ConQUR training.

Refer to the paper ConQUR: Mitigating Delusional Bias in Deep Q-learning
(https://arxiv.org/abs/2002.12399) for more details.
"""
import collections
import os
import queue

from absl import app
from absl import flags
from dopamine.discrete_domains import checkpointer
import numpy as np
import scipy as sp
import tensorflow.compat.v1 as tf

from conqur import atari

flags.DEFINE_string("atari_roms_source", None,
                    "Full path to atari roms source.")
flags.DEFINE_string("atari_roms_path", None, "Full path to atari roms path.")
flags.DEFINE_multi_string("gin_files", None, "Path to gin configuration files.")
flags.DEFINE_multi_string(
    "gin_bindings", None,
    "Comma-separated gin bindings to override the config files.")

flags.DEFINE_float(
    "sampling_temperature", 1.0,
    "Boltzmann temperature parameter for generating random "
    "action assignments.")
flags.DEFINE_bool("sample_consis_buffer", False,
                  "Whether to sample transitions from the consistency buffer.")
flags.DEFINE_integer("training_iterations", 100, "Iterations to train for.")
flags.DEFINE_string("atari_game", None,
                    "Specify atari environment if env=\'atari\'.")
flags.DEFINE_integer("random_seed", 0, "Random seed.")
flags.DEFINE_string("tfboard_path", "./", "Tensorboard root path.")
flags.DEFINE_string("checkpoint_path", "./",
                    "Checkpoint path for loading and saving models.")
flags.DEFINE_integer(
    "max_num_nodes", 8,
    "Maxmimum number of nodes, or information sets to maintain.")
flags.DEFINE_integer("num_children", 2,
                     "Number of children to produce when splitting a node.")

flags.DEFINE_integer("batch_size", 32, "Mini batch size for gradient update.")
flags.DEFINE_float("learning_rate", 0.0000025, "Learning rate.")
flags.DEFINE_float("consistency_coeff", 1.0,
                   "Consistency penalization coefficient.")

flags.DEFINE_float("train_eps", 0.01, "Training exploration probability.")
flags.DEFINE_float("eval_eps", 0.001, "Evaluation exploration probability.")
flags.DEFINE_integer(
    "target_replacement_freq", 5,
    "Frequency (number of iterations) at which to replace the target network.")
flags.DEFINE_integer(
    "rollout_freq", 10,
    "Rate (number of iterations) at which to do rollout evaluations.")
flags.DEFINE_integer("num_best_nodes_for_expansion", 2,
                     "Number of best nodes to select for children expansion.")
flags.DEFINE_integer(
    "split_delay", -1,
    "Number of iterations to dive before splitting into children nodes.")

flags.DEFINE_integer("train_step_frequency", 10,
                     "Steps to take before starting one train operation.")
flags.DEFINE_integer("steps_per_iteration", 10000,
                     "Environment steps to collect in each iteration.")
flags.DEFINE_bool(
    "no_op", True,
    "Whether to take no-op actions for up to 30 steps before following the "
    "policy.")
flags.DEFINE_enum(
    "node_scoring_fn", "bellman_consistency",
    ["rollouts", "bellman_consistency"],
    "Scores the existing nodes to select the nodes to expand "
    "children (\"rollouts\": score policies by their rollout "
    "evaluations, \"bellman_consistency\": score policies by "
    "their Bellman error plus their consistency penalty.")
flags.DEFINE_integer("back_track_size", 30,
                     "Number of nodes to maintain at the backtracking queue.")
flags.DEFINE_bool("enable_back_tracking", True,
                  "Whether to enable the tree search back-tracking algorithm.")
flags.DEFINE_float(
    "consistency_buffer_fraction", 1.0,
    "Fraction of steps_per_iteration transitions to store on the buffer.")

FLAGS = flags.FLAGS

BACK_TRACK_CALIBRATION = 2.5
BOLTZMANN_PROBABILITY_OFFSET = 0.02
CHECKPOINT_FREQUENCY = 5
EPS_TOLERANCE = 1e-5
FORGET_LEVEL = 3

CheckpointData = collections.namedtuple("CheckpointData", [
    "batch_feature_vecs",
    "target_weight",
    "tree_exp_replay",
    "pre_load",
    "node_queue",
    "tree_level_size",
    "weights_to_be_rollout",
    "best_sampler",
    "back_tracking_count",
    "back_tracking_node_list",
    "start_i",
    "optimizer",
    "initial_batch_data",
])


class ConsistencyBuffer:
  """Data structure for storing experience and enforcing consistency.

  Example usage of the ConsistencyBuffer:
  1. Call get_foremost_indices
  2. Use the indices to call store_transition
  3. After storing the transition, call get_experience by passing in a
     list of indices (tracing the tree)
  4. If it reaches the end of the loop, call next_level to advance
     into next level.
  """

  def __init__(self, random_state, args_dict):
    self.replay = {}
    self.replay_ordering = []
    self.random_state = random_state
    self.args_dict = args_dict

  def store_transition(self, current_index, batch, parent_ls):
    """Stores a batch of transitions.

    Args:
      current_index: int, the index(id) of the current batch.
      batch: list, the list of experience to be store (for Consistency Penalty).
      parent_ls: list, the list of parent index (id).
    """
    if (self.args_dict["sample_consis_buffer"] or len(parent_ls) == 1):
      self.replay[current_index] = batch
      self.add_index(current_index)
      return

    parent_index = parent_ls[-1]
    # Most recent batch
    next_state, assigned_action, action = batch

    if parent_index in self.replay:
      # Batches stored on the buffer already
      (stored_next_state, stored_assigned_action,
       stored_action) = self.replay[parent_index]

      if FLAGS.consistency_buffer_fraction < 1.0:
        # NOTE: By sampling a prior fraction of transitions, we do not end up
        # with a shrinking buffer -- we converge to a constant size because
        # we are summing across a geometric series.
        mini_sample_size = int(self.args_dict["steps_per_iteration"] *
                               FLAGS.consistency_buffer_fraction)
        sampled_indices = self.random_state.choice(
            range(len(stored_next_state)),
            min(mini_sample_size, len(stored_next_state)))
        stored_next_state = stored_next_state[sampled_indices]
        stored_assigned_action = stored_assigned_action[sampled_indices]
        stored_action = stored_action[sampled_indices]

      next_state = np.concatenate((next_state, stored_next_state), axis=0)
      assigned_action = np.concatenate(
          (assigned_action, stored_assigned_action), axis=0)
      action = np.concatenate((action, stored_action), axis=0)

    self.replay[current_index] = (next_state, assigned_action, action)
    self.add_index(current_index)

  def get_foremost_indices(self):
    if not self.replay_ordering:
      self.next_level()
      return -1
    if not self.replay_ordering[-1]:
      # Return previous level indices
      return self.replay_ordering[-2][-1]
    elif self.replay_ordering:
      return self.replay_ordering[-1][-1]

  def next_level(self):
    self.replay_ordering.append([])

  def add_index(self, index):
    self.replay_ordering[-1].append(index)

  def forget(self):
    """Forget some of the older experience."""
    if self.get_numbers_of_level() >= FORGET_LEVEL:
      forget_list = self.replay_ordering[-FORGET_LEVEL]
      for forget_index in forget_list:
        del self.replay[forget_index]

  def get_experience(self, ls):
    return self.replay[ls[-1]]

  def get_numbers_of_level(self):
    return len(self.replay_ordering)


def loss_fn(next_states, states, actions, q_network,
            last_batch_action_assignment, q_labels, args_dict):
  """Defines the training loss.

  Args:
    next_states: np.ndarray, the next states in feature space.
    states: np.ndarray, the current states in feature space.
    actions: np.ndarray, the current batch actions taken.
    q_network: tf.keras.layer, fully-connected layer.
    last_batch_action_assignment: np.ndarray, the previous batches next actions
      assignments.
    q_labels: np.ndarray, the q-labels.
    args_dict: dictionary, contains all the necessary configuration params.

  Returns:
    gradient: tf.keras.backend.gradients, the gradient of the loss
    main_loss: numpy array, the bellman loss
    regularized_loss: numpy array, the regularization loss
  """
  q_labels = np.expand_dims(q_labels, axis=1)
  q_predictions = tf.gather(q_network(states), actions, batch_dims=-1)

  main_loss = tf.losses.mean_squared_error(q_labels, q_predictions)
  if abs(args_dict["consistency_coeff"]) <= EPS_TOLERANCE:
    return main_loss, 0.0, 0.0, 0.0

  online_q_next_state_action = q_network(next_states)

  # Replicating this num_actions times.
  last_batch_action_assignment = np.expand_dims(
      last_batch_action_assignment, axis=1)
  multiple_online_q = -1.0 * tf.gather(
      online_q_next_state_action, last_batch_action_assignment,
      batch_dims=-1) * tf.ones(
          [1, len(last_batch_action_assignment), args_dict["num_actions"]])

  # Repeat the chosen-Q actions by creating "num_actions" times of it
  other_actions_q = tf.reshape(online_q_next_state_action, (-1, 1))
  multiple_online_q = tf.reshape(
      multiple_online_q,
      (multiple_online_q.shape[1] * multiple_online_q.shape[2], 1))
  regularized_loss = multiple_online_q + other_actions_q

  regularized_loss = regularized_loss * tf.to_float(
      tf.greater(regularized_loss, 0))
  regularized_loss = tf.reduce_mean(
      regularized_loss) * args_dict["consistency_coeff"]
  return (
      main_loss +
      regularized_loss), main_loss.numpy(), regularized_loss.numpy(), np.mean(
          q_predictions.numpy())


def grad_loss_fn(states, actions, children_layer, next_states, q_labels,
                 last_batch_action_assignment, num_children, args_dict):
  """Defines gradient of bellman loss plus consistency penalization."""
  with tf.GradientTape() as tape:
    loss_value, bellman_loss, regularized_loss, q_average = loss_fn(
        next_states, states, actions, children_layer,
        last_batch_action_assignment, q_labels[:, num_children], args_dict)
    return (tape.gradient(loss_value, children_layer.variables),
            loss_value.numpy(), bellman_loss, regularized_loss, q_average)


# TODO(dijiasu): Rewrite this function.
def extract_sa_features(batch, batch_feature_vecs, batch_action_vecs):
  """Extract state-action features."""
  batch_feature_vecs[:len(batch), :] = 0
  for i, (state, _, action, _, _, _, _) in enumerate(batch):
    batch_feature_vecs[i] = state[:-1]
    batch_action_vecs[i] = action


# TODO(dijiasu): Rewrite this function.
def process_initial_batch(initial_batch, args_dict):
  """Process an initial batch of transitions."""
  batch_state = np.zeros((len(initial_batch), args_dict["state_dimensions"]),
                         dtype="f8")
  for i, (state, _, _, _, _, _, _) in enumerate(initial_batch):
    batch_state[i] = np.expand_dims(state[:args_dict["state_dimensions"]], 0)
  return batch_state


def create_single_layer_from_weights(num_actions, state_dimensions,
                                     weights_vector):
  """Creates a Keras layer.

  Args:
    num_actions: int, number of actions for this env
    state_dimensions: int, state dimension
    weights_vector: list, two elements, 1st elmn is weight, 2nd elmn is bias

  Returns:
    keras.layer object with the params determined by the weights_vector
  """
  if isinstance(weights_vector[0]) == tuple:
    weights_vector[0] = weights_vector[0][0]
  if isinstance(weights_vector[1]) == tuple:
    weights_vector[1] = weights_vector[1][0]

  single_layer = tf.keras.Sequential()
  single_layer.add(
      tf.keras.layers.Dense(num_actions, input_dim=state_dimensions))
  single_layer.layers[0].set_weights(weights_vector)
  return single_layer


def sample_action_assignments(q_values, sampling_temperature, num_children):
  """Samples action from Boltzmann distribution.

  If the largest action probability is too small or too large, clip it to
  a reasonable interval.

  Args:
    q_values: np.ndarray, q-values.
    sampling_temperature: float, Boltzmann temperature.
    num_children: int, number of children nodes.

  Returns:
    np.ndarray of sampled actions.
  """
  logits = q_values / sampling_temperature
  logits -= logits.max()  # Normalize the logits
  np.clip(logits, -20, None, out=logits)
  action_probabilities = sp.special.softmax(logits)

  # Clip the max probability, so it lies in [lower_bound, upper_bound]
  uniform_probability = 1.0 / logits.size
  max_probability = action_probabilities.max()
  lower_bound = uniform_probability + BOLTZMANN_PROBABILITY_OFFSET
  upper_bound = 1 - BOLTZMANN_PROBABILITY_OFFSET
  below_lower_bound = (max_probability < lower_bound)
  above_upper_bound = (max_probability > upper_bound)
  if below_lower_bound or above_upper_bound:
    non_max_sums = np.exp(logits).sum() - 1
    max_probability_index = action_probabilities.argmax()
    # This is the logit offset needed to clip the maximum probability.
    logits += np.log(
        (1 / (lower_bound if below_lower_bound else upper_bound) - 1) /
        non_max_sums)
    logits[0, max_probability_index] = 0
  return tf.random.categorical(
      logits, num_children, seed=FLAGS.random_seed).numpy()


def process_batch_data(batch, split, level, iteration, args_dict):
  """Post-processing the target input batch.

  Args:
    batch: np.ndarray, input batch of data.
    split: bool, whether we split nodes.
    level: int, the level used in the ConsistencyBuffer.
    iteration: int, the training iteration.
    args_dict: dictionary, contains all the necessary configuration params.

  Returns:
    next_states: np.ndarray, a batch of next states.
    batch_q: np.ndarray, a batch of q-labels.
    batch_q_seq: np.ndarray, the q-values of all actions of the current batch.
    children_q_sampled: np.ndarray, the q-values of the all action of
      children batch.
    children_action_vec: np.ndarray, the action sampled for the child nodes.
    batch_q_seq_no_reward: np.ndarray, the q-values only (without reward)
      for all actions.
  """
  batch_q = np.zeros((len(batch),), dtype="f8")
  batch_q_seq = np.zeros((len(batch), args_dict["num_actions"]), dtype="f8")
  batch_q_seq_no_reward = np.zeros((len(batch), args_dict["num_actions"]),
                                   dtype="f8")
  children_q_sampled = np.zeros((len(batch), args_dict["num_children"]),
                                dtype="f8")
  children_action_vec = np.zeros((len(batch), args_dict["num_children"]),
                                 dtype="int")

  next_states = np.zeros((len(batch), args_dict["state_dimensions"]),
                         dtype="f8")

  # Unroll the data batches
  for (i, (_, s_next, _, _, q_target, q_target_seq,
           q_target_seq_no_reward)) in enumerate(batch):
    q_target = q_target[level]
    q_target_seq = q_target_seq[level]
    q_target_seq_no_reward = q_target_seq_no_reward[level]
    next_states[i] = s_next
    batch_q[i] = q_target
    batch_q_seq[i, :] = q_target_seq.numpy()
    batch_q_seq_no_reward[i, :] = q_target_seq_no_reward.numpy()
    # TODO(dijiasu): Try to remove this EPS_TOLERANCE branch.
    if abs(args_dict["consistency_coeff"]) <= EPS_TOLERANCE:
      children_action = np.argmax(q_target_seq_no_reward.numpy())
    else:
      if split or (iteration > 0 and
                   (iteration % args_dict["target_replacement_freq"] == 0)):
        children_action = sample_action_assignments(
            q_target_seq_no_reward.numpy(), FLAGS.sampling_temperature,
            args_dict["num_children"])
      else:
        children_action = np.argmax(q_target_seq_no_reward.numpy())

    children_action_vec[i, :] = children_action
    children_q_sampled[i] = np.take(q_target_seq.numpy(), children_action)

  return next_states, children_q_sampled, children_action_vec


def update_single_step(batch_feature_vecs, random_state, parent_weight,
                       parent_number, tree_exp_replay, parent_index,
                       old_target_weight, step_iteration, batch, level,
                       parent_score, optimizer, args_dict):
  """Runs a single training step.

  Args:
    batch_feature_vecs: numpy array, batch of feature vectors
    random_state: np.random.RandomState, maintain the random generator state.
    parent_weight: numpy array, the weights of the parent node
    parent_number: int, the indices of the parent node in the tree structure
      buffer
    tree_exp_replay: Instance of ConsistencyBuffer.
    parent_index: int, the indices of the parent node in the tree structure
      buffer
    old_target_weight: numpy array, the previous previous target network weights
    step_iteration: int, the number of iteration step in the training
    batch: bool, whether this is a single batch of data or not
    level: int, the level of ConsistencyBuffer
    parent_score: float, the score of the parent node
    optimizer: Optimizer instance.
    args_dict: dictionary, contains all the necessary configuration params.

  Returns:
    np.ndarray, weights of the child node after applying ConQUR updates.
  """
  is_max_queue_reach = parent_number >= args_dict["max_num_nodes"]
  is_dive_or_expand = (args_dict["split_delay"] >
                       0) and (step_iteration % (args_dict["split_delay"]) != 0)
  split = not (is_max_queue_reach or is_dive_or_expand)
  if split:
    tf.logging.info(f"Iteration = {step_iteration}, doing split")

  # Q-function to score and update. Sample a batch with exploration policy.
  batch_action_vecs = np.zeros((args_dict["steps_per_iteration"], 1),
                               dtype="int")
  # Post-process the data
  extract_sa_features(batch, batch_feature_vecs, batch_action_vecs)
  (next_states, children_q_label,
   children_action_vec) = process_batch_data(batch, split, level,
                                             step_iteration, args_dict)

  # Create q-labels and regress
  children_weights = []
  num_actions = args_dict["num_actions"]
  state_dimensions = args_dict["state_dimensions"]
  for i in range(args_dict["num_children"]):
    children_layer = create_single_layer_from_weights(num_actions,
                                                      state_dimensions,
                                                      parent_weight)
    tree_index = tree_exp_replay.get_foremost_indices() + 1
    children_action_index = i if split else 0
    tree_exp_replay.store_transition(
        tree_index, (next_states, children_action_vec[:, children_action_index],
                     batch_action_vecs), parent_index)

    (cumulative_next_states, cumulative_batch_next_action_vecs,
     _) = tree_exp_replay.get_experience(parent_index + [tree_index])

    total_bellman = 0
    total_regularized_loss = 0
    total_q = 0

    # Start the training
    for _ in range(
        int(
            len(batch_feature_vecs) * 1.0 /
            (FLAGS.train_step_frequency * FLAGS.batch_size))):
      mini_index = random_state.choice(
          len(batch_feature_vecs), FLAGS.batch_size)

      # Get the gradient
      grads, _, bellman_loss, regularized_loss, q_average = grad_loss_fn(
          batch_feature_vecs[mini_index], batch_action_vecs[mini_index],
          children_layer, cumulative_next_states, children_q_label[mini_index],
          cumulative_batch_next_action_vecs, i, args_dict)

      # Apply the gradient
      optimizer.apply_gradients(
          zip(grads, children_layer.variables),
          global_step=tf.train.get_or_create_global_step())
      total_bellman += bellman_loss
      total_regularized_loss += regularized_loss
      total_q += q_average
    children_weights.append(
        (children_layer.layers[0].get_weights(), parent_index + [tree_index],
         parent_weight, old_target_weight, total_bellman,
         total_regularized_loss, total_q, parent_score))
    if not split:
      break

  return children_weights


def run_training(environment, exploration_fn, random_state, args_dict=None):
  """Executes the training loop.

  Args:
    environment: Instance of Environment.
    exploration_fn: function, this can be linear decay or constant exploration
      rate.
    random_state: np.random.RandomState, maintain the random generator state.
    args_dict: dictionary, contains all the necessary configuration params.
  """
  last_layer_weights = environment.last_layer_weights(),
  last_layer_biases = environment.last_layer_biases(),
  last_layer_target_weights = environment.last_layer_target_weights(),
  last_layer_target_biases = environment.last_layer_target_biases(),
  num_actions = args_dict["num_actions"]
  state_dimensions = args_dict["state_dimensions"]

  checkpoint_dir = os.path.join(FLAGS.checkpoint_path, "checkpoint/")
  checkpoint_handler = checkpointer.Checkpointer(
      checkpoint_dir, checkpoint_frequency=CHECKPOINT_FREQUENCY)
  checkpoint_version = checkpointer.get_latest_checkpoint_number(checkpoint_dir)
  if checkpoint_version >= 0:
    checkpoint_data = checkpoint_handler.load_checkpoint(checkpoint_version)
    print(f"Restored checkpoint for iteration {checkpoint_version}.")
    # TODO(dijiasu): Revisit if need to run agent._sync_qt_ops().
  else:
    print("No checkpoint found. Initializing all variables.")
    # Initialize CheckpointData object if there is not checkpoint yet.
    target_weight = [last_layer_target_weights, last_layer_target_biases]
    pre_load = [last_layer_weights, last_layer_biases]
    node_queue = queue.Queue()
    if abs(args_dict["consistency_coeff"]) <= EPS_TOLERANCE:
      for _ in range(args_dict["max_num_nodes"]):
        node_queue.put(
            (pre_load, [0], target_weight, target_weight, 0, 0, 0, 0))
    else:
      node_queue.put((pre_load, [0], target_weight, target_weight, 0, 0, 0, 0))

    checkpoint_data = CheckpointData(
        batch_feature_vecs=np.zeros(
            (args_dict["steps_per_iteration"], state_dimensions), dtype="f8"),
        target_weight=[last_layer_target_weights, last_layer_target_biases],
        tree_exp_replay=ConsistencyBuffer(random_state, args_dict),
        pre_load=pre_load,
        node_queue=node_queue,
        tree_level_size=node_queue.qsize(),
        weights_to_be_rollout=pre_load,
        best_sampler=None,
        back_tracking_count=args_dict["back_tracking_count"],
        back_tracking_node_list=args_dict["back_tracking_node_list"],
        start_i=0,
        optimizer=tf.train.RMSPropOptimizer(
            learning_rate=FLAGS.learning_rate,
            decay=0.95,
            epsilon=0.00001,
            centered=True),
        initial_batch_data=None)

    # Check initial performance.
    # TODO(dijiasu): Conside remove.
    rollout_layer = create_single_layer_from_weights(num_actions,
                                                     state_dimensions, pre_load)
    (avg_actual_return, avg_predicted_return,
     avg_q_val) = environment.evaluate_policy(
         random_state, rollout_layer, epsilon_eval=args_dict["eval_eps"])
    print("initial batch #%d true_val: %.2f predicted_val: %.2f\n", 0,
          avg_actual_return, avg_predicted_return)
    for iteration_children in range(args_dict["max_num_nodes"]):
      with tf.name_scope("children={}_queue={}_lr={}_samp={}".format(
          args_dict["num_children"], args_dict["max_num_nodes"],
          args_dict["learning_rate"], args_dict["sample_consis_buffer"])):
        with tf.name_scope("children_cnt={}".format(iteration_children)):
          tf.compat.v2.summary.scalar(
              "actual_return", avg_actual_return, step=0)
          tf.compat.v2.summary.scalar(
              "predic_return", avg_predicted_return, step=0)
    with tf.name_scope("children={}_queue={}_lr={}_samp={}".format(
        args_dict["num_children"], args_dict["max_num_nodes"],
        args_dict["learning_rate"], args_dict["sample_consis_buffer"])):
      with tf.name_scope("Best!"):
        tf.compat.v2.summary.scalar(
            "best_actual_return", avg_actual_return, step=0)
        tf.compat.v2.summary.scalar("indice", 0, step=0)

  batch_feature_vecs = checkpoint_data.batch_feature_vecs
  target_weight = checkpoint_data.target_weight
  tree_exp_replay = checkpoint_data.tree_exp_replay
  pre_load = checkpoint_data.pre_load
  q = checkpoint_data.node_queue
  tree_level_size = checkpoint_data.tree_level_size
  weights_to_be_rollout = checkpoint_data.weights_to_be_rollout
  best_sampler = checkpoint_data.best_sampler
  back_tracking_count = checkpoint_data.back_tracking_count
  back_tracking_node_list = checkpoint_data.back_tracking_node_list
  start_i = checkpoint_data.start_i
  optimizer = checkpoint_data.optimizer
  initial_batch_data = checkpoint_data.initial_batch_data
  for i in range(start_i, FLAGS.training_iterations):
    print(f"Starting iteration {i}.")
    level_weights = []
    target_layer_list = []
    for level in range(tree_level_size):
      level_q = q.get()
      (parent_weight, parent_index, target_weight, old_target_weight, _, _, _,
       _) = level_q
      old_target = create_single_layer_from_weights(num_actions,
                                                    state_dimensions,
                                                    old_target_weight)
      target_layer_list.append(old_target)
      q.put(level_q)

    # Launch the agent and sample one batch of data from the env
    if best_sampler is not None:
      single_batch = exploration_fn(
          random_state,
          create_single_layer_from_weights(num_actions, state_dimensions,
                                           best_sampler), target_layer_list)
    else:
      single_batch = exploration_fn(
          random_state,
          create_single_layer_from_weights(num_actions, state_dimensions,
                                           pre_load), target_layer_list)
    if initial_batch_data is None:
      initial_batch_data = process_initial_batch(single_batch, args_dict)

    for level in range(tree_level_size):
      (parent_weight, parent_index, target_weight, old_target_weight, _, _, _,
       parent_score) = q.get()

      # We update the target weights
      if i > 0 and i % args_dict["target_replacement_freq"] == 0:
        old_target_weight = target_weight

      # We are doing Q-update here, and split the parent nodes into multiple
      # child nodes
      children_weights = update_single_step(batch_feature_vecs, random_state,
                                            parent_weight, tree_level_size,
                                            tree_exp_replay, parent_index,
                                            old_target_weight, i, single_batch,
                                            level, parent_score, optimizer,
                                            args_dict)

      for children_w in children_weights:
        q.put(children_w)
        level_weights.append(children_w)

    # Advance the experience buffer
    tree_exp_replay.next_level()
    # Deleting previous level experience
    tree_exp_replay.forget()
    tree_level_size = q.qsize()

    if i % args_dict["rollout_freq"] == 0:
      children_cnt = 0
      actual_return_ls = []
      children_score = []
      children_loss_bellman = []
      children_loss_regularize = []
      children_loss_total = []
      parent_loss_vector = []
      children_avg_q = []

      num_best_nodes_for_expansion = args_dict["num_best_nodes_for_expansion"]
      for children_w in level_weights:
        with tf.name_scope("children={}_queue={}_lr={}_samp={}".format(
            args_dict["num_children"], args_dict["max_num_nodes"],
            args_dict["learning_rate"], args_dict["sample_consis_buffer"])):
          with tf.name_scope("children_cnt={}".format(children_cnt)):
            (chid_weight_np, _, _, _, total_bellman, total_regularized_loss,
             total_q, parent_score) = children_w
            weights_to_be_rollout = chid_weight_np
            children_loss_bellman.append(total_bellman)
            children_loss_regularize.append(total_regularized_loss)
            children_loss_total.append(total_bellman + total_regularized_loss)
            children_avg_q.append(total_q -
                                  (total_bellman + total_regularized_loss))
            parent_loss_vector.append(parent_score)
            rollout_layer = create_single_layer_from_weights(
                num_actions, state_dimensions, weights_to_be_rollout)

            # Evaluate the policy
            (avg_actual_return, avg_predicted_return,
             avg_q_val) = environment.evaluate_policy(
                 random_state,
                 rollout_layer,
                 epsilon_eval=args_dict["eval_eps"])

            tf.compat.v2.summary.scalar(
                "actual_return", avg_actual_return, step=i + 1)
            tf.compat.v2.summary.scalar(
                "predic_return", avg_predicted_return, step=i + 1)
            tf.compat.v2.summary.scalar("avg_q", avg_q_val, step=i + 1)
            actual_return_ls.append(avg_actual_return)
            children_cnt += 1
            tf.logging.info("batch #%d true_val: %.2f predicted_val: %.2f\n", i,
                            avg_actual_return, avg_predicted_return)
            children_score.append(avg_actual_return)

      children_loss_bellman = np.array(children_loss_bellman)
      children_loss_regularize = np.array(children_loss_regularize)
      children_loss_total = np.array(children_loss_total)
      parent_loss_vector = np.array(parent_loss_vector)
      children_avg_q = np.array(children_avg_q)
      children_score = np.array(children_score)
      children_score_idx = children_score.argsort(
      )[-num_best_nodes_for_expansion:][::-1]
      chosen_children = []

      # Choose scoring function for selecting which nodes to expand
      if args_dict["node_scoring_fn"] == "rollouts":
        # This uses the rollouts as the scoring function
        if (i > 0) and (args_dict["consistency_coeff"] >
                        0) and args_dict["enable_back_tracking"]:
          for track_cell in back_tracking_node_list:
            np.append(children_score, track_cell[-1])
        children_score_idx = children_score.argsort(
        )[-num_best_nodes_for_expansion:][::-1]
        chosen_children = children_score.argsort()[::-1]

      elif args_dict["node_scoring_fn"] == "bellman_consistency":
        # Using bellman_consistency as the scoring function
        if (i > 0) and (args_dict["consistency_coeff"] >
                        0) and args_dict["enable_back_tracking"]:
          mean_loss_total_delta = np.mean(children_loss_total -
                                          parent_loss_vector)
          for track_cell in range(len(back_tracking_node_list)):
            back_tracking_node_list[track_cell][-1] = back_tracking_node_list[
                track_cell][-1] + mean_loss_total_delta * BACK_TRACK_CALIBRATION
            children_loss_total = np.append(
                children_loss_total, back_tracking_node_list[track_cell][-1])
        children_score_idx = children_loss_total.argsort(
        )[:num_best_nodes_for_expansion]
        chosen_children = children_loss_total.argsort()

      try:
        if children_score_idx[0] >= args_dict["max_num_nodes"]:
          best_sampler = back_tracking_node_list[children_score_idx[0] -
                                                 args_dict["max_num_nodes"]][0]
      except IndexError as e:
        tf.logging.error(e)
      if i > 0 and args_dict["consistency_coeff"] > 0:
        for queue_iteration in range(len(level_weights)):
          q.get()
        tf.logging.info("Pruning nodes...")
        if args_dict["enable_back_tracking"]:
          for queue_iteration in range(len(level_weights)):
            level_weights[queue_iteration] = list(
                level_weights[queue_iteration])
            if args_dict["node_scoring_fn"] == "rollouts":
              level_weights[queue_iteration][-1] = children_score[
                  queue_iteration]
            elif args_dict["node_scoring_fn"] == "bellman_consistency":
              level_weights[queue_iteration][-1] = level_weights[
                  queue_iteration][-4] + level_weights[queue_iteration][-3]
            level_weights[queue_iteration] = tuple(
                level_weights[queue_iteration])
        # Queue_iteration is the indices that perform the best according to
        # Chosen scoring function
        for queue_iteration in children_score_idx:
          if args_dict["enable_back_tracking"]:
            if queue_iteration >= args_dict["max_num_nodes"]:
              adjust_idx = queue_iteration - args_dict["max_num_nodes"]
              q.put(tuple(back_tracking_node_list[adjust_idx]))
              back_tracking_count += 1
            else:
              q.put(level_weights[queue_iteration])
          else:
            q.put(level_weights[queue_iteration])
        # Pick the highest scoring nodes
        if args_dict["enable_back_tracking"]:
          tmp_back_tracking_node_list = []
          for idx__ in chosen_children[num_best_nodes_for_expansion:]:
            if idx__ >= args_dict["max_num_nodes"]:
              adjust_idx = idx__ - args_dict["max_num_nodes"]
              tmp_back_tracking_node_list.append(
                  back_tracking_node_list[adjust_idx])
            else:
              tmp_back_tracking_node_list.append(list(level_weights[idx__]))

            if len(tmp_back_tracking_node_list) >= args_dict["back_track_size"]:
              break
          back_tracking_node_list = tmp_back_tracking_node_list
        tree_level_size = num_best_nodes_for_expansion

      with tf.name_scope("children={}_queue={}_lr={}_samp={}".format(
          args_dict["num_children"], args_dict["max_num_nodes"],
          args_dict["learning_rate"], args_dict["sample_consis_buffer"])):
        with tf.name_scope("Best!"):
          tf.compat.v2.summary.scalar(
              "best_actual_return", np.max(actual_return_ls), step=i + 1)
          # TODO(dijiasu): Check if renaming from indice to index breaks this.
          tf.compat.v2.summary.scalar(
              "indice", np.argmax(actual_return_ls), step=i + 1)
    if i == 0:
      tf.logging.info("Copying the online network weights to target network.")
      # TODO(dijiasu): Revisit if need to run agent._sync_qt_ops().

    # TODO(dijiasu): Keep checkpoint_data variable to avoid re-creating it.
    checkpoint_data = CheckpointData(
        batch_feature_vecs=batch_feature_vecs,
        target_weight=target_weight,
        tree_exp_replay=tree_exp_replay,
        pre_load=pre_load,
        q=q,
        tree_level_size=tree_level_size,
        weights_to_be_rollout=weights_to_be_rollout,
        best_sampler=best_sampler,
        back_tracking_count=back_tracking_count,
        back_tracking_node_list=back_tracking_node_list,
        start_i=i + 1,
        optimizer=optimizer,
        initial_batch_data=initial_batch_data,
    )
    checkpoint_handler.save_checkpoint(i, checkpoint_data)
    args_dict["back_tracking_count"] = back_tracking_count
    args_dict["back_tracking_node_list"] = back_tracking_node_list


def run_atari():
  """Main entry point for learning Atari."""
  tf.random.set_random_seed(FLAGS.random_seed)
  random_state = np.random.RandomState(seed=FLAGS.random_seed)
  # TODO(dijiasu): Fix hard coding.
  environment = atari.Atari(FLAGS.atari_roms_source, FLAGS.atari_roms_path,
                            FLAGS.gin_files, FLAGS.gin_bindings,
                            FLAGS.random_seed, FLAGS.no_op, True)
  args_dict = {
      "atari_game": FLAGS.atari_game,
      "enable_back_tracking": FLAGS.enable_back_tracking,
      "sample_consis_buffer": FLAGS.sample_consis_buffer,
      "max_num_nodes": FLAGS.max_num_nodes,
      "num_children": FLAGS.num_children,
      "learning_rate": FLAGS.learning_rate,
      "consistency_coeff": FLAGS.consistency_coeff,
      "eval_eps": FLAGS.eval_eps,
      "target_replacement_freq": FLAGS.target_replacement_freq,
      "rollout_freq": FLAGS.rollout_freq,
      "num_best_nodes_for_expansion": FLAGS.num_best_nodes_for_expansion,
      "split_delay": FLAGS.split_delay,
      "steps_per_iteration": FLAGS.steps_per_iteration,
      "node_scoring_fn": FLAGS.node_scoring_fn,
      "back_track_size": FLAGS.back_track_size,
      "state_dimensions": atari.STATE_DIM - 1,
      "num_actions": environment.num_actions,
      "back_tracking_count": 0,
      "back_tracking_node_list": [],
  }

  def exploration_fn(random_state, parent_layer, target_layer):
    return environment.sample_batch(FLAGS.train_eps, FLAGS.steps_per_iteration,
                                    random_state, parent_layer, target_layer)

  writer = tf.compat.v2.summary.create_file_writer(
      FLAGS.tfboard_path, flush_millis=100, name="testing")
  writer.set_as_default()
  print("Starting ConQUR training...")
  print(f"queue size = {FLAGS.max_num_nodes}, children = {FLAGS.num_children}.")
  run_training(environment, exploration_fn, random_state, args_dict=args_dict)


def main(_):
  tf.compat.v1.enable_eager_execution()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  logger = tf.get_logger()
  logger.setLevel(tf.logging.ERROR)
  np.set_printoptions(suppress=True, precision=6, linewidth=120)
  run_atari()


if __name__ == "__main__":
  app.run(main)
